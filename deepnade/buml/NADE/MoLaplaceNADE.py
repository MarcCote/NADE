# A NADE that has for output distribution a Mixture of Gaussians
from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from ParameterInitialiser import Gaussian
from Model import *
from NADE import MixtureNADE
from Utils.theano_helpers import log_sum_exp, constantX, floatX
import Utils


class MoLaplaceNADE(MixtureNADE):
    def __init__(self, n_visible, n_hidden, n_components, nonlinearity="RLU"):
        MixtureNADE.__init__(self, n_visible, n_hidden, n_components, nonlinearity)
        self.add_parameter(TensorParameter("W", (n_visible, n_hidden)), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("b_alpha", (n_visible, n_components)), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V_alpha", (n_visible, n_hidden, n_components)), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("b_mu", (n_visible, n_components)), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V_mu", (n_visible, n_hidden, n_components)), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("b_sigma", (n_visible, n_components)), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V_sigma", (n_visible, n_hidden, n_components)), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("activation_rescaling", (n_visible)), optimise=True, regularise=False)
        self.recompile()

    def initialize_parameters(self, W_initialiser=Gaussian(std=0.01), b_initialiser=Gaussian(std=0.01)):
        for p in [self.W, self.V_mu, self.V_sigma, self.V_alpha]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(theano.config.floatX))
        for p in [self.b_mu, self.b_sigma, self.b_alpha]:
            p.set_value(b_initialiser.get_tensor(p.get_value().shape).astype(theano.config.floatX))
        self.b_sigma.set_value(self.b_sigma.get_value() + 1.0)
        self.activation_rescaling.set_value(np.ones(self.n_visible, dtype=theano.config.floatX))

    def initialize_parameters_cover_domain(self, domains, W_initialiser=Gaussian(std=0.01)):
        self.activation_rescaling.set_value(np.ones(self.n_visible, dtype=theano.config.floatX))
        for p in [self.W, self.V_mu, self.V_sigma, self.V_alpha]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(theano.config.floatX))
        b_alpha = np.zeros(self.b_alpha.get_value().shape, dtype=theano.config.floatX)
        b_mu = np.zeros(self.b_mu.get_value().shape, dtype=theano.config.floatX)
        b_sigma = np.zeros(self.b_sigma.get_value().shape, dtype=theano.config.floatX)
        for i, (a, b) in enumerate(domains):
            s = (b - a) / (self.n_components + 1)
            b_mu[i] = np.arange(1, self.n_components + 1) * s + a
            b_sigma[i] = np.log(s)
        self.b_alpha.set_value(b_alpha)
        self.b_mu.set_value(b_mu)
        self.b_sigma.set_value(b_sigma)

    def initialize_parameters_from_dataset(self, dataset, W_initialiser=Gaussian(std=0.01), sample_size):
        self.activation_rescaling.set_value(np.ones(self.n_visible, dtype=theano.config.floatX))
        for p in [self.W, self.V_mu, self.V_sigma, self.V_alpha]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(theano.config.floatX))
        b_alpha = np.zeros(self.b_alpha.get_value().shape, dtype=theano.config.floatX)
        b_mu = np.zeros(self.b_mu.get_value().shape, dtype=theano.config.floatX)
        b_sigma = np.zeros(self.b_sigma.get_value().shape, dtype=theano.config.floatX)
        data_sample = dataset.sample_data(sample_size)[0].astype(floatX)
        domains = zip(data_sample.min(axis=0), data_sample.max(axis=0))
        for i, (a, b) in enumerate(domains):
            s = (b - a) / (self.n_components + 1)
            b_mu[i] = np.arange(1, self.n_components + 1) * s + a
            b_sigma[i] = np.log(s)
        self.b_alpha.set_value(b_alpha)
        self.b_mu.set_value(b_mu)
        self.b_sigma.set_value(b_sigma)

    def sym_logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, V_alpha, b_alpha, V_mu, b_mu, V_sigma, b_sigma, activations_factor, p_prev, a_prev, x_prev):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            h = self.nonlinearity(a * activations_factor)  # BxH

            Alpha = T.nnet.softmax(T.dot(h, V_alpha) + T.shape_padleft(b_alpha))  # BxC
            Mu = T.dot(h, V_mu) + T.shape_padleft(b_mu)  # BxC
            Sigma = T.exp((T.dot(h, V_sigma) + T.shape_padleft(b_sigma)))  # BxC
            p = p_prev + log_sum_exp(T.log(Alpha) - T.log(2 * Sigma) - T.abs_(Mu - T.shape_padright(x, 1)) / Sigma)
            return (p, a, x)
        # First element is different (it is predicted from the bias only)
        a0 = T.zeros_like(T.dot(x.T, self.W))  # BxH
        p0 = T.zeros_like(x[0])
        x0 = T.ones_like(x[0])
        ([ps, _as, _xs], updates) = theano.scan(density_given_previous_a_and_x,
                                                sequences=[x, self.W, self.V_alpha, self.b_alpha, self.V_mu, self.b_mu, self.V_sigma, self.b_sigma, self.activation_rescaling],
                                                outputs_info=[p0, a0, x0])
        return (ps[-1], updates)

    def sym_gradients_new(self, X):
        non_linearity_name = self.parameters["nonlinearity"].get_name()
        assert(non_linearity_name == "sigmoid" or non_linearity_name == "RLU")
        # First element is different (it is predicted from the bias only)
        init_a = T.zeros_like(T.dot(X.T, self.W))  # BxH
        init_x = T.ones_like(X[0])

        def a_i_given_a_im1(x, w, a_prev, x_prev):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            return (a, x)
        ([As, _], updates) = theano.scan(a_i_given_a_im1, sequences=[X, self.W], outputs_info=[init_a, init_x])
        top_activations = As[-1]
        Xs_m1 = T.set_subtensor(X[1:, :], X[0:-1, :])
        Xs_m1 = T.set_subtensor(Xs_m1[0, :], 1)

        # Reconstruct the previous activations and calculate (for that visible dimension) the density and all the gradients
        def density_and_gradients(x_i, x_im1, w_i, V_alpha, b_alpha, V_mu, b_mu, V_sigma, b_sigma, activation_factor, a_i, lp_accum, dP_da_ip1):
            B = T.cast(x_i.shape[0], theano.config.floatX)
            pot = a_i * activation_factor
            h = self.nonlinearity(pot)  # BxH

            z_alpha = T.dot(h, V_alpha) + T.shape_padleft(b_alpha)
            z_mu = T.dot(h, V_mu) + T.shape_padleft(b_mu)
            z_sigma = T.dot(h, V_sigma) + T.shape_padleft(b_sigma)

            Alpha = T.nnet.softmax(z_alpha)  # BxC
            Mu = z_mu  # BxC
            Sigma = T.exp(z_sigma)  # BxC

            Phi = -T.log(2 * Sigma) - T.abs_(Mu - T.shape_padright(x_i, 1)) / Sigma
            wPhi = T.maximum(Phi + T.log(Alpha), constantX(-100.0))

            lp_current = log_sum_exp(wPhi)
            # lp_current_sum = T.sum(lp_current)

            Pi = T.exp(wPhi - T.shape_padright(lp_current, 1))  # #
            dp_dz_alpha = Pi - Alpha  # BxC
            # dp_dz_alpha = T.grad(lp_current_sum, z_alpha)
            gb_alpha = dp_dz_alpha.mean(0, dtype=theano.config.floatX)  # C
            gV_alpha = T.dot(h.T, dp_dz_alpha) / B  # HxC

            # dp_dz_mu = T.grad(lp_current_sum, z_mu)
            dp_dz_mu = Pi * T.sgn(T.shape_padright(x_i, 1) - Mu) / Sigma
            # dp_dz_mu = dp_dz_mu * Sigma
            gb_mu = dp_dz_mu.mean(0, dtype=theano.config.floatX)
            gV_mu = T.dot(h.T, dp_dz_mu) / B

            # dp_dz_sigma = T.grad(lp_current_sum, z_sigma)
            dp_dz_sigma = Pi * (T.abs_(T.shape_padright(x_i, 1) - Mu) / Sigma - 1)
            gb_sigma = dp_dz_sigma.mean(0, dtype=theano.config.floatX)
            gV_sigma = T.dot(h.T, dp_dz_sigma) / B

            dp_dh = T.dot(dp_dz_alpha, V_alpha.T) + T.dot(dp_dz_mu, V_mu.T) + T.dot(dp_dz_sigma, V_sigma.T)  # BxH
            if non_linearity_name == "sigmoid":
                dp_dpot = dp_dh * h * (1 - h)
            elif non_linearity_name == "RLU":
                dp_dpot = dp_dh * (pot > 0)

            gfact = (dp_dpot * a_i).sum(1).mean(0, dtype=theano.config.floatX)  # 1

            dP_da_i = dP_da_ip1 + dp_dpot * activation_factor  # BxH
            gW = T.dot(T.shape_padleft(x_im1, 1), dP_da_i).flatten() / B

            return (a_i - T.dot(T.shape_padright(x_im1, 1), T.shape_padleft(w_i, 1)),
                    lp_accum + lp_current,
                    dP_da_i,
                    gW, gb_alpha, gV_alpha, gb_mu, gV_mu, gb_sigma, gV_sigma, gfact)

        p_accum = T.zeros_like(X[0])
        dP_da_ip1 = T.zeros_like(top_activations)
        ([_, ps, _, gW, gb_alpha, gV_alpha, gb_mu, gV_mu, gb_sigma, gV_sigma, gfact], updates2) = theano.scan(density_and_gradients,
                                                go_backwards=True,
                                                sequences=[X, Xs_m1, self.W, self.V_alpha, self.b_alpha, self.V_mu, self.b_mu, self.V_sigma, self.b_sigma, self.activation_rescaling],
                                                outputs_info=[top_activations, p_accum, dP_da_ip1, None, None, None, None, None, None, None, None])
        # scan with go_backwards returns the matrices in the order they were created, so we have to reverse the order of the rows
        gW = gW[::-1, :]
        gb_alpha = gb_alpha[::-1, :]
        gV_alpha = gV_alpha[::-1, :, :]
        gb_mu = gb_mu[::-1, :]
        gV_mu = gV_mu[::-1, :, :]
        gb_sigma = gb_sigma[::-1, :]
        gV_sigma = gV_sigma[::-1, :, :]
        gfact = gfact[::-1]

        updates.update(updates2)  # Returns None
        return (ps[-1], gW, gb_alpha, gV_alpha, gb_mu, gV_mu, gb_sigma, gV_sigma, gfact, updates)

    def sample(self, n):
        W = self.W.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()
        activation_rescaling = self.activation_rescaling.get_value()
        samples = np.zeros((self.n_visible, n))
        for s in xrange(n):
            a = np.zeros((self.n_hidden,))  # H
            for i in xrange(self.n_visible):
                if i == 0:
                    a = W[i, :]
                else:
                    a = a + W[i, :] * samples[i - 1, s]
                h = self.parameters["nonlinearity"].get_numpy_f()(a * activation_rescaling[i])
                alpha = Utils.nnet.softmax(np.dot(h, V_alpha[i]) + b_alpha[i])  # C
                Mu = np.dot(h, V_mu[i]) + b_mu[i]  # C
                # Sigma = np.minimum(np.exp(np.dot(h, V_sigma[i]) + b_sigma[i]), 1)
                Sigma = np.exp(np.dot(h, V_sigma[i]) + b_sigma[i])
                comp = Utils.nnet.random_component(alpha)
                samples[i, s] = np.random.laplace(Mu[comp], Sigma[comp])
        return samples

    def conditional_logdensities(self, x_lt_i, range):
        raise(Exception("Not implemented"))
        W = self.W.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()
        activation_rescaling = self.activation_rescaling.get_value()
        # Calculate
        i = len(x_lt_i)
        a = W[0, :] + np.dot(x_lt_i, W[1:len(x_lt_i) + 1, :])
        h = self.parameters["nonlinearity"].get_numpy_f()(a * activation_rescaling[i])
        alpha = Utils.nnet.softmax(np.tanh(np.dot(h, V_alpha[i]) + b_alpha[i]) * 10.0)  # C
        Mu = np.dot(h, V_mu[i]) + b_mu[i]  # C
        Sigma = np.log(1.0 + np.exp((np.dot(h, V_sigma[i]) + b_sigma[i]) * 10)) / 10  # C

        def ld(x):
            lds = np.array([scipy.stats.norm.logpdf(x, Mu[c], Sigma[c]) for c in xrange(self.n_components)])
            return Utils.nnet.logsumexp(lds + np.log(alpha))
        return np.array([ld(x) for x in range])

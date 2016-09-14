# A NADE that has Bernoullis for output distribution
from __future__ import division
import theano
import theano.tensor as T
import numpy as np
from itertools import izip
from ParameterInitialiser import Gaussian
from Model import SizeParameter, TensorParameter
from NADE import NADE
import Utils
from Utils.theano_helpers import log_sum_exp, constantX, floatX
from Utils.nnet import logsumexp
from Utils.Estimation import Estimation


class OrderlessMoGNADE(NADE):
    def __init__(self, n_visible, n_hidden, n_layers, n_components, nonlinearity="RLU"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(SizeParameter("n_layers"))
        self.n_layers = n_layers
        self.add_parameter(SizeParameter("n_components"))
        self.n_components = n_components
        self.add_parameter(TensorParameter("Wflags", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("W1", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("b1", (n_hidden), theano=True), optimise=True, regularise=False)
        if self.n_layers > 1:
            self.add_parameter(TensorParameter("Ws", (n_layers, n_hidden, n_hidden), theano=True), optimise=True, regularise=True)
            self.add_parameter(TensorParameter("bs", (n_layers, n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V_mu", (n_visible, n_hidden, n_components), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("b_mu", (n_visible, n_components), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V_sigma", (n_visible, n_hidden, n_components), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("b_sigma", (n_visible, n_components), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V_alpha", (n_visible, n_hidden, n_components), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("b_alpha", (n_visible, n_components), theano=True), optimise=True, regularise=False)
        self.setup_n_orderings(1)
        self.recompile()

    @classmethod
    def create_from_params(cls, params):
        n_visible, n_hidden, n_layers, n_components = (params["n_visible"], params["n_hidden"], params["n_layers"], params["n_components"])
        model = cls(n_visible, n_hidden, n_layers, n_components, params["nonlinearity"])
        model.set_parameters(params)
        return model

    @classmethod
    def create_from_smaller_NADE(cls, small_NADE, add_n_hiddens=1, W_initialiser=Gaussian(std=0.01), domains=None):
        n_visible, n_hidden, n_layers, n_components, nonlinearity = (small_NADE.n_visible, small_NADE.n_hidden, small_NADE.n_layers, small_NADE.n_components, small_NADE.parameters["nonlinearity"].get_name())
        model = cls(n_visible, n_hidden, n_layers + add_n_hiddens, n_components, nonlinearity)
        # Copy first layer
        model.Wflags.set_value(small_NADE.Wflags.get_value())
        model.W1.set_value(small_NADE.W1.get_value())
        model.b1.set_value(small_NADE.b1.get_value())
        # Copy the hidden layers from the smaller NADE and initialise the rest
        Ws = W_initialiser.get_tensor(model.Ws.get_value().shape)
        bs = W_initialiser.get_tensor(model.bs.get_value().shape)
        if n_layers > 1:
            Ws[0:n_layers - 1, :, :] = small_NADE.Ws.get_value()[0:n_layers - 1, :, :]
            bs[0:n_layers - 1, :] = small_NADE.bs.get_value()[0:n_layers - 1, :]
        model.Ws.set_value(Ws)
        model.bs.set_value(bs)
        for p in [model.V_mu, model.V_sigma, model.V_alpha]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        if domains is None:
            model.b_alpha.set_value(small_NADE.b_alpha.get_value())
            model.b_mu.set_value(small_NADE.b_mu.get_value())
            model.b_sigma.set_value(small_NADE.b_sigma.get_value())
        else:
            b_alpha = np.zeros(model.b_alpha.get_value().shape, dtype=floatX)
            b_mu = np.zeros(model.b_mu.get_value().shape, dtype=floatX)
            b_sigma = np.zeros(model.b_sigma.get_value().shape, dtype=floatX)
            for i, (a, b) in enumerate(domains):
                s = (b - a) / (model.n_components + 1)
                b_mu[i] = np.arange(1, model.n_components + 1) * s + a
                b_sigma[i] = s
            model.b_alpha.set_value(b_alpha)
            model.b_mu.set_value(b_mu)
            model.b_sigma.set_value(b_sigma)
        return model

    def initialize_parameters_cover_domain(self, domains, W_initialiser=Gaussian(std=0.01)):
        self.Wflags.set_value(W_initialiser.get_tensor(self.Wflags.get_value().shape))
        self.W1.set_value(W_initialiser.get_tensor(self.W1.get_value().shape))
        self.b1.set_value(W_initialiser.get_tensor(self.b1.get_value().shape))
        if self.n_layers > 1:
            self.Ws.set_value(W_initialiser.get_tensor(self.Ws.get_value().shape))
            self.bs.set_value(W_initialiser.get_tensor(self.bs.get_value().shape))
        for p in [self.V_mu, self.V_sigma, self.V_alpha]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        b_alpha = np.zeros(self.b_alpha.get_value().shape, dtype=floatX)
        b_mu = np.zeros(self.b_mu.get_value().shape, dtype=floatX)
        b_sigma = np.zeros(self.b_sigma.get_value().shape, dtype=floatX)
        for i, (a, b) in enumerate(domains):
            s = (b - a) / (self.n_components + 1)
            b_mu[i] = np.arange(1, self.n_components + 1) * s + a
            b_sigma[i] = s
        self.b_alpha.set_value(b_alpha)
        self.b_mu.set_value(b_mu)
        self.b_sigma.set_value(b_sigma)

    def initialize_parameters_from_dataset(self, dataset, W_initialiser=Gaussian(std=0.01), sample_size=10000):
        self.Wflags.set_value(W_initialiser.get_tensor(self.Wflags.get_value().shape))
        self.W1.set_value(W_initialiser.get_tensor(self.W1.get_value().shape))
        self.b1.set_value(W_initialiser.get_tensor(self.b1.get_value().shape))
        if self.n_layers > 1:
            self.Ws.set_value(W_initialiser.get_tensor(self.Ws.get_value().shape))
            self.bs.set_value(W_initialiser.get_tensor(self.bs.get_value().shape))
        for p in [self.V_mu, self.V_sigma, self.V_alpha]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        b_alpha = np.zeros(self.b_alpha.get_value().shape, dtype=floatX)
        b_mu = np.zeros(self.b_mu.get_value().shape, dtype=floatX)
        b_sigma = np.zeros(self.b_sigma.get_value().shape, dtype=floatX)
        data_sample = dataset.sample_data(sample_size)[0].astype(floatX)
        domains = zip(data_sample.min(axis=0), data_sample.max(axis=0))
        for i, (a, b) in enumerate(domains):
            s = (b - a) / (self.n_components + 1)
            b_mu[i] = np.arange(1, self.n_components + 1) * s + a
            b_sigma[i] = s
        self.b_alpha.set_value(b_alpha)
        self.b_mu.set_value(b_mu)
        self.b_sigma.set_value(b_sigma)

    def recompile(self):
        x = T.matrix('x', dtype=floatX)
        m = T.matrix('m', dtype=floatX)
        logdensity = self.sym_mask_logdensity_estimator(x, m)
        self.compiled_mask_logdensity_estimator = theano.function([x, m], logdensity, allow_input_downcast=True)
#         par = "V_mu"
#         _, g = self.sym_gradient_mask_logdensity_auto(x, m)
#         self.compiled_gradient_mask_logdensity_auto = theano.function([x, m], g[par], allow_input_downcast=True)
#
#         _, g = self.sym_gradient_mask_logdensity(x, m)
#         self.compiled_gradient_mask_logdensity = theano.function([x, m], g[par], allow_input_downcast=True)

    def setup_n_orderings(self, n=None, orderings=None):
        assert(not (n is None and orderings is None))
        self.orderings = list()
        if orderings is not None:
            self.orderings = orderings
            self.n_orderings = len(orderings)
        else:
            self.n_orderings = n
            from copy import copy
            for _ in xrange(self.n_orderings):
                o = range(self.n_visible)
                np.random.shuffle(o)
                self.orderings.append(copy(o))

    def set_ordering(self, ordering):
        self.setup_n_orderings(orderings=[ordering])

    def logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        B = x.shape[1]
        nl = self.parameters["nonlinearity"].get_numpy_f()
        lp = np.zeros((B, self.n_orderings))

        W1 = self.W1.get_value()
        b1 = self.b1.get_value()
        Wflags = self.Wflags.get_value()
        if self.n_layers > 1:
            Ws = self.Ws.get_value()
            bs = self.bs.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()

        for o_index, o in enumerate(self.orderings):
            a = np.zeros((B, self.n_hidden)) + b1
            for j in xrange(self.n_visible):
                i = o[j]
                h = nl(a)
                for l in xrange(self.n_layers - 1):
                    h = nl(np.dot(h, Ws[l]) + bs[l])
                z_alpha = np.dot(h, V_alpha[i]) + b_alpha[i]
                z_mu = np.dot(h, V_mu[i]) + b_mu[i]
                z_sigma = np.dot(h, V_sigma[i]) + b_sigma[i]
                Alpha = Utils.nnet.softmax(z_alpha)  # C
                Mu = z_mu  # C
                Sigma = np.exp(z_sigma)
                lp[:, o_index] += logsumexp(-0.5 * ((Mu - x[i][:, np.newaxis]) / Sigma) ** 2 - np.log(Sigma) - 0.5 * np.log(2 * np.pi) + np.log(Alpha))
                if j < self.n_visible - 1:
                    a += np.dot(x[i][:, np.newaxis], W1[i][np.newaxis, :]) + Wflags[i]
        return logsumexp(lp + np.log(1 / self.n_orderings))

    def estimate_average_loglikelihood_for_dataset_using_masks(self, x_dataset, masks_dataset, minibatch_size=20000, loops=1):
        loglikelihood = 0.0
        loglikelihood_sq = 0.0
        n = 0
        x_iterator = x_dataset.iterator(batch_size=minibatch_size, get_smaller_final_batch=True)
        m_iterator = masks_dataset.iterator(batch_size=minibatch_size)
        for _ in xrange(loops):
            for x, m in izip(x_iterator, m_iterator):
                x = x.T  # VxB
                batch_size = x.shape[1]
                m = m.T[:, :batch_size]
                n += batch_size
                lls = self.compiled_mask_logdensity_estimator(x, m)
                loglikelihood += np.sum(lls)
                loglikelihood_sq += np.sum(lls ** 2)
        return Estimation.sample_mean_from_sum_and_sum_sq(loglikelihood, loglikelihood_sq, n)

    def sym_mask_logdensity_estimator_intermediate(self, x, mask):
        non_linearity_name = self.parameters["nonlinearity"].get_name()
        assert(non_linearity_name == "sigmoid" or non_linearity_name == "RLU")
        x = x.T  # BxD
        mask = mask.T  # BxD
        output_mask = constantX(1) - mask  # BxD
        D = constantX(self.n_visible)
        d = mask.sum(1)  # d is the 1-based index of the dimension whose value to infer (not the size of the context)
        masked_input = x * mask  # BxD
        h = self.nonlinearity(T.dot(masked_input, self.W1) + T.dot(mask, self.Wflags) + self.b1)  # BxH
        for l in xrange(self.n_layers - 1):
            h = self.nonlinearity(T.dot(h, self.Ws[l]) + self.bs[l])  # BxH
        z_alpha = T.tensordot(h, self.V_alpha, [[1], [1]]) + T.shape_padleft(self.b_alpha)
        z_mu = T.tensordot(h, self.V_mu, [[1], [1]]) + T.shape_padleft(self.b_mu)
        z_sigma = T.tensordot(h, self.V_sigma, [[1], [1]]) + T.shape_padleft(self.b_sigma)
        temp = T.exp(z_alpha)  # + 1e-6
        # temp += T.shape_padright(temp.sum(2)/1e-3)
        Alpha = temp / T.shape_padright(temp.sum(2))  # BxDxC
        Mu = z_mu  # BxDxC
        Sigma = T.exp(z_sigma)  # + 1e-6 #BxDxC

        # Alpha = Alpha * T.shape_padright(output_mask) + T.shape_padright(mask)
        # Mu = Mu * T.shape_padright(output_mask)
        # Sigma = Sigma * T.shape_padright(output_mask) + T.shape_padright(mask)
        # Phi = -constantX(0.5) * T.sqr((Mu - T.shape_padright(x*output_mask)) / Sigma) - T.log(Sigma) - constantX(0.5 * np.log(2*np.pi)) #BxDxC

        Phi = -constantX(0.5) * T.sqr((Mu - T.shape_padright(x)) / Sigma) - T.log(Sigma) - constantX(0.5 * np.log(2 * np.pi))  # BxDxC
        logdensity = (log_sum_exp(Phi + T.log(Alpha), axis=2) * output_mask).sum(1) * D / (D - d)
        return (logdensity, z_alpha, z_mu, z_sigma, Alpha, Mu, Sigma, h)

    def sym_mask_logdensity_estimator(self, x, mask):
        """ x is a matrix of column datapoints (DxB) D = n_visible, B = batch size """
        logdensity, z_alpha, z_mu, z_sigma, Alpha, Mu, Sigma, top_h = self.sym_mask_logdensity_estimator_intermediate(x, mask)
        return logdensity

    def sym_masked_neg_loglikelihood_gradient_(self, x, mask):
        loglikelihood = self.sym_mask_logdensity_estimator(x, mask)
        loss = -loglikelihood.mean()
        # Gradients
        gradients = {}
        for param in self.parameters_to_optimise:
            gradients[param] = T.grad(loss, self.__getattribute__(param))
        return (loss, gradients)

    def sym_masked_neg_loglikelihood_gradient(self, x, mask):
        """ x is a matrix of column datapoints (DxB) D = n_visible, Bfloat = batch size """
        logdensity, z_alpha, z_mu, z_sigma, Alpha, Mu, Sigma, h = self.sym_mask_logdensity_estimator_intermediate(x, mask)

#        nnz = output_mask.sum(0)
#        sparsity_multiplier = T.shape_padright(T.shape_padleft((B+1e-6)/(nnz+1e-6)))

#        wPhi = T.maximum(Phi + T.log(Alpha), constantX(-100.0)) #BxDxC
#        lp_current = log_sum_exp(wPhi, axis = 2) * output_mask #BxD
#        lp_current_sum = (lp_current.sum(1) * D / (D-d)).sum() #1

        loglikelihood = logdensity.mean(dtype=floatX)
        loss = -loglikelihood

        dp_dz_alpha = T.grad(loss, z_alpha)  # BxDxC
        gb_alpha = dp_dz_alpha.sum(0)  # DxC
        gV_alpha = T.tensordot(h.T, dp_dz_alpha, [[1], [0]]).dimshuffle((1, 0, 2))  # DxHxC

        dp_dz_mu = T.grad(loss, z_mu)  # BxDxC
        dp_dz_mu = dp_dz_mu * Sigma  # Heuristic
        gb_mu = dp_dz_mu.sum(0)  # DxC
        gV_mu = T.tensordot(h.T, dp_dz_mu, [[1], [0]]).dimshuffle((1, 0, 2))  # DxHxC

        dp_dz_sigma = T.grad(loss, z_sigma)  # BxDxC
        gb_sigma = dp_dz_sigma.sum(0)  # DxC
        gV_sigma = T.tensordot(h.T, dp_dz_sigma, [[1], [0]]).dimshuffle((1, 0, 2))  # DxHxC

        if self.n_layers > 1:
            gWs, gbs, gW1, gWflags, gb1 = T.grad(loss, [self.Ws, self.bs, self.W1, self.Wflags, self.b1])
            gradients = {"V_alpha":gV_alpha, "b_alpha":gb_alpha, "V_mu":gV_mu, "b_mu":gb_mu, "V_sigma":gV_sigma, "b_sigma":gb_sigma, "Ws":gWs, "bs":gbs, "W1":gW1, "b1":gb1, "Wflags":gWflags}
        else:
            gW1, gWflags, gb1 = T.grad(loss, [self.W1, self.Wflags, self.b1])
            gradients = {"V_alpha":gV_alpha, "b_alpha":gb_alpha, "V_mu":gV_mu, "b_mu":gb_mu, "V_sigma":gV_sigma, "b_sigma":gb_sigma, "W1":gW1, "b1":gb1, "Wflags":gWflags}
        # Gradients
        return (loss, gradients)

    def sample(self, n=1):
        W1 = self.W1.get_value()
        b1 = self.b1.get_value()
        Wflags = self.Wflags.get_value()
        if self.n_layers > 1:
            Ws = self.Ws.get_value()
            bs = self.bs.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()
        nl = self.parameters["nonlinearity"].get_numpy_f()
        samples = np.zeros((self.n_visible, n))
        for s in xrange(n):
            # Sample an ordering
            ordering = self.orderings[np.random.randint(len(self.orderings))]
            a = np.zeros((self.n_hidden,)) + b1  # H
            for j in xrange(self.n_visible):
                i = ordering[j]
                h = nl(a)
                for l in xrange(self.n_layers - 1):
                    h = nl(np.dot(h, Ws[l]) + bs[l])
                Alpha = Utils.nnet.softmax(np.dot(h, V_alpha[i]) + b_alpha[i])  # C
                Mu = np.dot(h, V_mu[i]) + b_mu[i]  # C
                Sigma = np.minimum(np.exp(np.dot(h, V_sigma[i]) + b_sigma[i]), 1)
                comp = Utils.nnet.random_component(Alpha)
                samples[i, s] = np.random.normal(Mu[comp], Sigma[comp])
                if j < self.n_visible - 1:
                    a += np.dot(samples[i, s][np.newaxis, np.newaxis], W1[i][np.newaxis, :]).flatten() + Wflags[i]
        return samples

    def conditional_logdensity(self, x):
        """ n is the number of samples """
        W1 = self.W1.get_value()
        b1 = self.b1.get_value()
        Wflags = self.Wflags.get_value()
        if self.n_layers > 1:
            Ws = self.Ws.get_value()
            bs = self.bs.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()
        nl = self.parameters["nonlinearity"].get_numpy_f()
        # Sample an order of the dimensions (from the list of orders that conform this MoNADE, which can be just one BTW)
        conditionals = np.zeros((self.n_orderings, self.n_visible))
        for o_index, o in enumerate(self.orderings):
            a = np.zeros((self.n_hidden,)) + b1
            for j in xrange(self.n_visible):
                i = o[j]
                h = nl(a)
                for l in xrange(self.n_layers - 1):
                    h = nl(np.dot(h, Ws[l]) + bs[l])
                z_alpha = np.dot(h, V_alpha[i]) + b_alpha[i]
                z_mu = np.dot(h, V_mu[i]) + b_mu[i]
                z_sigma = np.dot(h, V_sigma[i]) + b_sigma[i]
                Alpha = Utils.nnet.softmax(z_alpha)  # C
                Mu = z_mu  # C
                Sigma = np.exp(z_sigma)
                conditionals[o_index, i] += logsumexp(-0.5 * ((Mu - x[i]) / Sigma) ** 2 - np.log(Sigma) - 0.5 * np.log(2 * np.pi) + np.log(Alpha), axis=1)
                if j < self.n_visible - 1:
                    a += x[i] * W1[i] + Wflags[i]
        return logsumexp(conditionals + np.log(1 / self.n_orderings), axis=0)

    def conditional_logdensities(self, x, ys):
        """ n is the number of samples """
        W1 = self.W1.get_value()
        b1 = self.b1.get_value()
        Wflags = self.Wflags.get_value()
        if self.n_layers > 1:
            Ws = self.Ws.get_value()
            bs = self.bs.get_value()
        V_alpha = self.V_alpha.get_value()
        b_alpha = self.b_alpha.get_value()
        V_mu = self.V_mu.get_value()
        b_mu = self.b_mu.get_value()
        V_sigma = self.V_sigma.get_value()
        b_sigma = self.b_sigma.get_value()
        nl = self.parameters["nonlinearity"].get_numpy_f()
        # Sample an order of the dimensions (from the list of orders that conform this MoNADE, which can be just one BTW)
        conditionals = np.zeros((self.n_orderings, self.n_visible, len(ys)))
        for o_index, o in enumerate(self.orderings):
            a = np.zeros((self.n_hidden,)) + b1
            for j in xrange(self.n_visible):
                i = o[j]
                h = nl(a)
                for l in xrange(self.n_layers - 1):
                    h = nl(np.dot(h, Ws[l]) + bs[l])
                z_alpha = np.dot(h, V_alpha[i]) + b_alpha[i]
                z_mu = np.dot(h, V_mu[i]) + b_mu[i]
                z_sigma = np.dot(h, V_sigma[i]) + b_sigma[i]
                Alpha = Utils.nnet.softmax(z_alpha)  # C
                Mu = z_mu  # C
                Sigma = np.exp(z_sigma)
                conditionals[o_index, i, :] += logsumexp(-0.5 * ((Mu[np.newaxis, :] - ys[:, np.newaxis]) / Sigma) ** 2 - np.log(Sigma) - 0.5 * np.log(2 * np.pi) + np.log(Alpha), axis=1)
                a += x[i] * W1[i] + Wflags[i]
        return logsumexp(conditionals + np.log(1 / self.n_orderings), axis=0)

def test_orderless_gradient():
    # # Check gradients
    D = 3
    H = 2
    L = 1
    C = 2
    B = 3
    # Instantiate an object
    nade = OrderlessMoGNADE(D, H, L, C)
    # Initialize its parameters
    nade.initialize_parameters_cover_domain([(-1.0, 1.0)] * D)
    # Calculate gradient by finite differences for a few orderings
    orderings = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    X = np.asarray([[0.5, -0.1, 0.2], [-0.5, 0.1, 0.2], [-0.5, -0.1, 0.2]]).T
    M = np.asarray([[1, 1, 0], [1, 0, 0], [0, 1, 0]]).T
    # Calculate gradient using the corresponding method
    # Create a minibatch where the data has all possible masks applied
    auto_gradients = nade.compiled_gradient_mask_logdensity_auto(X, M)
    gradients = nade.compiled_gradient_mask_logdensity(X, M)
    print(auto_gradients)
    print(gradients)

if __name__ == "__main__":
    test_orderless_gradient()

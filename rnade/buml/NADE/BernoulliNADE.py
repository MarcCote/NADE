# A NADE that has for output distribution a Mixture of Gaussians
from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from ParameterInitialiser import Gaussian
from theano.tensor.shared_randomstreams import RandomStreams
from Model import *
from NADE import NADE
import Utils
import scipy
from Utils.theano_helpers import constantX, floatX


class BernoulliNADE(NADE):
    def __init__(self, n_visible, n_hidden, nonlinearity="RLU"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(TensorParameter("W", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("V", (n_visible, n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("b", (n_visible), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("activation_rescaling", (n_visible), theano=True), optimise=False, regularise=False)
        self.recompile()

    def initialize_parameters(self, marginal, W_initialiser=Gaussian(std=0.01)):
        for p in [self.W, self.V]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        self.b.set_value((-np.log((1 - marginal) / marginal)).astype(floatX))
        self.activation_rescaling.set_value(np.ones(self.activation_rescaling.get_value().shape).astype(floatX))

    def initialize_parameters_from_dataset(self, dataset, W_initialiser=Gaussian(std=0.01), sample_size=1000):
        for p in [self.W, self.V]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        data_sample = dataset.sample_data(sample_size)[0].astype(floatX)
        marginal = data_sample.mean(axis=0)
        self.b.set_value((-np.log((1 - marginal) / marginal)).astype(floatX))
        self.activation_rescaling.set_value(np.ones(self.activation_rescaling.get_value().shape).astype(floatX))

    def sym_logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, v, b, activations_factor, p_prev, a_prev, x_prev):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            h = self.nonlinearity(a * activations_factor)  # BxH
            t = T.dot(h, v) + b
            p_xi_is_one = T.nnet.sigmoid(t) * constantX(0.9999) + constantX(0.0001 * 0.5)  # Make logistic regression more robust by having the sigmoid saturate at 0.00005 and 0.99995
            p = p_prev + x * T.log(p_xi_is_one) + (1 - x) * T.log(1 - p_xi_is_one)
            return (p, a, x)
        # First element is different (it is predicted from the bias only)
        a0 = T.zeros_like(T.dot(x.T, self.W))  # BxH
        p0 = T.zeros_like(x[0])
        x0 = T.ones_like(x[0])
        ([ps, _, _], updates) = theano.scan(density_given_previous_a_and_x,
                                            sequences=[x, self.W, self.V, self.b, self.activation_rescaling],
                                            outputs_info=[p0, a0, x0])
        return (ps[-1], updates)


class BernoulliNADE_FixedBackground(NADE):
    # This version calculates the bias on the fly, by interpreting the logistic units as linear energy model Bayes' classifiers, fixing one of the models to assign p=0.5 to each input
    def __init__(self, n_visible, n_hidden, nonlinearity="sigmoid"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(TensorParameter("W", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("b", (n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V", (n_visible, n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("c", (n_visible), theano=True), optimise=True, regularise=False)
        self.recompile()

    def initialize_parameters(self, marginal, W_initialiser=Gaussian(std=0.01)):
        for p in [self.W, self.V]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        self.c.set_value((-np.log((1 - marginal) / marginal)).astype(floatX))

    def initialize_parameters_from_dataset(self, dataset, W_initialiser=Gaussian(std=0.01), sample_size=1000):
        for p in [self.W, self.V]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        data_sample = dataset.sample_data(sample_size)[0].astype(floatX)
        marginal = data_sample.mean(axis=0)
        self.b.set_value((-np.log((1 - marginal) / marginal)).astype(floatX))
        self.activation_rescaling.set_value(np.ones(self.activation_rescaling.get_value().shape).astype(floatX))

    def sym_logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, v, c, p_prev, a_prev, x_prev, bias_prev):
            a = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
            bias = bias_prev + constantX(np.log(2)) - T.log(1 + T.exp(w))
            h = self.nonlinearity(a + bias + self.b)  # BxH
            t = T.dot(h, v) + c
            p_xi_is_one = T.nnet.sigmoid(t) * constantX(0.9999) + constantX(0.0001 * 0.5)  # Make logistic regression more robust by having the sigmoid saturate at 0.00005 and 0.99995
            p = p_prev + x * T.log(p_xi_is_one) + (1 - x) * T.log(1 - p_xi_is_one)
            return (p, a, x, bias)

        a0 = T.zeros_like(T.dot(x.T, self.W))  # BxH
        p0 = T.zeros_like(x[0])
        x0 = T.zeros_like(x[0])
        bias0 = T.zeros_like(self.b)
        ([ps, _, _, _], updates) = theano.scan(density_given_previous_a_and_x,
                                            sequences=[x, self.W, self.V, self.c],
                                            outputs_info=[p0, a0, x0, bias0])
        return (ps[-1], updates)

#     def sym_neg_loglikelihood_gradient(self, x):
#         loglikelihood, updates = self.sym_logdensity(x)
#         mean_loglikelihood = -loglikelihood.mean()
#         # Gradients
#         gradients = {}
#         for param in self.parameters_to_optimise:
#             gradients[param] = T.grad(mean_loglikelihood, self.__getattribute__(param))
#         return (mean_loglikelihood, gradients, updates)


class BernoulliNADE_FittedBackground(NADE):
    # This version calculates the bias on the fly, by interpreting the logistic units as linear energy model Bayes' classifiers, the background model is also trained
    def __init__(self, n_visible, n_hidden, nonlinearity="sigmoid"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(TensorParameter("W", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("WB", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("b", (n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V", (n_visible, n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("c", (n_visible), theano=True), optimise=True, regularise=False)
        self.recompile()

    def initialize_parameters(self, marginal, W_initialiser=Gaussian(std=0.01)):
        for p in [self.W, self.WB, self.V, self.b]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        self.c.set_value((-np.log((1 - marginal) / marginal)).astype(floatX))

    def sym_logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, wb, v, c, p_prev, a_prev, bias_prev):
            h = self.nonlinearity(a_prev + bias_prev)  # BxH
            t = T.dot(h, v) + c
            p_xi_is_one = T.nnet.sigmoid(t) * constantX(0.9999) + constantX(0.0001 * 0.5)  # Make logistic regression more robust by having the sigmoid saturate at 0.00005 and 0.99995
            p = p_prev + x * T.log(p_xi_is_one) + (1 - x) * T.log(1 - p_xi_is_one)
            a = a_prev + T.dot(T.shape_padright(x, 1), T.shape_padleft(w - wb, 1))
            bias = bias_prev + T.log(1 + T.exp(wb)) - T.log(1 + T.exp(w))
            return (p, a, bias)
        a0 = T.zeros_like(T.dot(x.T, self.W))  # BxH
        p0 = T.zeros_like(x[0])
        bias0 = self.b
        ([ps, _, _], updates) = theano.scan(density_given_previous_a_and_x,
                                            sequences=[x, self.W, self.WB, self.V, self.c],
                                            outputs_info=[p0, a0, bias0])
        return (ps[-1], updates)

#     def sym_neg_loglikelihood_gradient(self, x):
#         loglikelihood, updates = self.sym_logdensity(x)
#         mean_loglikelihood = -loglikelihood.mean()
#         # Gradients
#         gradients = {}
#         for param in self.parameters_to_optimise:
#             gradients[param] = T.grad(mean_loglikelihood, self.__getattribute__(param))
#         return (mean_loglikelihood, gradients, updates)


class BernoulliNADE_FittedBackgroundSparse(NADE):
    # This version calculates the bias on the fly, by interpreting the logistic units as linear energy model Bayes' classifiers, the background model is also trained
    def __init__(self, n_visible, n_hidden, nonlinearity="sigmoid"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(TensorParameter("W", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("WB", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("b", (n_hidden), theano=True), optimise=False, regularise=False)
        self.add_parameter(TensorParameter("V", (n_visible, n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("c", (n_visible), theano=True), optimise=True, regularise=False)
        self.recompile()

    def initialize_parameters(self, marginal, W_initialiser=Gaussian(std=0.01)):
        for p in [self.W, self.WB, self.V]:
            p.set_value(W_initialiser.get_tensor(p.get_value().shape).astype(floatX))
        self.c.set_value((-np.log((1 - marginal) / marginal)).astype(floatX))
        self.b.set_value((np.ones(self.b.get_value().shape) * -2.2).astype(floatX))

    def sym_logdensity(self, x):
        """ x is a matrix of column datapoints (VxB) V = n_visible, B = batch size """
        def density_given_previous_a_and_x(x, w, wb, v, c, p_prev, a_prev, bias_prev):
            h = self.nonlinearity(a_prev + bias_prev)  # BxH
            t = T.dot(h, v) + c
            p_xi_is_one = T.nnet.sigmoid(t) * constantX(0.9999) + constantX(0.0001 * 0.5)  # Make logistic regression more robust by having the sigmoid saturate at 0.00005 and 0.99995
            p = p_prev + x * T.log(p_xi_is_one) + (1 - x) * T.log(1 - p_xi_is_one)
            a = a_prev + T.dot(T.shape_padright(x, 1), T.shape_padleft(w - wb, 1))
            bias = bias_prev + T.log(1 + T.exp(wb)) - T.log(1 + T.exp(w))
            return (p, a, bias)
        a0 = T.zeros_like(T.dot(x.T, self.W))  # BxH
        p0 = T.zeros_like(x[0])
        bias0 = self.b
        ([ps, _, _], updates) = theano.scan(density_given_previous_a_and_x,
                                            sequences=[x, self.W, self.WB, self.V, self.c],
                                            outputs_info=[p0, a0, bias0])
        return (ps[-1], updates)

#     def sym_neg_loglikelihood_gradient(self, x):
#         loglikelihood, updates = self.sym_logdensity(x)
#         mean_loglikelihood = -loglikelihood.mean()
#         # Gradients
#         gradients = {}
#         for param in self.parameters_to_optimise:
#             gradients[param] = T.grad(mean_loglikelihood, self.__getattribute__(param))
#         return (mean_loglikelihood, gradients, updates)

# A NADE that has Bernoullis for output distribution
from __future__ import division
from Model.Model import SizeParameter, TensorParameter
from NADE import NADE
from ParameterInitialiser import Gaussian
from Utils.Estimation import Estimation
from Utils.nnet import sigmoid, logsumexp
from Utils.theano_helpers import constantX, floatX
from itertools import izip
import numpy as np
import theano
import theano.tensor as T


class OrderlessBernoulliNADE(NADE):
    def __init__(self, n_visible, n_hidden, n_layers, nonlinearity="RLU"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(SizeParameter("n_layers"))
        self.n_layers = n_layers
        self.add_parameter(TensorParameter("Wflags", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("W1", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("b1", (n_hidden), theano=True), optimise=True, regularise=False)
        if self.n_layers > 1:
            self.add_parameter(TensorParameter("Ws", (n_layers, n_hidden, n_hidden), theano=True), optimise=True, regularise=True)
            self.add_parameter(TensorParameter("bs", (n_layers, n_hidden), theano=True), optimise=True, regularise=False)
        self.add_parameter(TensorParameter("V", (n_visible, n_hidden), theano=True), optimise=True, regularise=True)
        self.add_parameter(TensorParameter("c", (n_visible), theano=True), optimise=True, regularise=False)
        self.setup_n_orderings(1)
        self.recompile()

    @classmethod
    def create_from_params(cls, params):
        n_visible, n_hidden, n_layers = (params["n_visible"], params["n_hidden"], params["n_layers"])
        model = cls(n_visible, n_hidden, n_layers, params["nonlinearity"])
        model.set_parameters(params)
        return model

    @classmethod
    def create_from_smaller_NADE(cls, small_NADE, add_n_hiddens=1, W_initialiser=Gaussian(std=0.01), marginal=None):
        n_visible, n_hidden, n_layers, nonlinearity = (small_NADE.n_visible, small_NADE.n_hidden, small_NADE.n_layers, small_NADE.parameters["nonlinearity"].get_name())
        model = cls(n_visible, n_hidden, n_layers + add_n_hiddens, nonlinearity)
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
        model.V.set_value(W_initialiser.get_tensor(model.V.get_value().shape))
        if marginal is None:
            model.c.set_value(small_NADE.c.get_value())
        else:
            model.c.set_value(-np.log((1 - marginal) / marginal).astype(floatX))
        return model

    def recompile(self):
        x = T.matrix('x', dtype=floatX)
        m = T.matrix('m', dtype=floatX)
        logdensity = self.sym_mask_logdensity_estimator(x, m)
        self.compiled_mask_logdensity_estimator = theano.function([x, m], logdensity, allow_input_downcast=True)

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

    def initialize_parameters(self, marginal, W_initialiser=Gaussian(std=0.01)):
        self.Wflags.set_value(W_initialiser.get_tensor(self.Wflags.get_value().shape))
        self.W1.set_value(W_initialiser.get_tensor(self.W1.get_value().shape))
        self.b1.set_value(W_initialiser.get_tensor(self.b1.get_value().shape))
        if self.n_layers > 1:
            self.Ws.set_value(W_initialiser.get_tensor(self.Ws.get_value().shape))
            self.bs.set_value(W_initialiser.get_tensor(self.bs.get_value().shape))
        self.V.set_value(W_initialiser.get_tensor(self.V.get_value().shape))
        self.c.set_value(-np.log((1 - marginal) / marginal).astype(floatX))

    def initialize_parameters_from_dataset(self, dataset, W_initialiser=Gaussian(std=0.01), sample_size=1000):
        self.Wflags.set_value(W_initialiser.get_tensor(self.Wflags.get_value().shape))
        self.W1.set_value(W_initialiser.get_tensor(self.W1.get_value().shape))
        self.b1.set_value(W_initialiser.get_tensor(self.b1.get_value().shape))
        if self.n_layers > 1:
            self.Ws.set_value(W_initialiser.get_tensor(self.Ws.get_value().shape))
            self.bs.set_value(W_initialiser.get_tensor(self.bs.get_value().shape))
        self.V.set_value(W_initialiser.get_tensor(self.V.get_value().shape))
        data_sample = dataset.sample_data(sample_size)[0].astype(floatX)
        marginal = data_sample.mean(axis=0)
        self.c.set_value(-np.log((1 - marginal) / marginal).astype(floatX))

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
        V = self.V.get_value()
        c = self.c.get_value()

        for o_index, o in enumerate(self.orderings):
            a = np.zeros((B, self.n_hidden))
            input_mask_contribution = np.zeros((B, self.n_hidden))
            for j in xrange(self.n_visible):
                i = o[j]
                x_i = x[i]
                h = nl(input_mask_contribution + a + b1)
                for l in xrange(self.n_layers - 1):
                    h = nl(np.dot(h, Ws[l]) + bs[l])
                t = np.dot(h, V[i]) + c[i]
                p_xi_is_one = sigmoid(t) * 0.9999 + 0.0001 * 0.5
                lp[:, o_index] += x_i * np.log(p_xi_is_one) + (1 - x_i) * np.log(1 - p_xi_is_one)
                a += np.dot(x[i][:, np.newaxis], W1[i][np.newaxis, :])
                input_mask_contribution += Wflags[i]
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

    def sym_mask_logdensity_estimator(self, x, mask):
        """ x is a matrix of column datapoints (DxB) D = n_visible, B = batch size """
        # non_linearity_name = self.parameters["nonlinearity"].get_name()
        # assert(non_linearity_name == "sigmoid" or non_linearity_name=="RLU")
        x = x.T  # BxD
        mask = mask.T  # BxD
        output_mask = constantX(1) - mask  # BxD
        D = constantX(self.n_visible)
        d = mask.sum(1)  # d is the 1-based index of the dimension whose value to infer (not the size of the context)
        masked_input = x * mask  # BxD
        h = self.nonlinearity(T.dot(masked_input, self.W1) + T.dot(mask, self.Wflags) + self.b1)  # BxH
        for l in xrange(self.n_layers - 1):
            h = self.nonlinearity(T.dot(h, self.Ws[l]) + self.bs[l])  # BxH
        t = T.dot(h, self.V.T) + self.c  # BxD
        p_x_is_one = T.nnet.sigmoid(t) * constantX(0.9999) + constantX(0.0001 * 0.5)  # BxD
        lp = ((x * T.log(p_x_is_one) + (constantX(1) - x) * T.log(constantX(1) - p_x_is_one)) * output_mask).sum(1) * D / (D - d)  # B
        return lp

    def sym_masked_neg_loglikelihood_gradient(self, x, mask):
        loglikelihood = self.sym_mask_logdensity_estimator(x, mask)
        mean_loglikelihood = -loglikelihood.mean()
        # Gradients
        gradients = {}
        for param in self.parameters_to_optimise:
            gradients[param] = T.grad(mean_loglikelihood, self.__getattribute__(param))
        return (mean_loglikelihood, gradients)

    def sample(self, n=1):
        W1 = self.W1.get_value()
        b1 = self.b1.get_value()
        Wflags = self.Wflags.get_value()
        if self.n_layers > 1:
            Ws = self.Ws.get_value()
            bs = self.bs.get_value()
        V = self.V.get_value()
        c = self.c.get_value()
        nl = self.parameters["nonlinearity"].get_numpy_f()
        samples = np.zeros((self.n_visible, n))
        for s in xrange(n):
            # Sample an ordering
            ordering = self.orderings[np.random.randint(len(self.orderings))]
            a = np.zeros((self.n_hidden,))  # H
            input_mask_contribution = np.zeros((self.n_hidden))
            for j in xrange(self.n_visible):
                i = ordering[j]
                h = nl(input_mask_contribution + a + b1)
                for l in xrange(self.n_layers - 1):
                    h = nl(np.dot(h, Ws[l]) + bs[l])
                t = np.dot(h, V[i]) + c[i]
                p_xi_is_one = sigmoid(t) * 0.9999 + 0.0001 * 0.5  # B
                input_mask_contribution += Wflags[i]
                a += np.dot(samples[i, s][np.newaxis, np.newaxis], W1[i][np.newaxis, :])
                samples[i, s] = np.random.random() < p_xi_is_one
        return samples

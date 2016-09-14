from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from Model import *
from Utils.Estimation import Estimation
import Utils


class NADE(Model):
    """A NADE abstract class"""
    def __init__(self, n_visible, n_hidden, nonlinearity="RLU"):
        self.theano_rng = RandomStreams(np.random.randint(2 ** 30))
        self.add_parameter(SizeParameter("n_visible"))
        self.add_parameter(SizeParameter("n_hidden"))
        self.add_parameter(NonLinearityParameter("nonlinearity"))
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.parameters["nonlinearity"].set_value(nonlinearity)

    def logdensity(self, x):
        return self.compiled_logdensity(x)

    def logdensity_new(self, x):
        return self.compiled_logdensity_new(x)

    def gradients(self, x):
        return self.compiled_gradients(x)

    def gradients_new(self, x):
        return self.compiled_gradients_new(x)

    def sample(self):
        return self.compiled_sample()

    def estimate_loglikelihood_for_dataset(self, x_dataset, minibatch_size=1000):
        loglikelihood = 0.0
        loglikelihood_sq = 0.0
        n = 0
        iterator = x_dataset.iterator(batch_size=minibatch_size, get_smaller_final_batch=True)
        for x in iterator:
            x = x.T  # VxB
            n += x.shape[1]
            ld = self.logdensity(x)
            loglikelihood += ld.sum()
            loglikelihood_sq += (ld ** 2).sum()
        return Estimation.sample_mean_from_sum_and_sum_sq(loglikelihood, loglikelihood_sq, n)

    def recompile(self):
        x = T.matrix('x', dtype=theano.config.floatX)
        logdensity, updates = self.sym_logdensity(x)
        # self.compiled_logdensity = theano.function([x], logdensity, allow_input_downcast = True, updates = updates, mode=theano.compile.MonitorMode(post_func=Utils.theano_helpers.detect_nan))
        self.compiled_logdensity = theano.function([x], logdensity, allow_input_downcast=True, updates=updates)
#         gradients, updates = self.sym_gradients(x)
#         self.compiled_gradients = theano.function([x], gradients, allow_input_downcast=True, updates=updates)

    def sym_logdensity(self, X):
        pass

    def sym_neg_loglikelihood_gradient(self, X):
        ret = self.sym_logdensity(X)
        if isinstance(ret, tuple):
            assert(len(ret) == 2)
            loglikelihood, updates = ret
        else:
            loglikelihood = ret
            updates = dict()
        loss = -loglikelihood.mean()
        # Gradients
        gradients = dict([(param, T.grad(loss, self.get_parameter(param))) for param in self.get_parameters_to_optimise()])
        return (loss, gradients, updates)

    @classmethod
    def create_from_params(cls, params):
        model = cls(params["n_visible"], params["n_hidden"], params["nonlinearity"])
        model.set_parameters(params)
        return model


class MixtureNADE(NADE):
    """ An abstract NADE model, that outputs a mixture model for each element """
    def __init__(self, n_visible, n_hidden, n_components, nonlinearity="RLU"):
        NADE.__init__(self, n_visible, n_hidden, nonlinearity)
        self.add_parameter(SizeParameter("n_components"))
        self.n_components = n_components

    @classmethod
    def create_from_params(cls, params):
        model = cls(params["n_visible"], params["n_hidden"], params["n_components"], params["nonlinearity"])
        model.set_parameters(params)
        return model

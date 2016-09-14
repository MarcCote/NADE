import numpy as np
from Optimization.Optimizer import Optimizer, has_parameter
from Epochable import Epochable
from itertools import izip
import theano
import Data


@has_parameter("minibatch_size", 100)
@has_parameter("learning_rate", 0.1, theano_param=True)
class SGD(Epochable, Optimizer):
    def __init__(self, model, loss):
        Epochable.__init__(self)
        Optimizer.__init__(self, model, loss)

    def before_training(self):
        super(SGD, self).before_training()

    def before_training_iteration(self):
        self.n_minibatches = 0
        self.training_loss.set_value(0)
        super(SGD, self).before_training_iteration()

    def optimization_step(self):
        self.n_minibatches += 1
        self.compiled_optimization_step()

    def train(self):
        its = self.get_data_iterators()
        mb = self.get_minibatch_tuple(its)
        self.compile_optimization_step(*mb)
        self.before_training()
        while not self.is_training_finished():
            self.before_training_iteration()
            for _ in izip(*its):
                self.optimization_step()
            self.after_training_iteration()
        self.after_training()

    def get_training_loss(self):
        return self.training_loss.get_value() / self.n_minibatches

    def get_data_iterators(self):
        return [Data.TheanoDatasetIteratorAdapter(ds.iterator(batch_size=self.minibatch_size, shuffle=True), n_batches=self.updates_per_epoch) for ds in self.datasets]

    def get_minibatch_tuple(self, its):
        elements = list()
        for it in its:
            for mb in it.minibatch:
                if self.datapoints_as_columns:
                    elements.append(mb.T)
                else:
                    elements.append(mb)
        return elements

    def ret_to_loss_gradient_updates(self, ret):
        if len(ret) == 3:
            step_loss, gradient, updates = ret
        else:
            updates = theano.compat.python2x.OrderedDict()
            step_loss, gradient = ret
        return step_loss, gradient, updates

    def compile_optimization_step(self, *params):
        # LL accumulator for each epoch
        self.training_loss = theano.shared(np.array(0.0, dtype=theano.config.floatX), "training_loss")  # @UndefinedVariable
        # Calculate loss for minibatch and gradient, optionally there's a 3rd returned variable which is used to initialize the theano function updates (needed if there is a scan in the symbolic loss function)
        ret = self.loss(*params)
        step_loss, gradient, updates = self.ret_to_loss_gradient_updates(ret)
        updates[self.training_loss] = self.training_loss + step_loss
        # Update parameters
        for param, g in gradient.iteritems():
            param_value = self.model.get_parameter(param)
            updates[param_value] = param_value - self.learning_rate * g
        # Compile function that does all that
        self.compiled_optimization_step = theano.function([], [], updates=updates)

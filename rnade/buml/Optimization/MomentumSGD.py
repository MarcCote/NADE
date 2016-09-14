from Optimization.Optimizer import has_parameter
import numpy as np
import theano
from SGD import SGD


@has_parameter("momentum", 0.0, theano_param=True)
class MomentumSGD(SGD):
    def __init__(self, model, loss):
        SGD.__init__(self, model, loss)

    def compile_optimization_step(self, *params):
        # Initialise velocity variables for each parameter of the model
        self.velocities = self.model.k_like_parameters_to_optimise(0.0, "velocity_")
        # LL accumulator for each epoch
        self.training_loss = theano.shared(np.array(0.0, dtype=theano.config.floatX), "training_loss")  # @UndefinedVariable
        # Calculate loss for minibatch and gradient, optionally there's a 3rd returned variable which is used to initialize the theano function updates (needed if there is a scan in the symbolic loss function)
        ret = self.loss(*params)
        step_loss, gradient, updates = self.ret_to_loss_gradient_updates(ret)
        updates[self.training_loss] = self.training_loss + step_loss
        # Update parameters
        for param, g in gradient.iteritems():
            param_value = self.model.get_parameter(param)
            param_velocity = self.velocities[param]
            new_velocity = self.momentum * param_velocity - self.learning_rate * g
            updates[param_value] = param_value + new_velocity
            updates[param_velocity] = new_velocity

        # Compile function that does all that
        self.compiled_optimization_step = theano.function([], [], updates=updates)

import copy
from TrainingController import TrainingController
import numpy as np


class AdaptiveLearningRate(TrainingController):
    def __init__(self, initial_rate, min_rate, epochs=200, decrease="linearly", min_improvement=-np.inf, scaling_factor=0.5):
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.min_improvement = min_improvement
        self.scaling_factor = scaling_factor
        self.last_training_loss = None
        self.decrease = decrease
        if decrease == "linearly":
            if min_rate == 0:
                self.epoch_constant = (initial_rate - min_rate) / (epochs + 1)
            else:
                self.epoch_constant = (initial_rate - min_rate) / epochs
        else:  # geometrically
            self.epoch_factor = np.exp((np.log(min_rate) - np.log(initial_rate)) / epochs)

    def before_training(self, trainable):
        trainable.set_learning_rate(self.initial_rate)

    def after_training_iteration(self, trainable):
        '''If training loss improves by less than min_improvement, then the model parameters are restored to their value at the beginning of the last iteration'''
        training_loss = trainable.get_training_loss()
        if self.last_training_loss is not None and self.last_training_loss - training_loss < self.min_improvement:
            trainable.set_learning_rate(trainable.get_learning_rate() * self.scaling_factor)
            trainable.model.set_parameters(self.last_params)
        else:
            self.last_training_loss = training_loss
            if self.decrease == "linearly":
                trainable.set_learning_rate(trainable.get_learning_rate() - self.epoch_constant)
            else:
                trainable.set_learning_rate(trainable.get_learning_rate() * self.epoch_factor)
            self.last_params = copy.deepcopy(trainable.model.get_parameters())
        return trainable.get_learning_rate() < self.min_rate

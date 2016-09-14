import copy
from TrainingController import TrainingController
import numpy as np

class MontrealLearningRate(TrainingController):
    def __init__(self, initial_rate, decrease_constant):
        self.initial_rate = initial_rate
        self.decrease_constant = decrease_constant
        self.counter = 0        

    def before_training(self, trainable):
        trainable.set_learning_rate(self.initial_rate)
        self.counter = 0

    def after_training_iteration(self, trainable):
        '''If training error improves by less than min_improvement, then the model parameters are restored to their value at the beginning of the last iteration'''
        trainable.set_learning_rate(self.initial_rate/(1+self.decrease_constant*self.counter))
        self.counter += 1
        return False

import numpy as np
from TrainingController import TrainingController


class EarlyStopping(TrainingController):
    def __init__(self, measurement, n_iterations, maximization=False):
        self.measurement = measurement
        self.n_iterations = n_iterations
        self.maximization = maximization
        if self.maximization:
            self.best_so_far = -np.inf
        else:
            self.best_so_far = np.inf
        self.counter = 0

    def after_training_iteration(self, trainer):
        '''If training error improves by less than min_improvement, then stop'''
        value = self.measurement.take_measurement(trainer)
        if (self.maximization and value > self.best_so_far) or (not self.maximization and value < self.best_so_far):
            self.best_so_far = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.n_iterations

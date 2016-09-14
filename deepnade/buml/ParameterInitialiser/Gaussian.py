from ParameterInitialiser import ParameterInitialiser
import numpy as np

class Gaussian(ParameterInitialiser):
    def __init__(self, std = 0.05 , mean = 0.0):
        self.std = std
        self.mean = mean

    def get_value(self):
        return np.random.normal(0.0, 1.0) * self.std + self.mean

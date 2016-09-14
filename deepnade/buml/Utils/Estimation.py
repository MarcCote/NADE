import numpy as np


class Estimation(object):
    def __init__(self, estimation, se):
        self.estimation = estimation
        self.se = se

    def __str__(self):
        return "%f (se:%f)" % (self.estimation, self.se)

    def __float__(self):
        return self.estimation

    def __lt__(self, other):
        # NOTE: Comparions two estimations is a tricky business, I will do it very naively:
        return self.estimation < float(other)

    def __gt__(self, other):
        # NOTE: Comparions two estimations is a tricky business, I will do it very naively:
        return self.estimation > float(other)

    def describe(self):
        return {"estimation": self.estimation, "se": self.se}

    def __neg__(self):
        return Estimation(-self.estimation, self.se)

    @classmethod
    def sample_mean_from_sum_and_sum_sq(cls, sum_, sum_sq, n):
        """Returns an estimation of the population mean from the sum, sum of squares and numberf of samples"""
        expected_value = sum_ / n
        expected_sq_value = sum_sq / n
        return Estimation(expected_value, np.sqrt((expected_sq_value - expected_value ** 2) / (n - 1)))

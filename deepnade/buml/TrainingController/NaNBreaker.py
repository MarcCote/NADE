import numpy as np
from TrainingController import TrainingController


class NaNBreaker(TrainingController):
    def __init__(self):
        pass

    def after_training_iteration(self, trainer):
        '''If training error improves by less than min_improvement, then stop'''
        if np.isnan(trainer.get_training_loss()):
            trainer.success = False
            return True
        else:
            return False

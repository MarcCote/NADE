from TrainingController import TrainingController


class MaxIterations(TrainingController):
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    def after_training_iteration(self, trainable):
        '''Stop training if epochs is greater than max_iterations'''
        return trainable.epoch >= self.max_iterations

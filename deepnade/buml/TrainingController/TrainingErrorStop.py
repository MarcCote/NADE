from TrainingController import TrainingController


class TrainingErrorStop(TrainingController):
    """
    Stops the training when the training error is lower than 'stop value'
    """
    def __init__(self, stop_value):
        self.stop_value = stop_value

    def after_training_iteration(self, trainable):
        return trainable.get_training_error() < self.stop_value

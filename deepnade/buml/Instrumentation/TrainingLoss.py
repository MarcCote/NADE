from Measurement import Measurement


class TrainingLoss(Measurement):
    """The instrumentable object must implement the get_error method"""
    def __init__(self, *args, **kwargs):
        self.attribute = "training_error"

    def take_measurement(self, instrumentable):
        return instrumentable.get_training_loss()

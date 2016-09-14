from Measurement import Measurement

class Parameters (Measurement):
    """The instrumentable object must implement the get_parameters method"""
    def __init__(self, *args, **kwargs):
        self.attribute = "parameters"

    def take_measurement(self, instrumentable):
        return instrumentable.model.get_parameters()

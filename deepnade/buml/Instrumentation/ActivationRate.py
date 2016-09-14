from Measurement import Measurement

class ActivationRate(Measurement):
    """The instrumentable object must implement the get_config method"""
    def __init__(self, *args, **kwargs):
        self.attribute = "activation_rate"
        
    def take_measurement(self, instrumentable):
        return instrumentable.get_activation_rate()

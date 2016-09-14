from Measurement import Measurement

class Configuration (Measurement):
    """The instrumentable object must implement the get_config method"""
    def __init__(self, *args, **kwargs):
        self.attribute = "configuration"
        
    def take_measurement(self, instrumentable):
        return instrumentable.get_parameters()

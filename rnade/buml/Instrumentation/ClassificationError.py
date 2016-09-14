from Measurement import Measurement

class ClassificationError (Measurement):
    """The instrumentable object must implement the get_error_for method"""
    def __init__(self, dataset,  name = "validation", *args, **kwargs):
        self.attribute = "%s_classification_error" % (name) 
        self.dataset = dataset
        
    def take_measurement(self, instrumentable):
        return instrumentable.model.get_classification_error(self.dataset)

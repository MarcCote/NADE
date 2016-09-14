from Measurement import Measurement

class Error (Measurement):
    """The instrumentable object must implement the get_error_for method"""
    def __init__(self, input_data, output_data = None,  name = "validation", cache = True, *args, **kwargs):        
        self.attribute = "%s_error" % (name) 
        self.input_data = input_data
        self.output_data = output_data
        self.cache = cache
        if self.cache:
            self.cache_storage = {}
        
    def take_measurement(self, instrumentable):
        if self.cache:
            entry = self.cache_storage.get(id(instrumentable), (None, None))
            if entry[0] == instrumentable.get_context():
                return entry[1] 
        if self.output_data is not None:
            error = instrumentable.model.get_error(self.input_data, self.output_data)
        else:
            error = instrumentable.model.get_error(self.input_data)
        if self.cache:
            self.cache_storage[id(instrumentable)] = (instrumentable.get_context(), error)
        return error
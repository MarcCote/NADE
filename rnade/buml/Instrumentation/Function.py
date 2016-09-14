from Measurement import Measurement

class Function (Measurement):
    def __init__(self, name, f, cache = True, *args, **kwargs):
        self.attribute = "%s" % (name)
        self.measurement_f = f
        self.cache = cache
        if self.cache:
            self.cache_storage = {} 
        
    def take_measurement(self, instrumentable):
        if self.cache:
            entry = self.cache_storage.get(id(instrumentable), (None, None))
            if entry[0] == instrumentable.get_context():
                return entry[1]
        value = self.measurement_f(instrumentable)
        if self.cache:
            self.cache_storage[id(instrumentable)] = (instrumentable.get_context(), value)
        return value
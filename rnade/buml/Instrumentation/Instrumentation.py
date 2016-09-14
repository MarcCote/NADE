import copy

class Instrumentation (object):
    def __init__(self, backends, measurement, at_lowest=[], at_highest=[], every=1, initial=True, final=True):
	self.backends = backends if isinstance(backends, list) else [backends]
        self.measurement = measurement
        self.lowest = at_lowest if isinstance(at_lowest, list) else [at_lowest]
        self.lowest_v = None
        self.lowest_measurements = None
        self.highest = at_highest if isinstance(at_highest, list) else [at_highest]
        self.highest_v = None
        self.highest_measurements = None
        self.every = every
        self.count = 0
        self.initial = initial
        self.final = final
        self.evaluated = False

    def initialize(self, instrumentable): pass

    def run(self, instrumentable, force = False):
        if not force:
            self.count += 1
            if self.count % self.every == 0:
                force = True
                self.count = 0
        if force or (self.initial and not self.evaluated):
            self.evaluated = True
            measurement = self.measurement.take_measurement(instrumentable)
            context = instrumentable.get_context()
            self.write(context, self.measurement.attribute, measurement)
            if len(self.lowest) > 0:
                if self.lowest_v is None or measurement < self.lowest_v:
                    self.lowest_v = measurement
                    self.lowest_measurements= map(lambda m: (m.attribute, copy.deepcopy(m.take_measurement(instrumentable))), self.lowest)
            if len(self.highest) > 0:
                if self.highest_v is None or measurement > self.highest_v:
                    self.highest_v = measurement
                    self.highest_measurements= map(lambda m: (m.attribute, copy.deepcopy(m.take_measurement(instrumentable))), self.highest)

    def end(self, instrumentable):
        if self.count != 0 and self.final:
            self.run(instrumentable, force = True)
        if len(self.lowest) > 0 and self.lowest_v is not None:
            context = ["lowest_%s" % (self.measurement.attribute)]
            for (k,v) in self.lowest_measurements:
                self.write(context, k,  v)
        if len(self.highest) > 0 and self.highest_v is not None:
            context = ["highest_%s" % (self.measurement.attribute)]
            for (k,v) in self.highest_measurements:
                self.write(context, k,  v)
 
    def take_measurement(self, instrumentable): pass
	
    def write(self, context, attribute, value):
        for b in self.backends:
            b.write(context, attribute, value)



	
	

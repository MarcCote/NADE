class Instrumentable (object):
    def __init__(self):
        self.instrumentation = []
        
    def add_instrumentation(self, ins):
        self.instrumentation.append(ins)

    def get_context(self):
        return []
        
    def initialize_instrumentation(self):
        for i in self.instrumentation: i.initialize(self)
        
    def run_instrumentation(self):
        for i in self.instrumentation: i.run(self)
        
    def end_instrumentation(self):
        for i in self.instrumentation: i.end(self)

from Measurement import Measurement
import time

class Timestamp(Measurement):
    def __init__(self, *args, **kwargs):
        self.attribute = "timestamp"
        self.time = time.time() 

    def take_measurement(self, instrumentable):
        previous_time = self.time
        self.time = time.time()
        return [self.time, self.time - previous_time]

import os
from Backend import Backend


class TextFile(Backend):
    def __init__(self, filename):
        self.filename = filename
        i = 1
        while os.path.exists(self.filename):
            i += 1
            self.filename = "%s_%d" % (filename, i)
        self.f = open(filename, 'w')
        self.last_route = ""

    def write(self, route, attribute, value):
        if route != self.last_route:
            self.f.write(str(route) + "\n")
            self.last_route = route
        try:
            self.f.write("\t%s : %s\n" % (str(attribute), str(value.__dict__)))
        except Exception:
            self.f.write("\t%s : %s\n" % (str(attribute), str(value)))

    def __del__(self):
        self.f.close()

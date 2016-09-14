from Backend import Backend


class Console(Backend):
    def __init__(self):
        self.last_route = ""

    def write(self, route, attribute, value):
        if route != self.last_route:
            print(route)
            self.last_route = route
        try:
            print("\t%s : %s" % (str(attribute), str(value.__dict__)))
        except Exception:
            print("\t%s : %s" % (str(attribute), str(value)))

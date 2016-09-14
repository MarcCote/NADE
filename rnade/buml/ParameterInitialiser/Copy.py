from ParameterInitialiser import ParameterInitialiser

class Copy(ParameterInitialiser):
    def __init__(self, values):
        self.values = values

    #TODO: Should accept a shape, instead of the value of the 2 dimensions
    def get_values(self, n_visible, n_hidden = 1):
        assert(self.values.shape == (n_visible, n_hidden))
        return self.values

from ParameterInitialiser import ParameterInitialiser

class Constant(ParameterInitialiser):
    def __init__(self, vals):
        self.vals = vals
        
    def get_tensor(self, shape):
        return self.vals.reshape(shape)

    def get_values(self, n_rows, n_cols=1):        
        return self.vals.reshape((n_rows, n_cols))

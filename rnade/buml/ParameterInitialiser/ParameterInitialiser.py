import numpy as np

class ParameterInitialiser(object):
    def __init__(self):
        pass

    def get_tensor(self, shape):
        w = np.zeros(shape, dtype = np.float32)
        if not (np.asarray(shape)==0).any():
            for x in np.nditer(w, op_flags=['readwrite']):        
                x[...] = self.get_value()
        return w

    #TODO: Should accept a shape, instead of the value of the 2 dimensions
    def get_values(self, n_visible, n_hidden = 1):
        w = np.zeros((n_visible, n_hidden), dtype = np.float32)
        for i in xrange(n_visible):
            for j in xrange(n_hidden):
                w[i,j] = self.get_value()
        return w

    #def get_value(self):
    #    raise Exception("Abstract method.")

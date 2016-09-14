from ParameterInitialiser import ParameterInitialiser
import math
import numpy as np

class Sparse(ParameterInitialiser):
    def __init__(self, sparse_connections = 15, std = 1.0):
        """
        """
        self.sparse_connections = sparse_connections
        self.std = std
        
    def get_tensor(self, shape):
        def sparse_vector(l,n):
            permutation = range(l)
            np.random.shuffle(permutation)
            #vec = np.zeros(l, dtype = np.float32)
            vec = np.random.normal(scale = 0.01, size = (l,)).astype(np.float32)
            vec[permutation[:n]] = np.random.normal(scale = self.std, size = (n,)).astype(np.float32)
            return vec
        if len(shape) == 1:
            return sparse_vector(shape[0], self.sparse_connections)
        else:
            return np.asarray([self.get_tensor(shape[1:]) for i in xrange(shape[0])])

    def get_values(self, n_visible, n_hidden=1):
        # Sparse can give the number of non-zero connections or the proportion of units from the visible layers to which each hidden unit connects
        if self.sparse_connections > 0.0 and self.sparse_connections < 1.0:
            sparse_connections = math.ceil(n_visible * self.sparse_connections)
        else:
            sparse_connections = self.sparse_connections        
        w = np.zeros((n_visible, n_hidden))
        for i in range(n_hidden):
            for j in np.random.random_integers(0, n_visible-1, sparse_connections):
                w[j,i] = np.random.normal(0.0, 1.0) * self.std
        return w

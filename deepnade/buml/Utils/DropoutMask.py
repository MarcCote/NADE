import Backends
import random
import numpy as np
from theano_helpers import floatX

def create_dropout_masks(route, fname, dimensionality, ks=1000):
    """
    route = path where to create a file
    fname = filename
    ks = thousand of masks to create (1e6 masks by default)
    """
    hdf5_backend = Backends.HDF5(route, fname)
    for i in xrange(ks):
        mask = random.random_binary_mask((dimensionality, 1000), np.random.randint(dimensionality, size=1000))
        mask = mask.astype(floatX)
        hdf5_backend.write([], "masks/%d/masks" % i, mask.T)
    del hdf5_backend
    
def test_dropout_mask_creation():
    create_dropout_masks("/tmp", "domask", 5, 2)
    
if __name__ == "__main__":
    test_dropout_mask_creation()
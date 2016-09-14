import numpy as np

def random_binary_mask(shape, ones_per_column):
    """
    Returns a random binary maks with ones_per_columns[i] ones on the i-th column
    
    Example: random_binary_maks((3,5),[1,2,3,1,2])
    Out: 
    array([[ 0.,  1.,  1.,  1.,  1.],
           [ 1.,  1.,  1.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  1.]]) 
    """
    # Assert that the number of columns in shape is equal to the length of the ones_per_column vector
    assert(shape[1] == len(ones_per_column))
    indexes = np.asarray(range(shape[0]))
    mask = np.zeros(shape, dtype="float32")
    for i,d in enumerate(ones_per_column):
        np.random.shuffle(indexes)
        mask[indexes[:d],i] = 1.0
    return mask

import random
import numpy as np
import numpy.random

def softmax(X):
    """Calculates softmax row-wise"""
    if X.ndim == 1:
        X = X - np.max(X)
        e = np.exp(X)
        return e/e.sum()
    else:
        X = X - np.max(X,1)[:,np.newaxis]
        e = np.exp(X)    
        return e/e.sum(1)[:,np.newaxis]
    
def inverse_softmax(X):
    """Calculates inverse of softmax row-wise"""
    if X.ndim == 1:
        t = np.log(X)
        return t - t.min()
    else:
        t = np.log(X)    
        return t - t.min(1)[:,np.newaxis]

def j_softmax(x):
    x = np.asarray(x)
    e = np.exp(x)
    s = np.sum(e)
    return np.diag(e)/s - np.outer(e,e)/s**2

def bp_softmax(do, x):
    return j_softmax(x).dot(do)

def bp_row_softmax(dW, W):
    o = np.zeros_like(W)
    for i in xrange(W.shape[0]):
        o[i,:] = bp_softmax(dW[i,:], W[i,:])
    return o

def normalize(x):        
    return x / x.sum()

def normalize_rows(X):
    return X / X.sum(1)[:,np.newaxis]

def j_normalize(x):
    D = len(x)
    s = np.sum(x)
    return np.eye(D)*1/s - np.ones((D,D))*x/s**2

def bp_normalize(do, x):
    return j_normalize(x).dot(do)

def logsumexp(X, axis = 1):
    """Calculates logsumexp row-wise"""
    if X.ndim == 1:
        m = np.max(X)
        return  m + np.log(np.sum(np.exp(X-m)))
    if X.ndim == 2:
        if axis == 0:
            m = np.max(X, 0)
            return  m + np.log(np.sum(np.exp(X-m[np.newaxis,:]), axis=0))
        elif axis == 1:
            m = np.max(X, 1)
            return  m + np.log(np.sum(np.exp(X-m[:,np.newaxis]), axis=1))
    if X.ndim == 3:
        if axis == 0:
            m = np.max(X, 0)
            return  m + np.log(np.sum(np.exp(X-m[np.newaxis,:,:]), axis=0))
        elif axis == 1:
            m = np.max(X, 1)
            return  m + np.log(np.sum(np.exp(X-m[:,np.newaxis,:]), axis=1))
        elif axis == 2:
            m = np.max(X, 2)
            return  m + np.log(np.sum(np.exp(X-m[:,:,np.newaxis]), axis=2))
    else:
        raise "Not implemented"

def random_component(component_probabilities):
    r = np.random.random(1)[0]
    accum = 0.0
    for i,p in enumerate(component_probabilities):
        accum += p
        if r <= accum:
            return i

def sigmoid(X):
    return 1/(1+np.exp(-X))

def relax_matrix(X, k=0.01):    
    props = softmax(numpy.random.rand(*X.shape))
    return X * (1.0-k) + k * props

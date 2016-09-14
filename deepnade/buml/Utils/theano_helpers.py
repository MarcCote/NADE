import numpy as np
import theano
import theano.tensor as T

checks = False
m1 = theano.shared(0)
m2 = theano.shared(0)
m3 = theano.shared(0)
m4 = theano.shared(0)
m5 = theano.shared(0)
m6 = theano.shared(0)
m7 = theano.shared(0)

floatX = theano.config.floatX  # @UndefinedVariable


def detect_nan(i, node, fn):
    '''
    x = theano.tensor.dscalar('x')
    f = theano.function([x], [theano.tensor.log(x) * x],
                        mode=theano.compile.MonitorMode(post_func=detect_nan))
    '''
    nan_detected = False
    for output in fn.outputs:
        if np.isnan(output[0]).any():
            nan_detected = True
            np.set_printoptions(threshold=np.nan)  # Print the whole arrays
            print '*** NaN detected ***'
            print '--------------------------NODE DESCRIPTION:'
            theano.printing.debugprint(node)
            print '--------------------------Variables:'
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break
    if nan_detected:
        exit()


def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype
    """
    return theano.tensor.constant(np.asarray(value, dtype=floatX))


def dropout(X, rate, rng):
    rnd = rng.uniform(X.shape)
    return X * (rnd > rate)


def log_sum_exp(x, axis=1):
    max_x = T.max(x, axis)
    return max_x + T.log(T.sum(T.exp(x - T.shape_padright(max_x, 1)), axis))

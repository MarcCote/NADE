import theano.tensor as T

import smartlearner.initializers as initer


WEIGHTS_INITIALIZERS = ["uniform", "zeros", "diagonal", "orthogonal", "gaussian"]


def weigths_initializer_factory(name, seed=1234):
    if name == "uniform":
        return initer.UniformInitializer(seed)
    elif name == "zeros":
        return initer.ZerosInitializer(seed)
    elif name == "diagonal":
        return initer.DiagonalInitializer(seed)
    elif name == "orthogonal":
        return initer.OrthogonalInitializer(seed)
    elif name == "gaussian":
        return initer.GaussienInitializer(seed)

    raise NotImplementedError("Unknown: " + str(name))


ACTIVATION_FUNCTIONS = ["sigmoid", "hinge", "softplus", "tanh"]


def activation_function_factory(name):
    if name == "sigmoid":
        return T.nnet.sigmoid
    elif name == "hinge":
        return lambda x: T.maximum(x, 0.0)
    elif name == "softplus":
        return T.nnet.softplus
    elif name == "tanh":
        return T.tanh

    raise NotImplementedError("Unknown: " + str(name))


def optimizer_factory(hyperparams, loss):
    #Set learning rate method that will be used.
    if hyperparams["SGD"] is not None:
        from smartlearner.optimizers import SGD
        from smartlearner.direction_modifiers import ConstantLearningRate
        options = hyperparams["SGD"].split()
        optimizer = SGD(loss=loss)
        optimizer.append_direction_modifier(ConstantLearningRate(lr=float(options[0])))
        return optimizer

    elif hyperparams["AdaGrad"] is not None:
        from smartlearner.optimizers import AdaGrad
        options = hyperparams["AdaGrad"].split()
        lr = float(options[0])
        eps = float(options[1]) if len(options) > 1 else 1e-6
        return AdaGrad(loss=loss, lr=lr, eps=eps)

    elif hyperparams["Adam"] is not None:
        from smartlearner.optimizers import Adam
        options = hyperparams["Adam"].split()
        lr = float(options[0]) if len(options) > 0 else 0.0001
        return Adam(loss=loss, lr=lr)

    elif hyperparams["RMSProp"] is not None:
        from smartlearner.optimizers import RMSProp
        lr = float(hyperparams["RMSProp"])
        return RMSProp(loss=loss, lr=lr)

    elif hyperparams["Adadelta"]:
        from smartlearner.optimizers import Adadelta
        return Adadelta(loss=loss)

    else:
        raise ValueError("The optimizer is mandatory!")

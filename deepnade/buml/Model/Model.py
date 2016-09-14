import numpy as np
import theano
import theano.tensor as T
from Utils.theano_helpers import floatX


class Model(object):
    def __init__(self):
        self.parameters = dict()
        self.parameters_to_optimise = list()
        self.parameters_to_regularise = list()

    def add_parameter(self, parameter, optimise=False, regularise=False):
        """
        Adds a parameter to the model
        """
        try:
            self.__getattribute__("parameters")
        except AttributeError:
            self.parameters = dict()
            # This should actually be a property of the trainingmethod, not of the model, but I'll do it like this now
            self.parameters_to_optimise = list()
            self.parameters_to_regularise = list()
        self.parameters[parameter.name] = parameter
        if optimise:
            self.parameters_to_optimise.append(parameter.name)
        if regularise:
            self.parameters_to_regularise.append(parameter.name)
        parameter.add_to_model(self)

    def get_parameter(self, param_id):
        return self.__getattribute__(param_id)

    def get_parameters(self):
        """
        Returns a hash with an entry per parameter, where the entry key is the parameter name and its value is a serialization of its value
        suitable for storing in an HDF5 file
        """
        params = dict([(name, parameter.get_value()) for name, parameter in self.parameters.iteritems()])
        params["__class__"] = self.__class__.__name__
        return params

    def set_parameters(self, params):
        """
        Receives a hash of the kind returned by get_parameters and updates the object's parameters with those values
        """
        for name in self.parameters.keys():
            self.parameters[name].set_value(params[name])

    def get_parameters_to_optimise(self):
        return self.parameters_to_optimise

    def get_parameters_to_regularise(self):
        return self.parameters_to_regularise

    def k_like_parameters_to_optimise(self, k, name):
        v = dict()
        for param in self.get_parameters_to_optimise():
            param_v = self.get_parameter(param).get_value()
            param_type = param_v.dtype
            v[param] = theano.shared(np.zeros_like(param_v, dtype=floatX) + np.asarray([k], dtype=param_type), name + "_" + str(param))
        return v

    def finite_diff_gradients(self, f, delta=1e-6):
        """
        f is called without parameters, the changes in the parameters happen as a side effect
        """
        gradients = dict()
        fx = f()
        for p in self.parameters_to_optimise:
            original = self.parameters[p].get_value()
            grad = np.zeros_like(original)
            if np.prod(original.shape) > 1:
                for index, _ in np.ndenumerate(original):
                    xh = original.copy()
                    xh[index] += delta
                    self.parameters[p].set_value(xh)
                    grad[index] = (f() - fx) / delta
                    self.parameters[p].set_value(original)
            else:
                xh = original.copy()
                xh += delta
                self.parameters[p].set_value(xh)
                grad = (f() - fx) / delta
                self.parameters[p].set_value(original)
            gradients[p] = grad
        return gradients


class CompositeModel(Model):
    def __init__(self):
        super(CompositeModel, self).__init__()
        self.models = dict()

    def add_model(self, name, model):
        self.models[name] = model

    def get_models(self):
        return self.models

    def get_parameter(self, param_id):
        if isinstance(param_id, tuple):
            return self.models[param_id[0]].__getattribute__(param_id[1])
        else:
            return self.__getattribute__(param_id)

    def get_parameters(self):
        """
        Returns a hash with an entry per parameter and submodel (recursively), where the entry key is the parameter name and its value is a serialization of its value
        suitable for storing in an HDF5 file
        """
        params = dict([(name, parameter.get_value()) for name, parameter in self.parameters.iteritems()])
        params["__class__"] = self.__class__.__name__
        for k, m in self.models.iteritems():
            params[k] = m.get_parameters()
        return params

    def set_parameters(self, params):
        """
        Receives a hash of the kind returned by get_parameters and updates the object's and submodel's parameters with those values
        """
        for k, v in self.parameters.iteritems():
            if isinstance(v, dict):
                self.models[k].set_parameters(v)
            else:
                self.parameters[k].set_value(v)

    def get_parameters_to_optimise(self):
        p = list()
        p += self.parameters_to_optimise
        for name, model in self.models.iteritems():
            p += [(name, param) for param in model.get_parameters_to_optimise()]
        return p

    def get_parameters_to_regularise(self):
        p = list()
        p += self.parameters_to_regularise
        for name, model in self.models.itervalues():
            p += [(name, param) for param in model.get_parameters_to_regularise()]
        return p


class Parameter(object):
    def __init__(self):
        pass

    def add_to_model(self, model):
        pass

    def set_value(self, value):
        pass

    def get_value(self):
        pass


class TensorParameter(Parameter):
    def __init__(self, name, shape, theano=True, theano_type=floatX):
        self.name = name
        self.shape = shape
        self.theano = theano
        self.theano_type = theano_type

    def add_to_model(self, model):
        self.model = model
        if self.theano:
            setattr(model, self.name, theano.shared(np.zeros(self.shape, dtype=self.theano_type), self.name))
        else:
            setattr(model, self.name, np.zeros(self.shape, dtype=self.theano_type))

    def set_value(self, value):
        if self.theano:
            self.model.__getattribute__(self.name).set_value(np.asarray(value).astype(self.theano_type))
        else:
            setattr(self.model, self.name, value)

    def get_value(self):
        if self.theano:
            return self.model.__getattribute__(self.name).get_value()
        else:
            return self.model.__getattribute__(self.name)


class ScalarParameter(Parameter):
    def __init__(self, name, default_value, theano=True, theano_type=floatX):
        self.name = name
        self.default_value = default_value
        self.theano = theano
        self.theano_type = theano_type

    def add_to_model(self, model):
        self.model = model
        if self.theano:
            setattr(model, self.name, theano.shared(np.array(self.default_value, dtype=self.theano_type), self.name))
        else:
            setattr(model, self.name, np.array(self.default_value, dtype=self.theano_type))

    def set_value(self, value):
        if self.theano:
            self.model.__getattribute__(self.name).set_value(np.asarray(value).astype(floatX))
        else:
            setattr(self.model, self.name, value)

    def get_value(self):
        if self.theano:
            return self.model.__getattribute__(self.name).get_value()
        else:
            return self.model.__getattribute__(self.name)


class SizeParameter(Parameter):
    def __init__(self, name):
        self.name = name

    def add_to_model(self, model):
        self.model = model
        setattr(model, self.name, 0)

    def set_value(self, value):
        setattr(self.model, self.name, value)

    def get_value(self):
        return self.model.__getattribute__(self.name)


class NonLinearityParameter(Parameter):
    def __init__(self, name):
        self.name = name
        self.options = {"tanh": [T.tanh, np.tanh],
                        "sigmoid": [T.nnet.sigmoid, lambda x: 1.0 / (1.0 + np.exp(-x))],
                        "RLU": [lambda x: x * (x > 0), lambda x: x * (x > 0)],
                        "softsign": [lambda x: x / (1 + T.abs_(x)), lambda x: x / (1 + np.abs(x))],
                        "exponential": [T.exp, np.exp]}

    def add_to_model(self, model):
        self.model = model
        self.value = self.options.items()[0][0]
        setattr(model, self.name, self.options.items()[0][1][0])

    def set_value(self, value):
        self.value = value
        setattr(self.model, self.name, self.options[value][0])

    def get_numpy_f(self):
        return self.options[self.value][1]

    def get_value(self):
        return self.value

    def get_name(self):
        return self.value


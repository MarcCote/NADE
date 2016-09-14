import theano
import types
import numpy as np
from Instrumentation import Instrumentable


def has_parameter(param_name, default_value=None, theano_param=False, theano_type=theano.config.floatX):  # @UndefinedVariable
    ''''Decorator used to add parameters to a training method.
    '''
    def decorator(cls):
        original_init = cls.__init__
        if theano_param:
            def getter(self):
                return self.__getattribute__(param_name).get_value()

            def setter(self, value):
                return self.__getattribute__(param_name).set_value(np.array(value,
                                                                 dtype=theano_type))

            def new_init(self, *args, **kwargs):
                setattr(self, param_name, theano.shared(np.array(default_value,
                                                                 dtype=theano_type),
                                                        param_name))
                setattr(self, "get_" + param_name, types.MethodType(getter, self, self.__class__))
                setattr(self, "set_" + param_name, types.MethodType(setter, self, self.__class__))
                original_init(self, *args, **kwargs)
        else:
            def getter(self):
                return self.__getattribute__(param_name)

            def setter(self, value):
                return self.__setattr__(param_name, value)

            def new_init(self, *args, **kwargs):
                setattr(self, param_name, default_value)
                setattr(self, "get_" + param_name, types.MethodType(getter, self, self.__class__))
                setattr(self, "set_" + param_name, types.MethodType(setter, self, self.__class__))
                original_init(self, *args, **kwargs)
        setattr(cls, "__init__", new_init)
        try:
            setattr(cls, "parameters", dict(getattr(cls, "parameters").items() + [(param_name, getter)]))
        except AttributeError:
            setattr(cls, "parameters", {param_name: getter})
        return cls
    return decorator


@has_parameter("datasets", None)
class Optimizer(Instrumentable):
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        self.controllers = []
        self.finished = False
        self.context_name = "training"
        self.success = True
        self.datapoints_as_columns = False
        Instrumentable.__init__(self)

    # Get training context
    def get_context(self):
        return [self.context_name]

    def set_context(self, context_name):
        self.context_name = context_name

    # Training controllers
    def add_controller(self, controller):
        self.controllers.append(controller)

    def is_training_finished(self):
        return self.finished

    def stop_training(self):
        self.finished = True

    def before_training(self):
        self.finished = False
        # Initialize controllers
        for c in self.controllers:
            c.before_training(self)
        self.initialize_instrumentation()

    def after_training(self):
        self.end_instrumentation()

    def before_training_iteration(self):
        for c in self.controllers:
            c.before_training_iteration(self)

    def after_training_iteration(self):
        self.run_instrumentation()
        self.finished = reduce(lambda t, c: t or c.after_training_iteration(self), self.controllers, False)

    def get_parameters(self):
        ''' Returns a dictionary with all the training method parameters '''
        return dict([(k, v(self)) for k, v in self.__class__.parameters.iteritems()])

    def was_successful(self):
        return self.success

    def set_datapoints_as_columns(self, value):
        self.datapoints_as_columns = value

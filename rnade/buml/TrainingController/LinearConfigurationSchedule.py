from TrainingController import TrainingController


class LinearConfigurationSchedule(TrainingController):
    def __init__(self, parameter_name, initial_value, increment):
        self.parameter_name = parameter_name
        self.current_value = initial_value
        self.increment = increment

    def before_training_iteration(self, training_method):
        training_method.__getattribute__("set_" + self.parameter_name)(self.current_value)

    def after_training_iteration(self, training_method):
        self.current_value += self.increment

from TrainingController import TrainingController


class ConfigurationSchedule(TrainingController):
    def __init__(self, parameter_name, schedule):
        self.parameter_name = parameter_name
        self.schedule = schedule
        self.index = 0

    def before_training_iteration(self, training_method):
        for i in xrange(len(self.schedule)):
            if training_method.epoch < self.schedule[i][0]:
                break
        training_method.__getattribute__("set_" + self.parameter_name)(self.schedule[i][1])

    def after_training_iteration(self, training_method):
        return False

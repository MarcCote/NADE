from Optimization import has_parameter


@has_parameter("updates_per_epoch", 1000)
class Epochable(object):
    def __init__(self):
        self.epoch = 0

    def get_context(self):
        context = super(Epochable, self).get_context()
        context.append(self.epoch)
        return context
#        return [self.get_context(), self.epoch]
#         return ["training", self.epoch]

    def before_training_iteration(self):
        self.epoch += 1
        super(Epochable, self).before_training_iteration()

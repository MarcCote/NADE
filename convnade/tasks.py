import pickle
import numpy as np
from os.path import join as pjoin

import theano

from smartlearner.tasks import Task


class DeepNadeOrderingTask(Task):
    """ This task changes the ordering before each update. """
    def __init__(self, D, batch_size, ordering_seed=1234):
        super(DeepNadeOrderingTask, self).__init__()
        self.rng = np.random.RandomState(ordering_seed)
        self.batch_size = batch_size
        self.D = D
        self.ordering_mask = theano.shared(np.zeros((batch_size, D), dtype=theano.config.floatX), name='ordering_mask', borrow=False)

    def pre_update(self, status):
        # Thanks to the broadcasting and `np.apply_along_axis`, we easily
        # generate `batch_size` orderings and compute their corresponding
        # $o_{<d}$ mask.
        d = self.rng.randint(self.D, size=(self.batch_size, 1))
        masks_o_lt_d = np.arange(self.D) < d
        map(self.rng.shuffle, masks_o_lt_d)  # Inplace shuffling each row.
        self.ordering_mask.set_value(masks_o_lt_d)

    def save(self, savedir="./"):
        filename = pjoin(savedir, "DeepNadeOrderingTask.pkl")
        pickle.dump(self.rng, open(filename, 'w'))

    def load(self, loaddir="./"):
        filename = pjoin(loaddir, "DeepNadeOrderingTask.pkl")
        self.rng = pickle.load(open(filename))


class DeepNadeTrivialOrderingsTask(Task):
    """ This task changes the ordering before each update.

    The ordering are sampled from the 8 trivial orderings.
    """
    def __init__(self, image_shape, batch_size, ordering_seed=1234):
        super(DeepNadeTrivialOrderingsTask, self).__init__()
        self.rng = np.random.RandomState(ordering_seed)
        self.batch_size = batch_size
        self.D = int(np.prod(image_shape))
        self.mask_o_d = theano.shared(np.zeros((batch_size, self.D), dtype=theano.config.floatX), name='mask_o_d', borrow=False)
        self.mask_o_lt_d = theano.shared(np.zeros((batch_size, self.D), dtype=theano.config.floatX), name='mask_o_lt_d', borrow=False)
        self.ordering_mask = self.mask_o_lt_d

        self.orderings = []
        base_ordering = np.arange(self.D).reshape(image_shape)
        # 8 trivial orderings
        # Top-left to bottom-right (row-major)
        self.orderings.append(base_ordering.flatten("C"))
        # Top-right to bottom-left (row-major)
        self.orderings.append(base_ordering[:, ::-1].flatten("C"))
        # Bottom-left to top-right (row-major)
        self.orderings.append(base_ordering[::-1, :].flatten("C"))
        # Bottom-right to top-left (row-major)
        self.orderings.append(base_ordering[::-1, ::-1].flatten("C"))
        # Top-left to bottom-right (column-major)
        self.orderings.append(base_ordering.flatten("F"))
        # Top-right to bottom-left (column-major)
        self.orderings.append(base_ordering[:, ::-1].flatten("F"))
        # Bottom-left to top-right (column-major)
        self.orderings.append(base_ordering[::-1, :].flatten("F"))
        # Bottom-right to top-left (column-major)
        self.orderings.append(base_ordering[::-1, ::-1].flatten("F"))

    def pre_update(self, status):
        # Compute the next $o_{<d}$ mask.
        idx_ordering = self.rng.randint(8)
        d = self.rng.randint(self.D, size=(self.batch_size, 1))
        masks_o_d = self.orderings[idx_ordering] == d
        masks_o_lt_d = self.orderings[idx_ordering] < d
        self.mask_o_d.set_value(masks_o_d)
        self.mask_o_lt_d.set_value(masks_o_lt_d)

    def save(self, savedir="./"):
        filename = pjoin(savedir, "DeepNadeOrderingTask.pkl")
        pickle.dump(self.rng, open(filename, 'w'))

    def load(self, loaddir="./"):
        filename = pjoin(loaddir, "DeepNadeOrderingTask.pkl")
        self.rng = pickle.load(open(filename))

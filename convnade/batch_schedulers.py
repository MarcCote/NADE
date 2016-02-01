import theano
import pickle
import numpy as np
import theano.tensor as T
from os.path import join as pjoin

from smartlearner.batch_schedulers import MiniBatchScheduler
from smartlearner.utils import sharedX

floatX = theano.config.floatX


class MiniBatchSchedulerWithAutoregressiveMask(MiniBatchScheduler):
    """ Batch of padded examples.
    """
    def __init__(self, dataset, batch_size, concatenate_mask=False, keep_mask=False, seed=1234):
        """
        Parameters
        ----------
        dataset : `SequenceDataset` object
            Dataset of datasets (one for each bundle).
        batch_size : int
            Number of examples per batch. *Must be greater than the number of
            bundles in `bundles_dataset`.*
        seed : int (optional)
            Seed of the random numbers generator used to sample a different
            regressive mask for each example.
        """
        super().__init__(dataset, batch_size)

        self.concatenate_mask = concatenate_mask
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.keep_mask = keep_mask

        # Allocate memory for the autoregressive mask.
        self.mask_shape = (len(dataset),) + self.dataset.input_shape
        self._shared_batch_mask = sharedX(np.zeros(self.mask_shape), name='autoregressive_mask', keep_on_cpu=True)

        # Add a new attribute: a symbolic variable representing the auto regressive mask.
        self._shared_batch_mask.set_value(self.generate_autoregressive_mask())
        # TODO: the symolic variable should be in dataset.
        self.dataset.symb_mask = T.TensorVariable(type=T.TensorType("floatX", [False]*dataset.inputs.ndim), name=dataset.name+'_symb_mask')

        # Keep only `batch_size` masks as test values.
        self.dataset.symb_mask.tag.test_value = self._shared_batch_mask.get_value()[:batch_size]  # For debugging Theano graphs.

        if self.concatenate_mask:
            self.dataset.symb_inputs.tag.test_value = np.concatenate([self.dataset.symb_inputs.tag.test_value,  # * self._shared_batch_mask[start:end],
                                                                      self.dataset.symb_mask.tag.test_value], axis=1)

    @property
    def updates(self):
        return {}  # No updates

    @property
    def givens(self):
        start = self.shared_batch_count * self._shared_batch_size
        end = (self.shared_batch_count + 1) * self._shared_batch_size

        if self.concatenate_mask:

            return {self.dataset.symb_inputs: T.concatenate([self.dataset.inputs[start:end] * self._shared_batch_mask[start:end],
                                                             self._shared_batch_mask[start:end]], axis=1),
                    self.dataset.symb_targets: self.dataset.targets[start:end],
                    self.dataset.symb_mask: self._shared_batch_mask[start:end]}

        return {self.dataset.symb_inputs: self.dataset.inputs[start:end] * self._shared_batch_mask[start:end],
                self.dataset.symb_targets: self.dataset.targets[start:end],
                self.dataset.symb_mask: self._shared_batch_mask[start:end]}

    def generate_autoregressive_mask(self):
        # Thanks to the broadcasting and `np.apply_along_axis`, we easily
        # generate `batch_size` orderings and compute their corresponding
        # $o_{<d}$ mask.
        D = self.mask_shape[1]
        d = self.rng.randint(D, size=(self.mask_shape[0], 1))
        masks_o_lt_d = np.arange(D) < d
        list(map(self.rng.shuffle, masks_o_lt_d))  # Inplace shuffling each row.
        return masks_o_lt_d

    def __iter__(self):
        for batch_count in range(self.nb_updates_per_epoch):
            self.shared_batch_count.set_value(batch_count)
            yield batch_count + 1

        # Once per epoch, change the autoregressive mask for all examples in the dataset.
        if not self.keep_mask:
            self._shared_batch_mask.set_value(self.generate_autoregressive_mask())

    def save(self, savedir):
        state = {"version": 1,
                 "seed": self.seed,
                 "concatenate_mask": self.concatenate_mask,
                 "batch_size": self.batch_size,
                 "shared_batch_count": self.shared_batch_count.get_value(),
                 "rng": pickle.dumps(self.rng),
                 "shared_batch_mask": self._shared_batch_mask.get_value(),
                 }

        np.savez(pjoin(savedir, 'mini_batch_scheduler_with_autoregressive_mask.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, 'mini_batch_scheduler_with_autoregressive_mask.npz'))
        self.batch_size = state["batch_size"]
        self.shared_batch_count.set_value(state["shared_batch_count"])
        self.rng = pickle.loads(state["rng"])
        self._shared_batch_mask.set_value(state["shared_batch_mask"])

import theano
import pickle
import numpy as np
import theano.tensor as T
from os.path import join as pjoin

from smartlearner.batch_schedulers import BatchScheduler, MiniBatchScheduler
from smartlearner.utils import sharedX

floatX = theano.config.floatX


class MiniBatchSchedulerWithAutoregressiveMask(MiniBatchScheduler):
    """ Batch of padded examples.
    """
    def __init__(self, dataset, batch_size, use_mask_as_input=False, keep_mask=False, seed=1234):
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

        self.use_mask_as_input = use_mask_as_input
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.keep_mask = keep_mask

        # Allocate memory for the autoregressive mask.
        self.mask_shape = (len(dataset),) + self.dataset.input_shape
        self._shared_mask_o_lt_d = sharedX(np.zeros(self.mask_shape), name='autoregressive_mask', keep_on_cpu=True)

        # Add a new attribute: a symbolic variable representing the auto regressive mask.
        self._shared_mask_o_lt_d.set_value(self.generate_autoregressive_mask())
        self.dataset.mask_o_lt_d = T.TensorVariable(type=T.TensorType("floatX", [False]*dataset.inputs.ndim), name=dataset.name+'_symb_mask')

        # Keep only `batch_size` masks as test values.
        self.dataset.mask_o_lt_d.tag.test_value = self._shared_mask_o_lt_d.get_value()[:batch_size]  # For debugging Theano graphs.

        if self.use_mask_as_input:
            self.dataset.symb_inputs.tag.test_value = np.concatenate([self.dataset.symb_inputs.tag.test_value * self.dataset.mask_o_lt_d.tag.test_value,
                                                                      self.dataset.mask_o_lt_d.tag.test_value], axis=1)

    @property
    def updates(self):
        return {}  # No updates

    @property
    def givens(self):
        start = self.shared_batch_count * self._shared_batch_size
        end = (self.shared_batch_count + 1) * self._shared_batch_size

        if self.use_mask_as_input:
            return {self.dataset.symb_inputs: T.concatenate([self.dataset.inputs[start:end] * self._shared_mask_o_lt_d[start:end],
                                                             self._shared_mask_o_lt_d[start:end]], axis=1),
                    self.dataset.symb_targets: self.dataset.targets[start:end],
                    self.dataset.mask_o_lt_d: self._shared_mask_o_lt_d[start:end]}

        return {self.dataset.symb_inputs: self.dataset.inputs[start:end] * self._shared_mask_o_lt_d[start:end],
                self.dataset.symb_targets: self.dataset.targets[start:end],
                self.dataset.mask_o_lt_d: self._shared_mask_o_lt_d[start:end]}

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
            self._shared_mask_o_lt_d.set_value(self.generate_autoregressive_mask())

    def save(self, savedir):
        state = {"version": 1,
                 "seed": self.seed,
                 "use_mask_as_input": self.use_mask_as_input,
                 "batch_size": self.batch_size,
                 "shared_batch_count": self.shared_batch_count.get_value(),
                 "rng": pickle.dumps(self.rng),
                 "shared_batch_mask": self._shared_mask_o_lt_d.get_value(),
                 }

        np.savez(pjoin(savedir, 'mini_batch_scheduler_with_autoregressive_mask.npz'), **state)

    def load(self, loaddir):
        state = np.load(pjoin(loaddir, 'mini_batch_scheduler_with_autoregressive_mask.npz'))
        self.batch_size = state["batch_size"]
        self.shared_batch_count.set_value(state["shared_batch_count"])
        self.rng = pickle.loads(state["rng"])
        self._shared_mask_o_lt_d.set_value(state["shared_batch_mask"])

    def __len__(self):
        return self.nb_updates_per_epoch


class BatchSchedulerWithAutoregressiveMasks(BatchScheduler):
    """ Batch of padded examples.
    """
    def __init__(self, dataset, batch_size, batch_id, ordering_id, use_mask_as_input=False, seed=1234):
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
        super().__init__(dataset)
        self.use_mask_as_input = use_mask_as_input
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.batch_size = batch_size
        self.batch_id = batch_id
        self.ordering_id = ordering_id

        # Determine the start and the end of the batch that will be used by this batch scheduler.
        assert batch_id*self.batch_size < len(self.dataset)
        self.batch_start = batch_id*self.batch_size
        self.batch_end = min((batch_id+1)*self.batch_size, len(dataset))

        # Determine the ordering that will be used by this batch scheduler.
        self.d = 0
        self.D = self.dataset.input_shape[0]
        self.ordering = np.arange(self.D)
        for _ in range(ordering_id+1):
            self.rng.shuffle(self.ordering)

        # Matrix mask that will be used when concatenating the mask.
        self._shared_Moltd = sharedX(np.zeros((self.batch_end-self.batch_start, self.D)), name='Moltd')

        # Vector mask that will be broadcasted across all inputs.
        # self._shared_mod = sharedX(np.zeros((1, self.D)), name='mod')
        self._shared_mod = sharedX(np.zeros((self.D,)), name='mod')

        # Add a new attributes: a symbolic variable representing the auto regressive mask.
        self.change_masks(self.d)
        self.Moltd = T.TensorVariable(type=T.TensorType("floatX", [False]*dataset.inputs.ndim), name="symb_Moltd")
        self.mod = T.TensorVariable(type=T.TensorType("floatX", [True, False]), name="symb_mod")

        # Keep only `(self.batch_end-self.batch_start)` examples as test values.
        self.dataset.symb_inputs.tag.test_value = self.dataset.inputs.get_value()[:(self.batch_end-self.batch_start)]
        if self.dataset.has_targets:
            self.dataset.symb_targets.tag.test_value = self.dataset.targets.get_value()[:(self.batch_end-self.batch_start)]

        self.Moltd.tag.test_value = self._shared_Moltd.get_value()[:(self.batch_end-self.batch_start)]
        self.mod.tag.test_value = self._shared_mod.get_value()[None, :]

        if self.use_mask_as_input:
            self.dataset.symb_inputs.tag.test_value = np.concatenate([self.dataset.symb_inputs.tag.test_value * self.Moltd.tag.test_value,
                                                                      self.Moltd.tag.test_value], axis=1)

    @property
    def updates(self):
        return {}  # No updates

    @property
    def givens(self):
        if self.use_mask_as_input:
            return {self.dataset.symb_inputs: T.concatenate([self.dataset.inputs[self.batch_start:self.batch_end] * self._shared_Moltd,
                                                             self._shared_Moltd[self.batch_start:self.batch_end]], axis=1),
                    self.dataset.symb_targets: self.dataset.targets[self.batch_start:self.batch_end],
                    self.Moltd: self._shared_Moltd,
                    self.mod: self._shared_mod[None, :]}

        return {self.dataset.symb_inputs: self.dataset.inputs[self.batch_start:self.batch_end] * self._shared_Moltd,
                self.dataset.symb_targets: self.dataset.targets[self.batch_start:self.batch_end],
                self.Moltd: self._shared_Moltd,
                self.mod: self._shared_mod[None, :]}

    def change_masks(self, d):
        # self._shared_mod.set_value((self.ordering == d).astype(theano.config.floatX)[None, :])
        self._shared_mod.set_value((self.ordering == d).astype(theano.config.floatX))
        moltd = (self.ordering < d).astype(theano.config.floatX)
        self._shared_Moltd.set_value(np.tile(moltd, (self.batch_size, 1)))

    def __iter__(self):
        for d in range(self.D):
            # print("{}/{}".format(d+1, self.D))
            self.change_masks(d)
            yield d + 1

    def __len__(self):
        return self.D

    # def save(self, savedir):
    #     state = {"version": 1,
    #              "seed": self.seed,
    #              "use_mask_as_input": self.use_mask_as_input,
    #              "batch_size": self.batch_size,
    #              "shared_batch_count": self.shared_batch_count.get_value(),
    #              "rng": pickle.dumps(self.rng),
    #              "shared_batch_mask": self._shared_mask_o_lt_d.get_value(),
    #              }

    #     np.savez(pjoin(savedir, 'mini_batch_scheduler_with_autoregressive_mask.npz'), **state)

    # def load(self, loaddir):
    #     state = np.load(pjoin(loaddir, 'mini_batch_scheduler_with_autoregressive_mask.npz'))
    #     self.batch_size = state["batch_size"]
    #     self.shared_batch_count.set_value(state["shared_batch_count"])
    #     self.rng = pickle.loads(state["rng"])
    #     self._shared_mask_o_lt_d.set_value(state["shared_batch_mask"])

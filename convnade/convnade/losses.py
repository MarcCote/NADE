import numpy as np
import theano.tensor as T

from smartlearner import Loss
from smartlearner.views import ItemGetter


#class NllEstimateUsingBinaryCrossEntropyWithAutoRegressiveMask(Loss):
class BinaryCrossEntropyEstimateWithAutoRegressiveMask(Loss):
    """ NLL estimate for a Deep NADE model where an auto regressive mask has been applied to the inputs.

    Notes
    -----
    This loss function assume the dataset has an attribute named `mask_o_lt_d` which is a symbolic variable representing
    the $d-1$ first dimensions in the ordering that are allowed to be used i.e. $x_i : i \in o_{<d}$.
    """
    def _get_updates(self):
        return {}  # There is no special updates for this loss.

    def _compute_losses(self, model_output):
        # We assume model_ouput is the preactivation.
        output = T.nnet.sigmoid(model_output)
        cross_entropies = T.nnet.binary_crossentropy(output, self.dataset.symb_targets)
        cross_entropies_masked = cross_entropies * (1-self.dataset.mask_o_lt_d)
        nll_estimate = T.sum(cross_entropies_masked, axis=1)

        # We unbias the NLL estimate as mentioned in Uria et al.
        D = np.float32(np.prod(self.model.image_shape))  # Scaling factor
        d = T.sum(self.dataset.mask_o_lt_d, axis=1)
        weighted_nll_estimate = nll_estimate * (D / (D-d+1))

        return weighted_nll_estimate


class NllUsingBinaryCrossEntropyWithAutoRegressiveMask(Loss):
    """ NLL for a Deep NADE model where an auto regressive mask has been applied to the inputs.

    Notes
    -----
    This loss function assume the dataset has an attribute named `mask_o_d` which is a symbolic variable representing
    the $d-1$ first dimensions in the ordering that are allowed to be used i.e. $x_i : i \in o_{<d}$.
    """
    def __init__(self, model, dataset, mod):
        super().__init__(model, dataset)
        self.mod = mod

    def _get_updates(self):
        return {}  # There is no special updates for this loss.

    def _compute_losses(self, model_output):
        # We assume model_ouput is the preactivation.
        output = T.nnet.sigmoid(model_output)
        cross_entropies = T.nnet.binary_crossentropy(output, self.dataset.symb_targets)

        # Keep only the d-th conditional
        cross_entropies_masked = cross_entropies * self.mod
        ln_p_xod_given_xoltd = -T.sum(cross_entropies_masked, axis=1)  # Keep only the d-th conditional.
        nll_xod_given_xoltd = -ln_p_xod_given_xoltd
        return nll_xod_given_xoltd


# class EvaluateDeepNadeNLLEstimateOnTrivial(Loss):
#     """ This tasks compute the mean/stderr NLL estimate for a Deep NADE model.  """
#     def __init__(self, conv_nade, dataset, batch_size=None, ordering_seed=1234):

#         dataset_shared = dataset
#         if isinstance(dataset, np.ndarray):
#             dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

#         if batch_size is None:
#             batch_size = len(dataset_shared.get_value())

#         nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

#         # Pre-generate the orderings that will be used to estimate the NLL of the Deep NADE model.
#         D = int(np.prod(conv_nade.image_shape))
#         ordering_task = DeepNadeTrivialOrderingsTask(conv_nade.image_shape, len(dataset_shared.get_value()), ordering_seed)

#         # $X$: batch of inputs (flatten images)
#         input = T.matrix('input')
#         mask_o_d = theano.shared(np.zeros((batch_size, D), dtype=theano.config.floatX), name='mask_o_d', borrow=False)
#         mask_o_lt_d = theano.shared(np.zeros((batch_size, D), dtype=theano.config.floatX), name='mask_o_lt_d', borrow=False)
#         loss = T.mean(-conv_nade.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d))

#         no_batch = T.iscalar('no_batch')
#         givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size]}
#         compute_loss = theano.function([no_batch], loss, givens=givens, name="NLL Estimate")
#         #theano.printing.pydotprint(compute_loss, '{0}_compute_nll_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

#         def _nll_mean_and_std():
#             nlls = np.zeros(len(dataset_shared.get_value()))
#             for i in range(nb_batches):
#                 # Hack: Change ordering mask in the model before computing the NLL estimate.
#                 mask_o_d.set_value(ordering_task.mask_o_d.get_value()[i*batch_size:(i+1)*batch_size])
#                 mask_o_lt_d.set_value(ordering_task.mask_o_lt_d.get_value()[i*batch_size:(i+1)*batch_size])
#                 nlls[i*batch_size:(i+1)*batch_size] = compute_loss(i)

#             return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

#         super(EvaluateDeepNadeNLLEstimateOnTrivial, self).__init__(_nll_mean_and_std)

#     @property
#     def mean(self):
#         return ItemGetter(self, attribute=0)

#     @property
#     def std(self):
#         return ItemGetter(self, attribute=1)


# class EvaluateDeepNadeNLLEstimate(Loss):
#     """ This tasks compute the NLL estimate for a Deep NADE model.  """
#     def __init__(self, conv_nade, dataset, ordering_mask, batch_size=None, ordering_seed=1234):

#         dataset_shared = dataset
#         if isinstance(dataset, np.ndarray):
#             dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

#         if batch_size is None:
#             batch_size = len(dataset_shared.get_value())

#         nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

#         # Pre-generate the orderings that will be used to estimate the NLL of the Deep NADE model.
#         rng = np.random.RandomState(ordering_seed)
#         D = dataset_shared.get_value().shape[1]
#         d = rng.randint(D, size=(len(dataset_shared.get_value()), 1))
#         masks_o_lt_d = np.arange(D) < d
#         map(rng.shuffle, masks_o_lt_d)  # Inplace shuffling along axis=1.

#         # $X$: batch of inputs (flatten images)
#         input = T.matrix('input')
#         loss = conv_nade.mean_nll_estimate_loss(input, ordering_mask)
#         no_batch = T.iscalar('no_batch')
#         givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size]}
#         compute_loss = theano.function([no_batch], loss, givens=givens, name="NLL Estimate")
#         #theano.printing.pydotprint(compute_loss, '{0}_compute_nll_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

#         def _nll_mean_and_std():
#             nlls = np.zeros(len(dataset_shared.get_value()))
#             for i in range(nb_batches):
#                 # Hack: Change ordering mask in the model before computing the NLL estimate.
#                 ordering_mask.set_value(masks_o_lt_d[i*batch_size:(i+1)*batch_size])
#                 nlls[i*batch_size:(i+1)*batch_size] = compute_loss(i)

#             return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

#         super(EvaluateDeepNadeNLLEstimate, self).__init__(_nll_mean_and_std)

#     @property
#     def mean(self):
#         return ItemGetter(self, attribute=0)

#     @property
#     def std(self):
#         return ItemGetter(self, attribute=1)


# class EvaluateDeepNadeNLLParallel(Evaluate):
#     """ This tasks compute the mean/stderr NLL (averaged across multiple orderings) for a Deep NADE model.

#     Notes
#     -----
#     This is slow but tractable.
#     """

#     def __init__(self, conv_nade, dataset,
#                  batch_size=None, no_part=1, nb_parts=1,
#                  no_ordering=None, nb_orderings=8, orderings_seed=None):

#         print("Part: {}/{}".format(no_part, nb_parts))
#         part_size = int(np.ceil(len(dataset.get_value()) / nb_parts))
#         dataset = dataset.get_value()[(no_part-1)*part_size:no_part*part_size]

#         dataset = theano.shared(dataset, name='dataset', borrow=True)

#         if batch_size is None:
#             batch_size = len(dataset.get_value())

#         #batch_size = min(batch_size, part_size)
#         nb_batches = int(np.ceil(len(dataset.get_value()) / batch_size))

#         # Generate the orderings that will be used to evaluate the Deep NADE model.
#         D = dataset.get_value().shape[1]
#         orderings = []
#         if orderings_seed is None:
#             base_ordering = np.arange(D).reshape(conv_nade.image_shape)
#             # 8 trivial orderings
#             # Top-left to bottom-right (row-major)
#             orderings.append(base_ordering.flatten("C"))
#             # Top-right to bottom-left (row-major)
#             orderings.append(base_ordering[:, ::-1].flatten("C"))
#             # Bottom-left to top-right (row-major)
#             orderings.append(base_ordering[::-1, :].flatten("C"))
#             # Bottom-right to top-left (row-major)
#             orderings.append(base_ordering[::-1, ::-1].flatten("C"))
#             # Top-left to bottom-right (column-major)
#             orderings.append(base_ordering.flatten("F"))
#             # Top-right to bottom-left (column-major)
#             orderings.append(base_ordering[:, ::-1].flatten("F"))
#             # Bottom-left to top-right (column-major)
#             orderings.append(base_ordering[::-1, :].flatten("F"))
#             # Bottom-right to top-left (column-major)
#             orderings.append(base_ordering[::-1, ::-1].flatten("F"))
#         else:
#             rng = np.random.RandomState(orderings_seed)
#             for i in range(nb_orderings):
#                 ordering = np.arange(D)
#                 rng.shuffle(ordering)
#                 orderings.append(ordering)

#         if no_ordering is not None:
#             orderings = [orderings[no_ordering]]

#         masks_o_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_d', borrow=True)
#         masks_o_lt_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_lt_d', borrow=True)

#         # Build theano function
#         # $X$: batch of inputs (flatten images)
#         input = T.matrix('input')
#         # $o_d$: index of d-th dimension in the ordering.
#         mask_o_d = T.vector('mask_o_d')
#         # $o_{<d}$: indices of the d-1 first dimensions in the ordering.
#         mask_o_lt_d = T.vector('mask_o_lt_d')

#         lnp_x_o_d_given_x_o_lt_d = conv_nade.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d)

#         no_batch = T.iscalar('no_batch')
#         d = T.iscalar('d')
#         givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size],
#                   mask_o_d: masks_o_d[d],
#                   mask_o_lt_d: masks_o_lt_d[d]}
#         compute_lnp_x_o_d_given_x_o_lt_d = theano.function([no_batch, d], lnp_x_o_d_given_x_o_lt_d, givens=givens, name="nll_of_x_o_d_given_x_o_lt_d")
#         #theano.printing.pydotprint(compute_lnp_x_o_d_given_x_o_lt_d, '{0}_compute_lnp_x_o_d_given_x_o_lt_d_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

#         def _nll():
#             nlls = -np.inf * np.ones(len(dataset.get_value()))
#             for o, ordering in enumerate(orderings):
#                 o_d = np.zeros((D, D), dtype=theano.config.floatX)
#                 o_d[np.arange(D), ordering] = 1
#                 masks_o_d.set_value(o_d)

#                 o_lt_d = np.cumsum(o_d, axis=0)
#                 o_lt_d[1:] = o_lt_d[:-1]
#                 o_lt_d[0, :] = 0
#                 masks_o_lt_d.set_value(o_lt_d)

#                 for i in range(nb_batches):
#                     print("Batch: {0}/{1}".format(i+1, nb_batches))
#                     ln_dth_conditionals = []
#                     start = time()
#                     for d in range(D):
#                         if d % 100 == 0:
#                             print("{0}/{1} dth conditional ({2:.2f} sec.)".format(d, D, time()-start))
#                             start = time()

#                         ln_dth_conditionals.append(compute_lnp_x_o_d_given_x_o_lt_d(i, d))

#                     from ipdb import set_trace as dbg
#                     dbg()

#                     # We average p(x) on different orderings, if needed.
#                     nlls[i*batch_size:(i+1)*batch_size] = np.logaddexp(nlls[i*batch_size:(i+1)*batch_size],
#                                                                        -np.sum(np.vstack(ln_dth_conditionals).T, axis=1))

#             nlls -= np.log(len(orderings))  # Average across all orderings
#             return nlls

#         super(EvaluateDeepNadeNLLParallel, self).__init__(_nll)


# class EvaluateDeepNadeNLL(Evaluate):
#     """ This tasks compute the mean/stderr NLL (averaged across multiple orderings) for a Deep NADE model.

#     Notes
#     -----
#     This is slow but tractable.
#     """

#     def __init__(self, conv_nade, dataset, batch_size=None, nb_orderings=10, ordering_seed=1234):

#         dataset_shared = dataset
#         if isinstance(dataset, np.ndarray):
#             dataset_shared = theano.shared(dataset, name='dataset', borrow=True)

#         if batch_size is None:
#             batch_size = len(dataset_shared.get_value())

#         nb_batches = int(np.ceil(len(dataset_shared.get_value()) / batch_size))

#         # Generate the orderings that will be used to evaluate the Deep NADE model.
#         D = dataset_shared.get_value().shape[1]
#         orderings = []
#         if nb_orderings > 0:
#             rng = np.random.RandomState(ordering_seed)
#             for i in range(nb_orderings):
#                 ordering = np.arange(D)
#                 rng.shuffle(ordering)
#                 orderings.append(ordering)

#         elif nb_orderings == 0:
#             base_ordering = np.arange(D).reshape(conv_nade.image_shape)
#             # 8 trivial orderings
#             # Top-left to bottom-right (row-major)
#             orderings.append(base_ordering.flatten("C"))
#             # Top-right to bottom-left (row-major)
#             orderings.append(base_ordering[:, ::-1].flatten("C"))
#             # Bottom-left to top-right (row-major)
#             orderings.append(base_ordering[::-1, :].flatten("C"))
#             # Bottom-right to top-left (row-major)
#             orderings.append(base_ordering[::-1, ::-1].flatten("C"))
#             # Top-left to bottom-right (column-major)
#             orderings.append(base_ordering.flatten("F"))
#             # Top-right to bottom-left (column-major)
#             orderings.append(base_ordering[:, ::-1].flatten("F"))
#             # Bottom-left to top-right (column-major)
#             orderings.append(base_ordering[::-1, :].flatten("F"))
#             # Bottom-right to top-left (column-major)
#             orderings.append(base_ordering[::-1, ::-1].flatten("F"))
#         else:
#             raise ValueError("Unknown value for 'nb_orderings': {0}".format(nb_orderings))

#         masks_o_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_d', borrow=True)
#         masks_o_lt_d = theano.shared(np.array([], ndmin=2, dtype=theano.config.floatX), name='masks_o_lt_d', borrow=True)

#         # Build theano function
#         # $X$: batch of inputs (flatten images)
#         input = T.matrix('input')
#         # $o_d$: index of d-th dimension in the ordering.
#         mask_o_d = T.vector('mask_o_d')
#         # $o_{<d}$: indices of the d-1 first dimensions in the ordering.
#         mask_o_lt_d = T.vector('mask_o_lt_d')

#         lnp_x_o_d_given_x_o_lt_d = conv_nade.lnp_x_o_d_given_x_o_lt_d(input, mask_o_d, mask_o_lt_d)

#         no_batch = T.iscalar('no_batch')
#         d = T.iscalar('d')
#         givens = {input: dataset[no_batch * batch_size:(no_batch + 1) * batch_size],
#                   mask_o_d: masks_o_d[d],
#                   mask_o_lt_d: masks_o_lt_d[d]}
#         compute_lnp_x_o_d_given_x_o_lt_d = theano.function([no_batch, d], lnp_x_o_d_given_x_o_lt_d, givens=givens, name="nll_of_x_o_d_given_x_o_lt_d")
#         #theano.printing.pydotprint(compute_lnp_x_o_d_given_x_o_lt_d, '{0}_compute_lnp_x_o_d_given_x_o_lt_d_{1}'.format(conv_nade.__class__.__name__, theano.config.device), with_ids=True)

#         def _nll_mean_and_std():
#             nlls = -np.inf * np.ones(len(dataset_shared.get_value()))
#             for o, ordering in enumerate(orderings):
#                 o_d = np.zeros((D, D), dtype=theano.config.floatX)
#                 o_d[np.arange(D), ordering] = 1
#                 masks_o_d.set_value(o_d)

#                 o_lt_d = np.cumsum(o_d, axis=0)
#                 o_lt_d[1:] = o_lt_d[:-1]
#                 o_lt_d[0, :] = 0
#                 masks_o_lt_d.set_value(o_lt_d)

#                 for i in range(nb_batches):
#                     print("Batch {0}/{1}".format(i, nb_batches))
#                     ln_dth_conditionals = []
#                     start = time()
#                     for d in range(D):
#                         if d % 100 == 0:
#                             print("{0}/{1} dth conditional ({2:.2f} sec.)".format(d, D, time()-start))
#                             start = time()

#                         ln_dth_conditionals.append(compute_lnp_x_o_d_given_x_o_lt_d(i, d))

#                     # We average p(x) on different orderings
#                     #nlls[i*batch_size:(i+1)*batch_size] += -np.sum(np.vstack(ln_dth_conditionals).T, axis=1)
#                     nlls[i*batch_size:(i+1)*batch_size] = np.logaddexp(nlls[i*batch_size:(i+1)*batch_size], -np.sum(np.vstack(ln_dth_conditionals).T, axis=1))

#             nlls -= np.log(len(orderings))  # Average across all orderings
#             return round(nlls.mean(), 6), round(nlls.std() / np.sqrt(nlls.shape[0]), 6)

#         super(EvaluateDeepNadeNLL, self).__init__(_nll_mean_and_std)

#     @property
#     def mean(self):
#         return ItemGetter(self, attribute=0)

#     @property
#     def std(self):
#         return ItemGetter(self, attribute=1)

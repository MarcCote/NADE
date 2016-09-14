import os
import numpy as np
import theano

from smartlearner.interfaces import Dataset

DATASETS_ENV = "DATASETS"


class ReconstructionDataset(Dataset):
    """ ReconstructionDataset interface.

    Behaves like a normal `Dataset` object but the targets are the inputs.

    Attributes
    ----------
    symb_inputs : `theano.tensor.TensorType` object
        Symbolic variables representing the inputs.

    Notes
    -----
    `symb_inputs` and `symb_targets` have test value already tagged to them. Use
    THEANO_FLAGS="compute_test_value=warn" to use them.
    """
    def __init__(self, inputs, name="dataset", keep_on_cpu=False):
        """
        Parameters
        ----------
        inputs : ndarray
            Training examples
        name : str (optional)
            The name of the dataset is used to name Theano variables. Default: 'dataset'.
        """
        super().__init__(inputs, name=name, keep_on_cpu=keep_on_cpu)
        self._targets_shared = self._inputs_shared
        self.symb_targets = self.symb_inputs.copy()


def load_binarized_mnist(keep_on_cpu=False):
    #Temporary patch until we build the dataset manager
    dataset_name = "binarized_mnist"

    datasets_repo = os.environ.get(DATASETS_ENV, './datasets')
    if not os.path.isdir(datasets_repo):
        os.mkdir(datasets_repo)

    repo = os.path.join(datasets_repo, dataset_name)
    dataset_npy = os.path.join(repo, 'data.npz')

    if not os.path.isfile(dataset_npy):
        if not os.path.isdir(repo):
            os.mkdir(repo)

            import urllib.request
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_train.txt', os.path.join(repo, 'mnist_train.txt'))
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_valid.txt', os.path.join(repo, 'mnist_valid.txt'))
            urllib.request.urlretrieve('http://www.cs.toronto.edu/~larocheh/public/datasets/mnist/mnist_test.txt', os.path.join(repo, 'mnist_test.txt'))

        train_file, valid_file, test_file = [os.path.join(repo, 'mnist_' + ds + '.txt') for ds in ['train', 'valid', 'test']]
        rng = np.random.RandomState(42)

        def parse_file(filename):
            data = np.array([np.fromstring(l, dtype=np.float32, sep=" ") for l in open(filename)])
            data = data[:, :-1]  # Remove target
            data = (data > rng.rand(*data.shape)).astype('int8')
            return data

        trainset, validset, testset = parse_file(train_file), parse_file(valid_file), parse_file(test_file)
        np.savez(dataset_npy,
                 trainset_inputs=trainset,
                 validset_inputs=validset,
                 testset_inputs=testset)

    data = np.load(dataset_npy)

    trainset = ReconstructionDataset(data['trainset_inputs'].astype(theano.config.floatX), name="trainset", keep_on_cpu=keep_on_cpu)
    validset = ReconstructionDataset(data['validset_inputs'].astype(theano.config.floatX), name="validset", keep_on_cpu=keep_on_cpu)
    testset = ReconstructionDataset(data['testset_inputs'].astype(theano.config.floatX), name="testset", keep_on_cpu=keep_on_cpu)

    return trainset, validset, testset


def load(dataset_name, keep_on_cpu=False):
    if dataset_name.lower() == "binarized_mnist":
        return load_binarized_mnist(keep_on_cpu=keep_on_cpu)
    # elif dataset_name.lower() == "caltech101_silhouettes28":
    #     return load_caltech101_silhouettes28()
    else:
        raise ValueError("Unknown dataset: {0}!".format(dataset_name))

import os
import numpy as np
import theano

from smartlearner.interfaces import Dataset

DATASETS_ENV = "DATASETS"


def load_binarized_mnist():
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
    trainset = Dataset(data['trainset_inputs'].astype(theano.config.floatX), data['trainset_inputs'].astype(theano.config.floatX), name="trainset")
    validset = Dataset(data['validset_inputs'].astype(theano.config.floatX), data['validset_inputs'].astype(theano.config.floatX), name="validset")
    testset = Dataset(data['testset_inputs'].astype(theano.config.floatX), data['testset_inputs'].astype(theano.config.floatX), name="testset")

    return trainset, validset, testset


def load(dataset_name):
    if dataset_name.lower() == "binarized_mnist":
        return load_binarized_mnist()
    # elif dataset_name.lower() == "caltech101_silhouettes28":
    #     return load_caltech101_silhouettes28()
    else:
        raise ValueError("Unknown dataset: {0}!".format(dataset_name))

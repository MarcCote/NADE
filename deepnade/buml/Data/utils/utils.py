import numpy as np
from Utils.Estimation import Estimation


def get_dataset_statistics(dataset, element=0):
    def reduction(sums_sqsums_n, *data):
        sums, sqsums, n = sums_sqsums_n
        sums += data[element].sum(0)
        sqsums += np.square(data[element]).sum(0)
        n += data[element].shape[0]
        return sums, sqsums, n
    sums, sqsums, n = dataset.reduce(reduction, (np.zeros((dataset.get_dimensionality(element),), dtype=dataset.get_type(element)), 
                                                 np.zeros((dataset.get_dimensionality(element),), dtype=dataset.get_type(element)), 
                                                 0))
    means = sums/n
    stds = np.sqrt(sqsums / n - np.square(means))
    return means, stds


def feedforward_dataset(dataset, layers):
    def feedforward(x):
        for i, l in enumerate(layers):
            x = l.feedforward(x.T).T.copy()
        return x,
    return dataset.map(feedforward)


def normalise_dataset(dataset, mean, std, element=0):
    def normalise_file(*data):
        data = list(data)
        data[element] = (data[element]-mean)/std
        return tuple(data)
    return dataset.map(normalise_file)


#TODO: reimplement as a reduction
def get_domains(dataset):
    mins = None
    maxs = None
    for x in dataset.file_iterator(path = True):
        min = x[0].min(0)
        max = x[0].max(0)
        if mins is None:
            mins = min
            maxs = max
        else:
            mins = np.minimum(mins, min)
            maxs = np.maximum(maxs, max)
    return zip(mins,maxs)


def estimate_loss_for_dataset(dataset, loss_f, minibatch_size=1000):
    loss_sum = 0.0
    loss_sq_sum = 0.0
    n = 0
    iterator = dataset.iterator(batch_size=minibatch_size, get_smaller_final_batch=True, shuffle=False)
    for x in iterator:
        if not isinstance(x, tuple):
            x = [x]
        n += x[0].shape[0]
        x = [e.T for e in x]
        loss = loss_f(*x)
        loss_sum += loss.sum()
        loss_sq_sum += (loss ** 2).sum()
    return Estimation.sample_mean_from_sum_and_sum_sq(loss_sum, loss_sq_sum, n)
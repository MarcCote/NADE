from __future__ import division
import numpy as np
import theano


class TheanoDatasetIteratorAdapter(object):
    def __init__(self, adapted_iterator, n_batches, max_mem=8):
        self.adapted_iterator = adapted_iterator
        self.batch_size = adapted_iterator.get_batch_size()
        self.max_mem_bytes = max_mem * 1024 * 1024
        self.n_batches = n_batches
        self.arity = self.adapted_iterator.get_arity()
        adapted_batch_size = self.max_mem_bytes // adapted_iterator.get_datapoint_sizes().sum()  # Number o datapoints that can be held in max_mem
        adapted_batch_size = (adapted_batch_size // self.batch_size) * self.batch_size  # Make it hold an exact integer multiple of self.batch_size
        if self.n_batches is None:
            adapted_iterator.n_batches = None
        else:
            adapted_iterator.n_batches = np.inf  # The adapted iterator must loop forever, limiting the number of batches is now done here.
        adapted_iterator.set_batch_size(adapted_batch_size, True)  # True means get smaller final minibatch
        # Create buffers for each of the elements
        self.buffers = []
        self.minibatch = []
        for i, dimensionality in enumerate(self.adapted_iterator.get_datapoint_dimensionalities()):
            self.buffers.append(theano.shared(np.zeros((adapted_batch_size, dimensionality),
                                                       dtype=self.adapted_iterator.dataset.get_type(i)),
                                             name="buffer_%d" % i))  # The big buffer that holds many batches
            self.minibatch.append(theano.shared(value=np.zeros((self.batch_size, dimensionality),
                                                               dtype=self.adapted_iterator.dataset.get_type(i)),
                                                name="minibatch_%d" % i))
        self.buffer_index = 0
        self.datapoints_in_buffer = 0

    def __iter__(self):
        self.adapted_iterator.__iter__()
        self.batches_returned = 0
        return self

    def fill_buffer(self):
        tmp = self.adapted_iterator.next()
        if self.arity == 1:
            self.buffers[0].set_value(tmp)
            self.datapoints_in_buffer = tmp.shape[0]
        else:
            for i in xrange(self.arity):
                self.buffers[i].set_value(tmp[i])
            self.datapoints_in_buffer = tmp[0].shape[0]
        self.buffer_index = 0

    def next(self):
        if self.batches_returned == self.n_batches:
            raise StopIteration
        self.batches_returned += 1
        # If the GPU buffer is empty, request from the adapted dataset and try again
        #if self.buffer_index + self.batch_size > self.datapoints_in_buffer:
        while self.buffer_index + self.batch_size > self.datapoints_in_buffer:
            self.fill_buffer()
        # Return a batch and increment the buffer index
        for i in xrange(self.arity):
            self.minibatch[i].set_value(self.buffers[i].get_value(borrow=True, return_internal_type=True)[self.buffer_index:self.buffer_index + self.batch_size], borrow=True)
        self.buffer_index += self.batch_size
        return self.minibatch

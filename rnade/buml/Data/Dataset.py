from __future__ import division
import Data
import h5py
import re
import numpy as np
import random
import ctypes
import tempfile
import os
import copy
from threading import Thread
from Queue import Queue, Empty

class Dataset(object):
    def __init__(self, data, block_length=1, use_blocks=None, offsets=None):
        """
        data can be a numpy array (in C order), a tuple of such arrays, or a list of such tuples or arrays
        """
        self.files = list()
        if isinstance(data, list): #Several files
            for file in data:
                if isinstance(file, tuple):
                    for d in file:
                        assert(isinstance(d, np.ndarray) and not np.isfortran(d))
                    self.files.append(file)
        elif isinstance(data, tuple): #Just one file
            for d in data:
                assert(isinstance(d, np.ndarray) and d.ndim == 2 and not np.isfortran(d))
            self.files.append(data)
        elif isinstance(data, np.ndarray): #One file with one kind of element only (not input-output)
            assert(isinstance(data, np.ndarray) and not np.isfortran(data))
            self.files.append(tuple([data]))
        # Support for block datapoints
        self.block_length = block_length
        if block_length == 1:
            self.block_lengths = [np.int(1)] * self.get_arity()
            self.offsets = [np.int(0)] * self.get_arity()
        elif block_length > 1:
            self.block_lengths = [np.int(block_length) if ub else np.int(1) for ub in use_blocks]  # np.asarray(dtype=np.int) and [np.int(x)] have elements with diff type. Careful!
            self.offsets = [np.int(off) for off in offsets]
            for ub, off in zip(use_blocks, offsets):
                if off != 0 and ub:
                    raise Exception("Can't have both a block size greater than 1 and an offset.")
        else:
            raise Exception("Block size must be positive")

    def get_n_files(self):
        return len(self.files)

    def get_arity(self):
        return len(self.files[0])

    def get_dimensionality(self, element=0):
        return self.get_file(element, 0).shape[1] * self.block_lengths[element]

    def get_type(self, element=0):
        return self.get_file(element, 0).dtype

    def get_file(self, element, index):
        return self.files[index][element]

    def get_file_shape(self, element, index):
        return self.get_file(element, index).shape

    def get_n_blocks_in_file(self, index):
        return np.maximum(self.get_file_shape(0, index)[0] - self.block_length + 1, 0)

    def iterator(self, batch_size=1, get_smaller_final_batch=False, shuffle=False, just_input=False, n_batches=None):
        """
        Returns an minibatch iterator
        get_smaller_final_batch: If True, it will return a final minibatch of size different to the specified. E.g. if the dataset has 1037 datapoints and the minibatch size is 100, it would return 10 batches of size 100 and one of size 37.
        shuffle: if True datapoints are shuffled
        just_input:  deprecated, useless now.
        n_batches: (optional) Number of minibatches to iterate through. If None then it will iterate through the whole dataset.
        """
        return DatasetIterator(self, batch_size, get_smaller_final_batch, shuffle, n_batches=n_batches)

    def describe(self):
        # Create a better description
        return "Dataset %d files" % (self.get_n_files())

    def file_iterator(self, **kwargs):
        return DatasetFileIterator(self, **kwargs)

    def sample_data(self, n=None, proportion=None):
        """
        With replacement
        """
        if n is None:
            assert(proportion is not None)
            total = sum([self.get_file_shape(0, i)[0] for i in xrange(len(self.file_paths))])
            n = total * proportion
        data = tuple(np.empty((n, self.get_dimensionality(i)), dtype=self.get_type(i)) for i in xrange(self.get_arity()))
        row = 0
        while row < n:
            file_index = np.random.randint(0, self.get_n_files())
            index = np.random.randint(0, self.get_file_shape(0, file_index)[0] - (self.block_length - 1))
            for i in xrange(self.get_arity()):
                data[i][row] = self.get_file(i, file_index)[index:index + self.block_lengths[i], :].flatten()
            row += 1
        return data

    def map(self, mapping_f):
        new_data = list()
        # Transform file by file
        for x in self.file_iterator():
            new_data.append(mapping_f(*x))
        return Dataset(new_data)

    def reduce(self, reduction_f, initial_accum):
        # Reduce file by file
        accum = initial_accum
        for x in self.file_iterator():
            accum = reduction_f(accum, *x)
        return accum


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, get_smaller_final_batch, shuffle, n_batches=None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_batches = n_batches
        # # Store an order for the files
        self.files_order = range(self.dataset.get_n_files())
        self.arity = self.dataset.get_arity()
        # # When iterating over a dataset with arity greater than one, a tuple or arrays is returned, otherwise just the array
        if self.arity == 1:
            self.return_type = lambda x: x[0]
        else:
            self.return_type = tuple
        self.element_dimensionalities = [self.dataset.get_dimensionality(i) for i in xrange(self.arity)]
        self.element_sizes = [np.int(self.dataset.get_file(i, 0).strides[0]) for i in xrange(self.arity)]
        self.element_types = [self.dataset.get_type(i) for i in xrange(self.arity)]
        self.element_block_lengths = self.dataset.block_lengths
        self.element_offsets = self.dataset.offsets
        self.block_length = self.dataset.block_length
        self.set_batch_size(batch_size, get_smaller_final_batch)
        # srcs
        self.n_srcs = np.sum([np.int(self.dataset.get_n_blocks_in_file(i)) for i in xrange(self.dataset.get_n_files())])
        self.srcs = [np.zeros(self.n_srcs, dtype="int") for i in xrange(self.dataset.get_arity())]
        src_count = 0
        for f in xrange(self.dataset.get_n_files()):
            for i in xrange(self.dataset.get_n_blocks_in_file(f)):
                for e in xrange(self.dataset.get_arity()):
                    self.srcs[e][src_count] = self.dataset.files[f][e].ctypes.data + self.element_sizes[e] * (self.element_offsets[e] +  i)
                src_count += 1
        self.restart()

    def set_batch_size(self, n, get_smaller_final_batch):
        self.batch_size = np.int(n)
        self.get_smaller_final_batch = get_smaller_final_batch
        self.batches = [np.empty((self.batch_size, self.element_dimensionalities[i]), dtype=self.element_types[i]) for i in xrange(self.arity)]

    def get_arity(self):
        return self.arity

    def get_batch_size(self):
        return self.batch_size

    def get_datapoint_sizes(self):
        return np.array(self.element_sizes)

    def get_datapoint_dimensionalities(self):
        return self.element_dimensionalities

    def __iter__(self):
        self.batches_returned = 0
        self.finished = False
        self.restart()
        return self

    def restart(self):
        # Shuffle if appropiate
        if self.shuffle:
            self.srcs_order = np.random.permutation(self.n_srcs)
        else:
            self.srcs_order = np.arange(self.n_srcs, dtype="int")
        self.srcs_index = np.int(0)

    def next(self):
        if self.batches_returned == self.n_batches or self.finished:
            raise StopIteration
        elif self.srcs_index + self.batch_size < self.n_srcs:
            for j in xrange(self.arity):
                dst = np.int(self.batches[j].ctypes.data)
                size = self.element_sizes[j] * self.element_block_lengths[j]
                for i in xrange(self.batch_size):
                    n = self.srcs_order[self.srcs_index + i]
                    ctypes.memmove(dst, np.int(self.srcs[j][n]), size)
                    dst += size
            self.srcs_index += self.batch_size
            self.batches_returned += 1
            return self.return_type(self.batches)
        elif self.n_batches is None:
            srcs_left = self.n_srcs - self.srcs_index
            if srcs_left ==0 or not self.get_smaller_final_batch:
                raise StopIteration
            else:
                for j in xrange(self.arity):
                    dst = np.int(self.batches[j].ctypes.data)
                    size = self.element_sizes[j] * self.element_block_lengths[j]
                    for i in xrange(srcs_left):
                        n = self.srcs_order[self.srcs_index + i]
                        ctypes.memmove(dst, np.int(self.srcs[j][n]), size)
                        dst += size
                self.srcs_index += srcs_left
                self.batches_returned += 1
                self.finished = True
                return self.return_type([b[0:srcs_left, :] for b in self.batches])
        else:
            self.restart()
            if self.batch_size >= self.n_srcs:
                if not self.get_smaller_final_batch:
                    raise "Dataset doesn't even contain one minibatch"
                else:
                    srcs_left = self.n_srcs - self.srcs_index
                    for j in xrange(self.arity):
                        dst = np.int(self.batches[j].ctypes.data)
                        size = self.element_sizes[j] * self.element_block_lengths[j]
                        for i in xrange(srcs_left):
                            n = self.srcs_order[self.srcs_index + i]
                            ctypes.memmove(dst, np.int(self.srcs[j][n]), size)
                            dst += size
                    self.srcs_index += srcs_left
                    self.batches_returned += 1
                    return self.return_type([b[0:srcs_left, :] for b in self.batches])
            else:
                for j in xrange(self.arity):
                    dst = np.int(self.batches[j].ctypes.data)
                    size = self.element_sizes[j] * self.element_block_lengths[j]
                    for i in xrange(self.batch_size):
                        n = self.srcs_order[self.srcs_index + i]
                        ctypes.memmove(dst, np.int(self.srcs[j][n]), size)
                        dst += size
                self.srcs_index += self.batch_size
                self.batches_returned += 1
            return self.return_type(self.batches)


class DatasetFileIterator(object):
    # TODO: it should be possible to obtain the files in a randomized order.
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

    def __iter__(self):
        self.f_index = 0
        return self

    def next(self):
        if self.f_index >= self.dataset.get_n_files():
            raise StopIteration
        else:
            self.f_index += 1
            x = [self.dataset.get_file(e, self.f_index - 1) for e in xrange(self.dataset.get_arity())]
            return tuple(x)

def load_dataset_from_hdf5(filename, entries_regexp, element_names, block_length=1, use_blocks=None, offsets=None):
        """
        filename: hdf5 filename
        entries_regexp: Regular expresion for selecting files in the hdf5 file (e.g. /training/.*/.*/.*)
        element_names: Name of the hdf5 tensors with the actual data for each of the files. (e.g. tuple("acoustics", "phone_state") ).
        delete_after_use: If set to True the hdf5 file will be deleted from the hard drive when the object is destroyed. It is used to create temporary hdf5 files as a results of "mapping" a dataset.
        """
        f = h5py.File(filename, "r")
        element_names = element_names if isinstance(element_names, tuple) else (element_names,)
        # # Find the entries that satisfy the regexp
        # split the pattern by "/"
        pats = entries_regexp.split("/")
        try:
            pats.remove("")
        except:
            pass
        # traverse from the root
        entries = [f["/"]]
        for p in pats:
            entries = [v for r in entries for k, v in r.items() if re.match("^%s$" % p, str(k))]
        data = [tuple([np.array(e[dsn].value) for dsn in element_names]) for e in entries]

        return Dataset(data, block_length, use_blocks, offsets)
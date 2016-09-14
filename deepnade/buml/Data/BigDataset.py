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


def speech_dataset(filename, entries_regexp, element_names, n_frames):
    if not isinstance(element_names, tuple) or len(element_names) == 1:
        return BigDataset(filename, entries_regexp, element_names, block_length=n_frames, use_blocks=(True,), offsets=(0,))
    if len(element_names) == 2:
        return BigDataset(filename, entries_regexp, element_names, block_length=n_frames, use_blocks=(True, False), offsets=(0, n_frames // 2))
    else:
        raise Exception("Not implemented")


class BigDataset(object):
    def __init__(self, filename, entries_regexp, element_names, delete_after_use=False, block_length=1, use_blocks=None, offsets=None):
        """
        filename: hdf5 filename
        entries_regexp: Regular expresion for selecting files in the hdf5 file (e.g. /training/.*/.*/.*)
        element_names: Name of the hdf5 tensors with the actual data for each of the files. (e.g. tuple("acoustics", "phone_state") ).
        delete_after_use: If set to True the hdf5 file will be deleted from the hard drive when the object is destroyed. It is used to create temporary hdf5 files as a results of "mapping" a dataset.
        """
        self.filename = filename
        self.delete_after_use = (delete_after_use == True)
        self.f = h5py.File(filename, "r")
        self.element_names = element_names if isinstance(element_names, tuple) else (element_names,)
        self.entries_regexp = entries_regexp
        # # Find the entries that satisfy the regexp
        # split the pattern by "/"
        pats = entries_regexp.split("/")
        try:
            pats.remove("")
        except:
            pass
        # traverse from the root
        entries = [self.f["/"]]
        for p in pats:
            entries = [v for r in entries for k, v in r.items() if re.match("^%s$" % p, str(k))]
        self.file_paths = [str(e.name) for e in entries]
        self.files = [[e[dsn] for e in entries] for dsn in self.element_names]
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

    def __del__(self):
        self.f.close()
        if self.delete_after_use:
            os.remove(self.filename)

    def close(self):
        self.f.close()

    def get_n_files(self):
        return len(self.files[0])

    def get_arity(self):
        return len(self.element_names)

    def get_dimensionality(self, element=0):
        return self.get_file(element, 0).shape[1] * self.block_lengths[element]

    def get_type(self, element=0):
        return self.get_file(element, 0).dtype

    def get_element_names(self):
        return self.element_names

    def get_file(self, element, index):
        return np.array(self.files[element][index].value)  # , order='C')
        # return np.array(self.files[element][index].value, dtype=np.float32, order='C')

    def get_file_shape(self, element, index):
        return self.files[element][index].shape

    def get_n_blocks_in_file(self, index):
        return np.maximum(self.get_file_shape(0, index)[0] - self.block_length + 1, 0)

    def get_file_path(self, index):
        return self.file_paths[index]

    def iterator(self, batch_size=1, get_smaller_final_batch=False, shuffle=False, just_input=False, n_batches=None):
        """
        Returns an minibatch iterator
        get_smaller_final_batch: If True, it will return a final minibatch of size different to the specified. E.g. if the dataset has 1037 datapoints and the minibatch size is 100, it would return 10 batches of size 100 and one of size 37.
        shuffle: if True datapoints are shuffled
        just_input:  deprecated, useless now.
        n_batches: (optional) Number of minibatches to iterate through. If None then it will iterate through the whole dataset.
        """
        return BigDatasetIterator(self, batch_size, get_smaller_final_batch, shuffle, n_batches=n_batches)

    def describe(self):
        return "%s:%s:(%s)" % (self.filename, self.entries_regexp, self.element_names)

    def file_iterator(self, **kwargs):
        return BigDatasetFileIterator(self, **kwargs)

    def save(self, filename):
        """
        """
        f = h5py.File(filename, "w")
        arity = self.get_arity()
        # Iterate through dataset one point at a time
        counter = 0
        for batch in self.iterator(batch_size=1000, get_smaller_final_batch=True, shuffle=False):
            # Make it always a tuple
            if not isinstance(batch, tuple):
                batch = (batch,)
            for row in xrange(batch[0].shape[0]):
                tdatapoint = [e[row] for e in batch]
                if counter == 0:
                    # Initialize dataset
                    dimensionalities = [tdatapoint[i].shape[0] for i in xrange(len(tdatapoint))]
                    out_dataset = [f.create_dataset('/data/%d' % (i), (1000, dimensionalities[i]), np.float32 , maxshape=(None, dimensionalities[i])) for i in xrange(len(tdatapoint))]
                    maxrows = 1000
                for i, x in enumerate(tdatapoint):
                    out_dataset[i][counter] = tdatapoint[i]
                counter += 1
                if counter == maxrows:
                    maxrows += 1000
                    for i, fds in enumerate(out_dataset):
                        fds.resize((maxrows, dimensionalities[i]))
        # Final resize to exact size
        for i, fds in enumerate(out_dataset):
            fds.resize((counter, dimensionalities[i]))
        f.close()
        return self

    def get_data(self, n=None, proportion=None, accept_less=True):
        if n is None:
            total = sum([self.get_file_shape(0, i)[0] for i in xrange(len(self.file_paths))])
            if proportion is not None:
                n = total * proportion
            else:
                n = total
        data = tuple(np.empty((n, self.get_dimensionality(i))) for i in xrange(self.get_arity()))
        row = 0
        for fs in self.file_iterator():
            for i, f in enumerate(fs):
                increment = min(f.shape[0], n - row)
                data[i][row:row + increment, :] = f[0:increment, :]
            row += increment
            if row >= n:
                break
        if accept_less and row < n:
            return tuple(d[0:row, :] for d in data)
        else:
            assert(n == row)
        return data

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

    def map(self, mapping_f, filename=None):
        if filename is None:
            fname = tempfile.mktemp()
            delete = True
        else:
            fname = filename
            delete = False
        f = h5py.File(fname, "w")
        # Transform file by file
        for x in self.file_iterator(path=True):
            route = x[-1]
            x = x[:-1]
            x = [Data.expand_array_in_blocks(element, self.block_length, self.offsets[i]) for i,element in enumerate(x)]
            y = mapping_f(*x)
            assert(isinstance(y, tuple))
            if np.prod(y[0].shape) > 0:
                for i, v in enumerate(y):
                    f[route + "/" + str(i)] = v
        f.close()
        return BigDataset(fname,
                          self.entries_regexp,
                          tuple(str(i) for i in xrange(len(y))),
                          delete_after_use=delete)

    def reduce(self, reduction_f, initial_accum):
        # Reduce file by file
        accum = initial_accum
        for x in self.file_iterator():
            blocks = [Data.expand_array_in_blocks(element, self.block_length, self.offsets[i]) for i,element in enumerate(x)]
            accum = reduction_f(accum, *blocks)
        return accum

#    def pmap(self, mapping_f, filename = None, n_workers = 4):
#        """
#        Theano is not thread-safe, do not use this!
#        """
#        if filename is None:
#            fname = tempfile.mktemp()
#            delete = True
#        else:
#            fname = filename
#            delete = False
#        f = h5py.File(fname, "w")
#        queue = Queue(maxsize = n_workers * 2)
#        worker_threads = [MappingThread(queue, mapping_f, f) for i in range(n_workers)]
#        # Start all threads
#        [t.start() for t in worker_threads]
#        # Transform file by file. This is done by queuing it and waiting for the worker threads to do it.
#        for x in self.file_iterator(path = True):
#            queue.put(x)
#        # Send a signal to all threads so they finish
#        [t.finish() for t in worker_threads]
#        # Wait for all the processing and writing to be finished before closing the file
#        queue.join()
#        f.close()
#        # Return the new dataset
#        return Dataset(fname,
#                       self.entries_regexp,
#                       tuple(str(i) for i in xrange(len(y))),
#                       delete_after_use = delete)
#
#
# class MappingThread(Thread):
#    """
#    """
#    #h5py is thread safe :)
#    def __init__(self, queue, mapping_f, dest_file):
#        Thread.__init__(self)
#        self.setDaemon(True)
#        self.queue = queue
#        self.dest_file = dest_file
#        self.mapping_f = mapping_f
#        self.finished = False
#
#    def finish(self):
#        self.finished = True
#
#    def run(self):
#        while True:
#            try:
#                x = self.queue.get(True, 1)
#                route = x[-1]
#                x = x[:-1]
#                y = self.mapping_f(*x)
#                if np.prod(y[0].shape) > 0:
#                    for i,v in enumerate(y):
#                        self.dest_file[route+"/"+str(i)] = v
#                self.queue.task_done()
#            except Empty:
#                #If finished break the loop
#                if self.finished:
#                    break


class BigDatasetIterator(object):
    def __init__(self, dataset, batch_size, get_smaller_final_batch, shuffle, n_batches=None, max_buffer_size=8):
        self.max_buffer_size = max_buffer_size * 1024 * 1024
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
        self.max_srcs = np.int(self.max_buffer_size // np.sum(np.multiply(self.element_sizes, self.element_block_lengths)))
        self.srcs = np.zeros((self.max_srcs, self.arity), dtype="int")
        self.srcs_count = np.int(0)
        self.srcs_index = np.int(0)
        self.files = list()
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
        if self.n_batches is None:
            self.restart()
            self.fill_files_buffer()
        return self

    def restart(self):
        # Shuffle if appropiate
        if self.shuffle:
            random.shuffle(self.files_order)
        self.file_index = 0

    def next(self):
        if self.batches_returned == self.n_batches or self.finished:
            raise StopIteration
        if self.srcs_index + self.batch_size < self.srcs_count:
            for j in xrange(self.arity):
                dst = np.int(self.batches[j].ctypes.data)
                size = self.element_sizes[j] * self.element_block_lengths[j]
                for i in xrange(self.batch_size):
                    n = self.srcs_order[self.srcs_index + i]
                    ctypes.memmove(dst, np.int(self.srcs[n][j]), size)
                    dst += size
            self.srcs_index += self.batch_size
        else:
            dsts = [np.int(b.ctypes.data) for b in self.batches]
            for i in xrange(self.batch_size):
                try:
                    if self.srcs_index >= self.srcs_count:
                        self.fill_files_buffer()
                    for j in xrange(self.arity):
                        n = self.srcs_order[self.srcs_index]
                        size = self.element_sizes[j] * self.element_block_lengths[j]
                        ctypes.memmove(dsts[j], np.int(self.srcs[n][j]), size)
                        dsts[j] += size
                    self.srcs_index += 1
                except StopIteration:
                    # We have filled exactly i
                    if i == 0 or not self.get_smaller_final_batch:
                        raise StopIteration
                    else:
                        self.batches_returned += 1
                        self.finished = True
                        return self.return_type([b[0:i, :] for b in self.batches])
        self.batches_returned += 1
        return self.return_type(self.batches)

    def fill_files_buffer(self):
        self.files = [[]] * self.arity
        # Fill the buffer
        if self.file_index >= len(self.files_order):
            if self.n_batches is None:
                raise StopIteration
            else:
                self.restart()
        # Fill the buffer with a number of files
        self.srcs_count = 0
        # While there's data to load and the buffer is not full
        while self.file_index < len(self.files_order) and self.srcs_count < self.max_srcs:
            # Check blocks of data available in the next file
            # f = self.dataset.get_file(0, self.files_order[self.file_index])
            f_index = self.files_order[self.file_index]
            blocks_in_file = self.dataset.get_n_blocks_in_file(f_index) #f.shape[0] - (self.block_length - 1)
            if blocks_in_file <= self.max_srcs - self.srcs_count:
                srcs_to_copy = blocks_in_file
                src_offset = 0
            else:
                srcs_to_copy = self.max_srcs - self.srcs_count
                src_offset = np.random.randint(blocks_in_file - (self.max_srcs - self.srcs_count) + 1)
            if srcs_to_copy >= 1:
                for i in xrange(self.arity):
                    f = self.dataset.get_file(i, f_index)
                    self.files[i].append(f)
                    for j in xrange(srcs_to_copy):
                        self.srcs[self.srcs_count + j, i] = f.ctypes.data + self.element_sizes[i] * (self.element_offsets[i] + src_offset + j)
                self.srcs_count += srcs_to_copy
            self.file_index += 1
        if self.shuffle:
            self.srcs_order = np.random.permutation(self.srcs_count)
        else:
            self.srcs_order = np.arange(self.srcs_count, dtype="int")
        self.srcs_index = 0


class BigDatasetFileIterator(object):
    # TODO: it should be possible to obtain the files in a randomized order.
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.return_path = kwargs.get('path', False)

    def __iter__(self):
        self.f_index = 0
        return self

    def next(self):
        if self.f_index >= self.dataset.get_n_files():
            raise StopIteration
        else:
            self.f_index += 1
            x = [self.dataset.get_file(e, self.f_index - 1) for e in xrange(self.dataset.get_arity())]
            if self.return_path:
                x.append(self.dataset.get_file_path(self.f_index - 1))
            return tuple(x)

from __future__ import division
import Data
import h5py
import re
import numpy as np
import random
import ctypes
import tempfile
import os
from Dataset import Dataset
#from Dataset import MappingThread
#from threading import Thread
#from Queue import Queue

class SpeechDataset(object):
    """
    Reads a speech dataset from an HDF5 file (speech specific). This class has special support 
    for returning frames in groups (framespan) and phone-state labels in a one-of-k representation.
    """
    def __init__(self, filename, entries_regexp, input_dataset, output_dataset=None, delete_after_use=False):
        """
        filename: hdf5 filename
        entries_regexp: Regular expresion for selecting files in the hdf5 file (e.g. /training/.*/.*/.*)
        input_dataset: Name of the hdf5 tensors (datasets in hdf5 parlance) with the "input" element for each of the files. (e.g. "acoustics" ).
        output_dataset: Name of the hdf5 tensors (datasets in hdf5 parlance) with the "output" element for each of the files. (e.g. "phone-state" ).
        delete_after_use: If set to True the hdf5 file will be deleted from the hard drive when the object is destroyed. It is used to create temporary hdf5 files as a results of "mapping" a dataset.
        """
        self.delete_after_use = delete_after_use
        self.filename = filename
        self.entries_regexp = entries_regexp
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.f = h5py.File(filename, "r")
        ## Find the entries that satisfy the regexp
        #split the pattern by "/"
        pats = entries_regexp.split("/")
        try:
            pats.remove("")
        except: pass
        #traverse from the root
        entries = [self.f["/"]]
        for p in pats:
            new_roots = []
            for r in entries:
                for k,v in r.items():
                    if re.match("^%s$" % p, str(k)):
                        new_roots.append(v)
            entries = new_roots
        ## Store the names of all files and their outputs (if present)        
        self.paths = [str(e.name) for e in entries]
        self.input_files = [e[input_dataset] for e in entries]
        if output_dataset is not None:
            try:
                entries[0][output_dataset].keys().index("times")
                self.has_times = True
            except ValueError:
                self.has_times = False            
            self.output_data_labels = [e[output_dataset + "/labels"] for e in entries]
            if self.has_times:
                self.output_data_times  = [e[output_dataset + "/times" ] for e in entries]
                self.frame_duration = entries[0][input_dataset].attrs["WINDOWSIZE"] / 1e7 ## HTK stores it in tenths of microseconds
                self.frame_time_shift = entries[0][input_dataset].attrs["TARGETRATE"] / 1e7
            else:
                self.output_data_times  = [e[output_dataset + "/frames" ] for e in entries]
        self.n_frames = 1
        self.frame_shift = 1
        
    def describe(self):
        return "%s:%s:(%s,%s)" % (self.filename, self.entries_regexp, self.input_dataset, self.output_dataset)

    def __del__(self):
        self.f.close()
        if self.delete_after_use:
            os.remove(self.filename)

    def get_n_files(self):
        return len(self.input_files)

    def has_output(self):
        try:
            return self.output_data_labels is not None
        except AttributeError:
            return False
    
    def get_input_dimensionality(self):
        return self.get_input_file(0).shape[1] * self.n_frames

    def get_output_dimensionality(self):
        return self.get_output_values_file(0).shape[1]
    
    def get_arity(self):
        return 2 if self.has_output() else 1

    def get_dimensionality(self, element=0):
        if element==0:
            return self.get_input_dimensionality()
        elif element==1:
            return self.get_output_dimensionality()
        else:
            raise "Invalid element of speech dataset, only 2 elements (acoustics and labels)"
    
    def get_file_path(self, index):
        return self.paths[index]
    
    def get_file_shape(self, element, index):
        assert(element == 0)
        return self.input_files[index].shape

    def get_input_file(self, index):
        return np.array(self.input_files[index].value, dtype=np.float32, order='C')

    def get_output_values_file(self, index):
        return np.array(self.output_data_labels[index].value, dtype=np.float32, order='C')

    def get_output_times_file(self, index):
        return self.output_data_times[index].value
    
    def set_n_frames(self, n_frames):
        """
        Set the framespan to n_frames. That is, the input element each datapoint will be a flattened vector of n frames.
        """
        self.n_frames = n_frames
        
    def set_frame_shift(self, frame_shift):
        self.frame_shift = frame_shift
            
    def iterator(self, batch_size = 1, get_smaller_final_batch = False, shuffle = False, just_input = False, max_batches = None):
        # TODO: cache, most of the time it is going to request one that is exactly the same. Can save on things like generating the labels, srcs...
        # Could be cached explicitely when used: more repetition, less magic... caching here would be bad if several identical iterators are being used (eg. multithreaded env)
        if not self.has_output() or just_input:
            return SpeechDatasetIterator(self, batch_size, get_smaller_final_batch, shuffle, self.n_frames, self.frame_shift, max_batches = max_batches)
        else:
            return LabeledSpeechDatasetIterator(self, batch_size, get_smaller_final_batch, shuffle, self.n_frames, self.frame_shift, max_batches = max_batches)

    def file_iterator(self, **kwargs):
        return SpeechDatasetFileIterator(self, **kwargs)
    
    def map(self, mapping_f, filename = None):
        if filename is None:
            fname = tempfile.mktemp()
            delete = True
        else:
            fname = filename
            delete = False            
        f = h5py.File(fname, "w")
        #Transform file by file        
        for x in self.file_iterator(path = True):
            route = x[-1]
            x = x[:-1]
            y = mapping_f(*x)
            if np.prod(y[0].shape) > 0:
                for i,v in enumerate(y):
                    f[route+"/"+str(i)] = v
        f.close()
        return Dataset(fname, 
                       self.entries_regexp, 
                       tuple(str(i) for i in xrange(len(y))), 
                       delete_after_use = delete)

    def reduce(self, reduction_f, initial_accum):
        #Reduce file by file 
        accum = initial_accum       
        for x in self.file_iterator():           
            accum = reduction_f(accum, *x)
        return accum
        
#    def pmap(self, mapping_f, filename = None, n_workers = 4):
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
    #TODO: Move to utils        
    def get_data(self, n = None, proportion = None, accept_less = True):        
        if n is None:
            total = sum([self.get_file_shape(0, i)[0] for i in xrange(len(self.file_paths))])
            if proportion is not None: 
                n = total * proportion
            else:
                n = total
        data = tuple(np.empty((n, self.get_dimensionality(i))) for i in xrange(self.get_arity()))
        row = 0
        for fs in self.file_iterator():
            for i,f in enumerate(fs):
                increment = min(f.shape[0], n-row)
                data[i][row:row+increment, :] = f[0:increment, :]
            row += increment
            if row >= n:
                break
        if accept_less and row < n:
            return tuple(d[0:row,:] for d in data)
        else:
            assert(n == row)
        return data
    
    def sample_data(self, n = None, proportion = None):
        """
        With replacement
        """        
        if n is None:
            assert(proportion is not None)
            total = sum([self.get_file_shape(0, i)[0] for i in xrange(len(self.file_paths))])
            n = total * proportion
        data = np.empty((n, self.get_dimensionality(0)))
        row = 0
        while row < n:
            file_index = np.random.randint(0, self.get_n_files())
            top = self.get_file_shape(0, file_index)[0] - self.n_frames
            if top <= 0:
                continue
            else:
                index = np.random.randint(0,top+1)                   
                data[row] = self.get_input_file(file_index)[index:index+self.n_frames, :].flatten()
                row += 1
        return data

class LabeledSpeechDatasetIterator(object):
    def __init__(self, dataset, batch_size, get_smaller_final_batch, shuffle, n_frames, frame_shift, max_batches = None,  max_buffer_size = 128):
        self.max_buffer_size = max_buffer_size * 1024 * 1024
        self.dataset = dataset
        self.shuffle = shuffle
        self.max_batches = max_batches
        self.n_frames = n_frames
        self.frame_shift = frame_shift
        self.set_batch_size(batch_size, get_smaller_final_batch)
        #Store and ordering for the files
        self.files_order = range(self.dataset.get_n_files())
        f = self.dataset.get_input_file(0)
        self.frame_dimensionality = f.shape[1]
        self.frame_size, input_element_size = f.strides
        self.block_size = int(self.frame_size * self.n_frames) #If frame_shift is 1, then we can copy the full n frames in one memcpy operation
        self.input_data_type = f.dtype
        f = self.dataset.get_output_values_file(0)
        self.output_dimensionality = f.shape[1]
        self.output_size, output_element_size = f.strides
        self.output_data_type = f.dtype

    def fill_files_buffer(self):
        self.input_files = []
        self.output_files = []
        self.srcs = []
        buffer_size = 0
        index = 0
        if self.file_index >= len(self.files_order):
            raise StopIteration
        #Fill the buffer
        while buffer_size < self.max_buffer_size and self.file_index < len(self.files_order):##TODO add size in bytes
            #Get the output and output files
            inp = self.dataset.get_input_file(self.files_order[self.file_index])
            outp = self.dataset.get_output_values_file(self.files_order[self.file_index])
            self.input_files.append(inp)
            self.output_files.append(outp)
            buffer_size += inp.shape[0] * inp.ctypes.strides[0]
            buffer_size += outp.shape[0] * outp.ctypes.strides[0]
            inp_src = inp.ctypes.data
            outp_src = outp.ctypes.data
            times  = self.dataset.get_output_times_file(self.files_order[self.file_index])
            if self.dataset.has_times:
                i = 0
                tb0 = 0.5 * (self.dataset.frame_duration + (self.n_frames-1) * self.dataset.frame_time_shift)
                time_step = self.dataset.frame_time_shift * self.frame_shift
                for j in xrange(Data.n_blocks(inp.shape[0], self.n_frames, self.frame_shift)):
                    # Time for the center of the block of frames
                    t = tb0 + time_step * j
                    # i is the index in the label file corresponding to that time
                    while t > times[i][1] and i < len(times)-1:
                        i+=1
                        outp_src += int(self.output_size)
                    self.srcs.append((inp_src, outp_src))
                    # Add an entry to the list of addresses
                    inp_src += int(self.frame_size * self.frame_shift)
            else: #Has frame indexes instead of times
                i = 0
#                for j in xrange(1+self.n_frames//2, Data.n_blocks(inp.shape[0], self.n_frames, self.frame_shift)):
#                    while j > times[i][1] and i < len(times)-1:
                for j in xrange(Data.n_blocks(inp.shape[0], self.n_frames, self.frame_shift)):
                    #while (self.n_frames//2 + j) > times[i][1] and i < len(times)-1:
                    while (self.n_frames//2 + j) >= times[i][1] and i < len(times)-1:
                        i+=1
                        outp_src += int(self.output_size)
                    self.srcs.append((inp_src, outp_src))
                    # Add an entry to the list of addresses
                    inp_src += int(self.frame_size * self.frame_shift)
            # Next file and next buffer index
            index += 1
            self.file_index += 1
        if self.shuffle:
            random.shuffle(self.srcs)
        self.srcs_index = 0
 
    def set_batch_size(self, n, get_smaller_final_batch):
        self.batch_size = n
        self.get_smaller_final_batch = get_smaller_final_batch

    def get_batch_size(self):
        return self.batch_size

    def get_datapoint_sizes(self):
        return np.array([self.block_size, self.output_size])

    def get_datapoint_size(self):
        return self.block_size

    def get_output_size(self):
        return self.output_size

    def get_datapoint_dimensionalities(self):
        return np.array([self.n_frames * self.frame_dimensionality,
                         self.output_dimensionality])        

    def get_datapoint_dimensionality(self):
        return self.n_frames * self.frame_dimensionality

    def get_output_dimensionality(self):
        return self.output_dimensionality
        
    def __iter__(self):
        self.input_batch  = np.empty((self.batch_size, self.n_frames * self.frame_dimensionality), dtype = self.input_data_type)
        self.output_batch = np.empty((self.batch_size, self.output_dimensionality), dtype = self.output_data_type)
        self.batches_returned = 0
        # Shuffle if appropriate
        if self.shuffle:
            random.shuffle(self.files_order)
        self.file_index = 0
        self.srcs_index = 0
        self.srcs = []
        return self

    def next(self):
        input_dst = self.input_batch.ctypes.data
        output_dst = self.output_batch.ctypes.data
        if self.batches_returned == self.max_batches:
            raise StopIteration
        self.batches_returned += 1
        for i in xrange(self.batch_size):
            try:
                if self.srcs_index >= len(self.srcs):
                    self.fill_files_buffer()
                inp_src, outp_src = self.srcs[self.srcs_index]
                self.srcs_index+=1
                if self.frame_shift == 1:
                    ctypes.memmove(input_dst, inp_src, self.block_size)
                    input_dst += int(self.block_size)
                else:
                    for j in xrange(self.n_frames):
                        ctypes.memmove(input_dst, inp_src, self.frame_size)
                        input_dst += int(self.frame_size)
                        input_src += int(self.frame_size * self.frame_shift)
                ctypes.memmove(output_dst, outp_src, self.output_size)
                output_dst += int(self.output_size)
            except StopIteration:
                #We have filled exactly i
                if i == 0 or not self.get_smaller_final_batch:
                    raise StopIteration
                else:
                    return (self.input_batch[0:i, :], self.output_batch[0:i, :])
        return (self.input_batch, self.output_batch)

class SpeechDatasetIterator(object):
    def __init__(self, dataset, batch_size, get_smaller_final_batch, shuffle, n_frames, frame_shift, max_batches = None, max_buffer_size = 128):
        self.max_buffer_size = max_buffer_size * 1024 * 1024
        self.max_batches = max_batches
        self.dataset = dataset
        self.shuffle = shuffle
        self.n_frames = n_frames
        self.frame_shift = frame_shift
        self.set_batch_size(batch_size, get_smaller_final_batch)
        #Store and ordering for the files
        self.files_order = range(self.dataset.get_n_files())
        f = self.dataset.get_input_file(0)
        self.frame_dimensionality = f.shape[1]
        self.frame_size, input_element_size = f.strides
        self.block_size = int(self.frame_size * self.n_frames) #If frame_shift is 1, then we can copy the full n frames in one memcpy operation
        self.input_data_type = f.dtype

    def fill_files_buffer(self):
        self.input_files = []
        self.srcs = []
        buffer_size = 0
        index = 0
        if self.file_index >= len(self.files_order):
            raise StopIteration
        #Fill the buffer
        while buffer_size < self.max_buffer_size and self.file_index < len(self.files_order):##TODO add size in bytes
            #Get the output and output files
            inp = self.dataset.get_input_file(self.files_order[self.file_index])
            self.input_files.append(inp)
            buffer_size += inp.shape[0] * inp.ctypes.strides[0]
            inp_src = inp.ctypes.data
            for j in xrange(Data.n_blocks(inp.shape[0], self.n_frames, self.frame_shift)):
                self.srcs.append(inp_src)
                # Add an entry to the list of addresses
                inp_src += int(self.frame_size * self.frame_shift)
            # Next file and next buffer index
            index += 1
            self.file_index += 1
        if self.shuffle:
            random.shuffle(self.srcs)
        self.srcs_index = 0
 
    def set_batch_size(self, n, get_smaller_final_batch):
        self.batch_size = n
        self.get_smaller_final_batch = get_smaller_final_batch

    def get_batch_size(self):
        return self.batch_size

    def get_datapoint_size(self):
        return self.block_size
    
    def get_datapoint_sizes(self):
        return np.array([self.block_size])

    def get_datapoint_dimensionality(self):
        return self.n_frames * self.frame_dimensionality

    def get_output_dimensionality(self):
        return self.output_dimensionality
    
    def get_datapoint_dimensionalities(self):
        return np.array([self.n_frames * self.frame_dimensionality])
        
    def __iter__(self):
        self.input_batch  = np.empty((self.batch_size, self.n_frames * self.frame_dimensionality), dtype = self.input_data_type)
        self.batches_returned = 0
        # Shuffle if appropiate
        if self.shuffle:
            random.shuffle(self.files_order)
        self.file_index = 0
        self.srcs_index = 0
        self.srcs = []
        return self

    def next(self):
        input_dst = self.input_batch.ctypes.data
        if self.batches_returned == self.max_batches:
            raise StopIteration
        self.batches_returned += 1
        for i in xrange(self.batch_size):
            try:
                if self.srcs_index >= len(self.srcs):
                    self.fill_files_buffer()
                inp_src = self.srcs[self.srcs_index]
                self.srcs_index+=1
                if self.frame_shift == 1:
                    ctypes.memmove(input_dst, inp_src, self.block_size)
                    input_dst += int(self.block_size)
                else:
                    for j in xrange(self.n_frames):
                        ctypes.memmove(input_dst, inp_src, self.frame_size)
                        input_dst += int(self.frame_size)
                        input_src += int(self.frame_size * self.frame_shift)
            except StopIteration:
                #We have filled exactly i
                if i == 0 or not self.get_smaller_final_batch:
                    raise StopIteration
                else:
                    return self.input_batch[0:i, :]
        return self.input_batch

class SpeechDatasetFileIterator(object):
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset        
        self.return_path = kwargs.get('path', False)
        self.max_files = kwargs.get('n_files', np.inf)
        self.shuffle = kwargs.get('shuffle', False)

    def __iter__(self):
        self.f_index = 0
        self.file_order = range(self.dataset.get_n_files())
        if self.shuffle:
            np.random.shuffle(self.file_order)        
        return self

    def next(self):
        if self.f_index >= self.dataset.get_n_files() or self.f_index > self.max_files:
            raise StopIteration
        else:
            self.f_index += 1
            f_number = self.file_order[self.f_index-1]
            acoustics_file = self.dataset.get_input_file(f_number)
            acoustics = Data.expand_array_in_blocks(acoustics_file, 
                                                    self.dataset.n_frames, 
                                                    self.dataset.frame_shift)            
            if self.dataset.has_output():
                i = 0
                output = []
                times  = self.dataset.get_output_times_file(f_number)
                labels = self.dataset.get_output_values_file(f_number)
                for j in xrange(Data.n_blocks(acoustics_file.shape[0], self.dataset.n_frames, self.dataset.frame_shift)):
                    while (self.dataset.n_frames//2 + j) >= times[i][1] and i < len(times)-1:
                        i+=1                        
                    output.append(labels[i])
                x = [acoustics, np.array(output)] # Get label for each t
            else:
                x = [acoustics]
            if self.return_path:
                x.append(self.dataset.get_file_path(f_number))
            return tuple(x)

def feed_forward_dataset(dataset, layers):
    ##TODO: Move to utils and transform into a mapping operation
    def feedforward(x):        
        for i, l in enumerate(layers):
            if i == 0:
                blocks = Data.expand_array_in_blocks(x, dataset.n_frames, dataset.frame_shift)
                x = l.feedforward(blocks.T).T.copy()
            else:            
                x = l.feedforward(x.T).T.copy()           
        return x
    #Create destination temporary file
    fname = tempfile.mktemp()
    f = h5py.File(fname, "w") 
    for i in xrange(dataset.get_n_files()):
        inp = feedforward(dataset.get_input_file(i))
        f["/data/f%d/input" % (i)] = inp        
        if dataset.has_output():
            f["/data/f%d/output/labels" % (i)] = dataset.get_output_values_file(i)
            if dataset.has_times:
                f["/data/f%d/output/times" % (i)] = dataset.get_output_times_file(i)
                f["/data/f%d/input" % (i)].attrs["WINDOWSIZE"] = dataset.frame_duration * 1e7
                f["/data/f%d/input" % (i)].attrs["TARGETRATE"] = dataset.frame_time_shift * 1e7
            else:
                f["/data/f%d/output/frames" % (i)] = dataset.get_output_times_file(i)
    f.close()
    if dataset.has_output():
        return SpeechDataset(fname, "data/.*", "input", "output", delete_after_use = True)
    else:
        return SpeechDataset(fname, "data/.*", "input", delete_after_use = True)


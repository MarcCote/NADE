from __future__ import division
import h5py
import numpy as np
import tempfile
import os
import struct
import re
from Dataset import Dataset
from SpeechDataset import SpeechDatasetIterator, LabeledSpeechDatasetIterator, SpeechDatasetFileIterator

def read_filelist(fname):
    with open(fname) as fl:
        return [l.rstrip('\n') for l in fl]

def all_files_in_directory(directory, file_extension=""):
    files = []
    def aux(directory):
        basedir = os.path.abspath(directory)        
        subdirlist = []
        for item in os.listdir(directory):
            if os.path.isfile(os.path.join(basedir, item)):
                if item.endswith(file_extension):
                    files.append(os.path.join(basedir, item))
            else:
                subdirlist.append(os.path.join(basedir, item))
        for subdir in subdirlist:
            aux(subdir)
    aux(os.path.abspath(directory))
    files = [os.path.splitext(os.path.relpath(f, directory))[0] for f in files]
    files.sort() #Modifies itself, doesn't return another list or itself :(
    return files

def read_HTK_feature_file(fname):
    """ Read an HTK file and return the data as a matrix, and a dictionary of file attributes. """
    #Read the HTK file to memory
    with open(fname, "rb") as htkf:
        nSamples, sampPeriod, sampSize, parmKind = struct.unpack(">IIhh", htkf.read(12))
        # Is the file compressed?
        compressed = parmKind & 0x400 == 0x400
        if compressed:
            type_to_read = ">i2"
            components = sampSize // 2
            if parmKind & 0x3f == 5:
                A = 32767.0
                B = 0.0
            else:
                A = np.fromfile(htkf, dtype = ">f", count = components)
                B = np.fromfile(htkf, dtype = ">f", count = components)
                nSamples -= 4
        else:            
            type_to_read = ">f"
            components = sampSize // 4           
        data = np.fromfile(htkf, dtype = type_to_read, count = components * nSamples).reshape((nSamples, components))
        if compressed:
            data = (data.astype('f') + B) / A
#    return (np.array(data, dtype=np.float32, order='C'),
#            {'nSamples':nSamples, 'sampPeriod':sampPeriod, 'sampSize':sampSize, 'parmKind':parmKind})
    return np.array(data, dtype=np.float32, order='C')     

def parse_alignments_file(alignments_file, file_list):
    route = None    
    labels = []
    def match_line(line, rexp):
        p = re.compile(rexp)
        res = p.match(line)
        if not res:
            raise Exception("Couldn't match line", line, rexp)
        else:
            return res.groups()
    def match_route(line):
        return str(match_line(line, "#\s*DATA:\s*(.*)$")[0])
    def match_label(line):
        return map(int, match_line(line, "(\d+)\s+(\d+)\s+GMM_(\d+)\s+"))
    with open(alignments_file, "r") as f:
        for line in f:
            if route is None:
                route = match_route(line)
                alignments = list()
            else:
                try:
                    init, length, component = match_label(line) 
                    alignments.append((init, init+length-1, component))                                        
                except:                    
                    #Read new route
                    labels.append((route, np.array(alignments)))                    
                    route = match_route(line)                    
                    alignments = list()                    
    return labels

class HTKSpeechDataset(object):
    def __init__(self, features_path, file_extension="mfc", filelist_fname=None, alignments_file=None, labels_cardinality = None, on_no_alignments="raise"):
        def intersect(l1, l2, str1, str2, key = lambda x: x):
            def aux(x):
                if key(x) in l2:
                    return True
                else:
                    if on_no_alignments == "raise":
                        raise IOError("%s file %s has no %s file" % (str1, key(x), str2))
                    elif on_no_alignments == "suppress":
                        pass
                    else:
                        print("%s file %s has no %s file" % (str1, key(x), str2))
                        return False                
            return filter(aux, l1) 
        self.features_path = features_path
        self.file_extension = file_extension
        if filelist_fname is not None:
            self.file_list = read_filelist(filelist_fname)
        else:
            self.file_list = all_files_in_directory(features_path, file_extension) #file list should not include extension
        # Sort the file list
        self.file_list.sort()
        if alignments_file is not None:
            self.has_times = False #Iterator uses this property to find out whether the alignment times are in secs or frames         
            assert(labels_cardinality is not None)
            self.labels_cardinality = labels_cardinality
            alignments = parse_alignments_file(alignments_file, self.file_list)
            alignments.sort()            
            #Warn or err if a feature file has not segmentation
            self.file_list  = intersect(self.file_list, map(lambda x: x[0], alignments), "feature", "segmentation")
            self.alignments = intersect(alignments, self.file_list, "segmentation", "feature", key = lambda x: x[0])                       
        else:
            self.alignments = None
        self.n_frames = 1
        self.frame_shift = 1
        
    def describe(self):
        return "%s:%s:(%s,%s)" % (self.filename, self.entries_regexp, self.input_dataset, self.output_dataset)

    def get_n_files(self):
        return len(self.file_list)

    def has_output(self):
        return self.alignments is not None
    
    #TODO: Deprecate
    def get_input_dimensionality(self):
        return self.get_input_file(0).shape[1] * self.n_frames

    #TODO: Deprecate
    def get_output_dimensionality(self):
        return self.labels_cardinality
    
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
        return self.file_list[index]

    def get_input_file(self, index):
        return read_HTK_feature_file(os.path.join(self.features_path, "%s.%s" % (self.file_list[index], self.file_extension))) 
        #return np.array(self.input_files[index].value, dtype=np.float32, order='C')

    #TODO: It may not be wise to store all the alignments in memory if the dataset is really huge. It should require about 12bytes per phone-state
    def get_output_values_file(self, index):
        def one_of_k_labels(alignments, cardinality):
            labels = np.zeros((len(alignments), cardinality), dtype="float32")
            for i,n in enumerate(alignments):
                labels[i,n] = 1
            return labels
        return one_of_k_labels(self.alignments[index][1][:,2], self.labels_cardinality)

    def get_output_times_file(self, index):
        return self.alignments[index][1][:, [0,1]]
    
    def set_n_frames(self, n_frames):
        self.n_frames = n_frames
        
    def set_frame_shift(self, frame_shift):
        self.frame_shift = frame_shift
            
    def iterator(self, batch_size = 1, get_smaller_final_batch = False, shuffle = False, just_input = False, max_batches = None):
        if not self.has_output() or just_input:
            return SpeechDatasetIterator(self, batch_size, get_smaller_final_batch, shuffle, self.n_frames, self.frame_shift, max_batches = max_batches)
        else:
            return LabeledSpeechDatasetIterator(self, batch_size, get_smaller_final_batch, shuffle, self.n_frames, self.frame_shift, max_batches = max_batches)

    def file_iterator(self, **kwargs):
        return SpeechDatasetFileIterator(self, **kwargs)
    
    def get_acoustic_statistics(self):
        accum    = np.zeros(self.get_input_dimensionality())
        accum_sq = np.zeros(self.get_input_dimensionality())
        n = 0
        for b in self.iterator(batch_size = 100, get_smaller_final_batch = True, shuffle = False, just_input = True):
            accum    += b.sum(0)
            accum_sq += (b**2).sum(0)
            n += b.shape[0]
        mean = accum/n
        std = np.sqrt(accum_sq/n - mean**2)
        return (mean, std)
    
    #TODO: This could be done much faster without transforming the labels outputs into matrices with 1 of k rows
    def get_state_priors(self, initial_counts = 1):
        accum = np.ones(self.get_output_dimensionality()) * initial_counts
        n = accum.sum()
        for _, states in self.iterator(batch_size = 100, get_smaller_final_batch = True, shuffle = False, just_input = False):
            accum    += states.sum(0)            
            n += states.shape[0]        
        return accum/n
    
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
                       ".*/.*/.*/.*", 
                       tuple(str(i) for i in xrange(len(y))), 
                       delete_after_use = delete)
        
#    def get_data(self, n = None, proportion = None, accept_less = True):        
#        if n is None:
#            total = sum([self.get_file_shape(0, i)[0] for i in xrange(len(self.file_paths))])
#            if proportion is not None: 
#                n = total * proportion
#            else:
#                n = total
#        data = tuple(np.empty((n, self.get_dimensionality(i))) for i in xrange(self.get_arity()))
#        row = 0
#        for fs in self.file_iterator():
#            for i,f in enumerate(fs):
#                increment = min(f.shape[0], n-row)
#                data[i][row:row+increment, :] = f[0:increment, :]
#            row += increment
#            if row >= n:
#                break
#        if accept_less and row < n:
#            return tuple(d[0:row,:] for d in data)
#        else:
#            assert(n == row)
#        return data

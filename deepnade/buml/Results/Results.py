import os
import pickle
import h5py
import numpy as np
import string
import re

def get_keys(f, route):
    keys = [int(x) for x in f[route].keys() ]
    keys.sort()
    return keys

def get_route(*route):
    return string.join(route, '/')

def series(f, route):
    keys = get_keys(f, route)
    x = list()
    y = list()    
    for k in keys:
        x.append(k)
        v = np.array(f[get_route(route, str(k))])        
        if v.shape == (1,1):
            y.append(v.A[0][0])
        else:
            y.append(v)
    return (x,y)
    
def epochs(f, route):
    f, k = series(f, route)
    return k
    
def series_generator(f, route):        
    keys = get_keys(f, route)        
    for k in keys:
        yield (k, np.array(f[get_route(route, str(k))]))

def last(f, route):
    keys = get_keys(f, route)
    return np.array(f[get_route(route, str(keys[-1]))])
    
def load_object(filename, dataset='pickle'):
    f = h5py.File(filename, 'r+')
    obj = pickle.loads(f[dataset].value)
    obj.results.filename = filename
    return obj

#######

class Results(object):    
    def __init__(self, path):
        self.f = h5py.File(path, 'r')

    @staticmethod
    def read(d):
        try:
            return d.value
        except Exception:
            #Is a datagroup
            r = {}
            for x in d.values():
                name = re.search(".*/(.*)", str(x.name))
                r[name.group(1)] = Results.read(x)
            return r

    def get(self, route):
        return Results.read(self.f[route])

    def get_series(self, pattern):
        prefix, postfix = re.search("([^\*]*)\*/?(.+)?", pattern).groups()
        group = self.f[prefix]
        items = group.items()
        if postfix is None:
            read = lambda pair: (float(pair[0]), pair[1].value)
        else:
            read = lambda pair: (float(pair[0]), pair[1][postfix].value)
        series = []
        for x in items:
            try:
                series.append(read(x))
            except Exception: pass
        series = sorted(series)
        return ( map(lambda x: x[0], series), map(lambda x: x[1], series))


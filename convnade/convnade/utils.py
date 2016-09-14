from __future__ import division, print_function

import sys
import json
import numpy as np
import theano
import itertools

import hashlib
from time import time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}

        return json.JSONEncoder(self, obj)


def json_numpy_obj_hook(dct):
    if isinstance(dct, dict) and '__ndarray__' in dct:
        return np.array(dct['__ndarray__'])

    return dct


def generate_uid_from_string(value):
    """ Creates unique identifier from a string. """
    return hashlib.sha256(value.encode()).hexdigest()


def save_dict_to_json_file(path, dictionary):
    """ Saves a dict in a json formatted file. """
    with open(path, "w") as json_file:
        json_file.write(json.dumps(dictionary, indent=4, separators=(',', ': '), cls=NumpyEncoder))


def load_dict_from_json_file(path):
    """ Loads a dict from a json formatted file. """
    with open(path, "r") as json_file:
        return json.loads(json_file.read(), object_hook=json_numpy_obj_hook)


class Timer():
    """ Times code within a `with` statement. """
    def __init__(self, txt):
        self.txt = txt

    def __enter__(self):
        self.start = time()
        print(self.txt + "... ", end="")
        sys.stdout.flush()

    def __exit__(self, type, value, tb):
        print("{:.2f} sec.".format(time()-self.start))


def cartesian(sequences, dtype=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    sequences : list of array-like
        1-D arrays to form the cartesian product of.
    dtype : data-type, optional
        Desired output data-type.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    if dtype is None:
        dtype = np.dtype(type(sequences[0][0]))

    return np.array(list(itertools.product(*sequences)), dtype=dtype)

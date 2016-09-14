from Backend import Backend
import h5py
import os
import string
import re

class HDF5(Backend):
    def __init__(self, path, filename, comment=""):
        self.path = path
        self.filename = "%s.hdf5" % (filename)
        i = 1
        while os.path.exists(os.path.join(self.path, self.filename)):
            i += 1
            self.filename = "%s_%d.hdf5" % (filename, i)
        self.f = h5py.File(os.path.join(self.path, self.filename), 'w')
        
#    def open_file(self):
#        return h5py.File(os.path.join(self.path, self.filename), 'a')

    def write_helper(self, f, route, attribute, value):
        #TODO: This is such a botched job! Everything is there for a reason, though. Be careful.
        rs = "/" + string.join(map(str, route + [attribute]), '/')
        try:
            try:
                del f[rs]
            except:
                pass
            try:
                f[rs] = value.describe()
            except AttributeError:
                f[rs] = value
        except TypeError:
            try:
                for k, v in value.iteritems():
                    self.write_helper(f, route + [attribute], k, v)
            except Exception:
                try:
                    for k, v in value.__dict__.iteritems():
                        self.write_helper(f, route + [attribute], k, v)
                except Exception:
                    f[rs] = str(value)

    def write(self, route, attribute, value):
        self.write_helper(self.f, route, attribute, value)
        
    def read(self, route):
        """
        Reads an hdf5 dataset or a group (if it is a group it is returned as nested hashes)
        """
        def aux(d):
            try:
                return d.value
            except Exception:
                #Is a datagroup
                r = {}
                for x in d.values():
                    name = re.search(".*/(.*)", str(x.name))
                    r[name.group(1)] = aux(x)
                return r
        return aux(self.f[route])

    def __del__(self):
        self.f.close()
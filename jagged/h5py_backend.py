# coding=utf-8
from operator import itemgetter
import os.path as op
import numpy as np
import h5py
from jagged.base import JaggedRawStore
from jagged.misc import ensure_dir
from whatami import whatable


# CHUNKING MATTERS!!!
# Be able to specify dtype


class JaggedByH5Py(JaggedRawStore):

    def __init__(self,
                 path,  # maybe we could also pass the path to the dset inside the h5py file
                 write=False):
        super(JaggedByH5Py, self).__init__()
        self._write = write
        self._path = ensure_dir(op.join(path, self.what().id()))
        self._path = op.join(self._path, 'data.h5')
        self._h5 = None
        self._dset = None

    def append(self, data):
        if not self._write:
            raise Exception('Reading data from repository, cannot write (yes, as you write it!)')

        if self._h5 is None:
            self._h5 = h5py.File(self._path, mode='a')
            if 'data' not in self._h5:
                self._dset = self._h5.create_dataset('data',
                                                     dtype=data.dtype,
                                                     shape=(0, data.shape[1]),
                                                     maxshape=(None, data.shape[1]))
            else:
                self._dset = self._h5['data']

        base = self._dset.shape[0]
        size = len(data)
        self._dset.resize(base + size, axis=0)
        self._dset[base:(base+size)] = data

        return base, size

    def is_writing(self):
        return self._write

    def get(self, segments, columns=None, factory=None):

        # Oversimplified design
        if self.is_writing():
            raise Exception('Writing data to repository, cannot read (yes, as you read it!)')

        # Sanity checks
        if self._h5 is None:
            self._h5 = h5py.File(self._path, mode='r')
            self._dset = self._h5['data']

        ne, nc = self._dset.shape

        if any((base + size) > ne for base, size in segments):
            raise Exception('Out of bounds query')

        # Prepare query and dest
        query_dest = []
        total_size = 0
        for b, l in segments:
            query_dest.append((b, total_size, l))
            total_size += l

        # Retrieve data to a single array
        dest = np.empty((total_size, nc), dtype=self._dset.dtype)

        # does not need to be the optimal strategy, but it usually will
        # bcolz caches 1 chunk at the moment
        views = []
        for base, dest_base, size in sorted(query_dest):
            # This is for eval, and this call is not really correct
            dest[dest_base:dest_base+size] = self._dset[base:(base+size)]  # any way to instruct h5py to copy to the array?
            views.append((dest_base, dest[dest_base:dest_base+size]))

        views = map(itemgetter(1), sorted(views))

        # We can choose between:
        #   - contiguity when writing (i.e. write to adjacent positions, order in dest array on increasing base)
        #   - contiguity for further reads (i.e. make order in dest array as the order of the passed segments)
        # Probably contiguity for further reads is better.

        # Wrap?
        if factory is None:
            return views

        return map(factory, views)

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

# atexit?

if __name__ == '__main__':
    with JaggedByH5Py('/home/santi/testh5py', write=True) as jbh:
        jbh.append(np.zeros((1000, 50)))
    with JaggedByH5Py('/home/santi/testh5py', write=False) as jbh:
        print jbh.get([(0, 300)])

# coding=utf-8
from operator import itemgetter
import os.path as op
import numpy as np
import h5py
from jagged.base import JaggedRawStore


class JaggedByH5Py(JaggedRawStore):

    def __init__(self,
                 path=None,
                 dset_name='data',
                 write=False,
                 chunks=None,
                 compression=None,
                 compression_opts=None,
                 shuffle=False,
                 checkum=False):
        super(JaggedByH5Py, self).__init__()

        self._write = write
        self._path = path
        self._dset_name = dset_name

        if path is not None:
            self._path = op.join(self._path, 'data.h5')
        self._h5 = None
        self._dset = None

        self.chunks = chunks
        self.compression = compression
        self.copts = compression_opts
        self.shuffle = shuffle
        self.fletcher32 = checkum

    def append(self, data):

        if not self._write:
            raise Exception('Cannot write while reading data from repository %s' % self.what().id())

        if self._h5 is None:
            self._h5 = h5py.File(self._path, mode='a')
            if 'data' not in self._h5:
                # http://docs.h5py.org/en/latest/high/dataset.html
                self._dset = self._h5.create_dataset(self._dset_name,
                                                     dtype=data.dtype,
                                                     shape=(0, data.shape[1]),
                                                     maxshape=(None, data.shape[1]),
                                                     chunks=self.chunks,
                                                     compression=self.compression,
                                                     compression_opts=self.copts,
                                                     shuffle=self.shuffle,
                                                     fletcher32=self.fletcher32)
            else:
                self._dset = self._h5[self._dset_name]

        base = self._dset.shape[0]
        size = len(data)
        self._dset.resize(base + size, axis=0)
        self._dset[base:(base+size)] = data

        return base, size

    def is_writing(self):
        return self._write

    def get(self, segments, columns=None, factory=None):

        # Oversimplified design
        if self._write:
            raise Exception('Cannot read while writing data from repository %s' % self.what.id())

        # Read
        if self._h5 is None:
            self._h5 = h5py.File(self._path, mode='r')
            self._dset = self._h5[self._dset_name]

        # Sanity checks
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

        views = []
        for base, dest_base, size in sorted(query_dest):
            # any way to instruct h5py to copy to the array?
            # dest[dest_base:dest_base+size] = self._dset[base:(base+size)]
            views.append((dest_base, dest[dest_base:dest_base+size]))

        # Unpack views
        views = [array for _, array in sorted(views, key=itemgetter(0))]
        # N.B. we must only use the first element of the tuple, this is correct because python sort is stable

        # Wrap?
        if factory is None:
            return views

        return map(factory, views)

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def shape(self):
        if self._dset is None:
            raise Exception('FIXME: the lifecycle is ill defined if calling shape, dtype... before reading or writing')
        return self._dset.shape

    @property
    def dtype(self):
        if self._dset is None:
            raise Exception('FIXME: the lifecycle is ill defined if calling shape, dtype... before reading or writing')
        return self._dset.dtype

    @staticmethod
    def factory(**kwargs):
        return JaggedByH5Py

#
# CHUNKING MATTERS!!!
# Use hdf5 filters/compressors exposed by h5py
# Be able to specify dtype
#

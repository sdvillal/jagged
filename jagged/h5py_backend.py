# coding=utf-8
import os.path as op
import numpy as np
import h5py
from jagged.base import JaggedRawStore, retrieve_contiguous


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

    def _read_segment_to(self, base, size, address):
        if size > 0:
            self._dset.read_direct(address, source_sel=np.s_[base:base+size])

    def get(self, segments=None, columns=None, factory=None, contiguity='read'):

        # Oversimplified design
        if self._write:
            raise Exception('Cannot read while writing data from repository %s' % self.what.id())

        # Open the dataset for reading
        if self._h5 is None:
            self._h5 = h5py.File(self._path, mode='r')
            self._dset = self._h5[self._dset_name]

        # Just read all...
        if segments is None:
            return self._dset[:]

        # ...or read the segments
        ne, nc = self.shape
        views = retrieve_contiguous(segments, self._read_segment_to, self.dtype, ne, nc, contiguity)
        return views if factory is None else map(factory, views)

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    @property
    def shape(self):
        if self._dset is None:
            return None
            # FIXME: the lifecycle is ill defined if calling (shape, dtype...) before reading or writing
        return self._dset.shape

    @property
    def dtype(self):
        if self._dset is None:
            return None
            # FIXME: the lifecycle is ill defined if calling (shape, dtype...) before reading or writing
        return self._dset.dtype

    @staticmethod
    def factory(**kwargs):
        return JaggedByH5Py

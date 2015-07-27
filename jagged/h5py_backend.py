# coding=utf-8
import os.path as op

import numpy as np
import h5py

from jagged.base import JaggedRawStore


class JaggedByH5Py(JaggedRawStore):

    def __init__(self,
                 path=None,
                 dset_name='data',
                 chunks=None,
                 compression=None,
                 compression_opts=None,
                 shuffle=False,
                 checkum=False):
        super(JaggedByH5Py, self).__init__()

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

    def _append(self, data):
        base = len(self)
        size = len(data)
        self._dset.resize(base + size, axis=0)
        self._dset[base:(base+size)] = data
        return base, size

    @property
    def is_writing(self):
        return self.is_open and self._h5.mode != 'r'

    @property
    def is_reading(self):
        return self.is_open and self._h5.mode == 'r'

    @property
    def is_open(self):
        return self._h5 is not None

    def _open_write(self, data=None):
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

    def _open_read(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._path, mode='r')
            self._dset = self._h5[self._dset_name]

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _read_segment_to(self, base, size, columns, address):
        if size > 0:
            if columns is not None:
                if not np.any(np.diff(columns) < 1):
                    self._dset.read_direct(address, source_sel=np.s_[base:base+size, columns])
                else:
                    # h5py only supports increasing order indices
                    #   https://github.com/h5py/h5py/issues/368
                    #   https://github.com/h5py/h5py/issues/368
                    # (boiling down to issues with hdf5 hyperslabs)
                    # better slow than unsupported...
                    columns, inverse = np.unique(columns, return_inverse=True)
                    address[:] = self._dset[base:base+size, tuple(columns)][:, inverse]
                    # n.b.: tuple(columns) to force 2d if columns happens to be a one element list
            else:
                self._dset.read_direct(address, source_sel=np.s_[base:base+size])

    def _backend_attr_hook(self, attr):
        return getattr(self._dset, attr)

    @property
    def shape(self):
        return self._backend_attr('shape')

    @property
    def dtype(self):
        return self._backend_attr('dtype')

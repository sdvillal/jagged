# coding=utf-8
import os.path as op

import numpy as np
import h5py

from jagged.base import LinearRawStorage


class JaggedByH5Py(LinearRawStorage):

    def __init__(self,
                 path=None,
                 journal=None,
                 order='C',
                 contiguity=None,
                 # hdf params
                 dset_name='data',
                 chunklen=None,
                 compression=None,
                 compression_opts=None,
                 shuffle=False,
                 checksum=False):
        super(JaggedByH5Py, self).__init__(path, journal=journal, order=order, contiguity=contiguity)

        self._dset_name = dset_name

        if self._path is not None:
            self._h5_path = op.join(self._path, 'data.h5')
        self._h5 = None
        self._dset = None

        self.chunklen = chunklen
        self.compression = compression
        self.compression_opts = compression_opts
        self.shuffle = shuffle
        self.checksum = checksum

    # --- Read

    def _open_read(self):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, mode='r')
            self._dset = self._h5[self._dset_name]

    def _get_hook(self, base, size, columns, dest):

        # h5py does not handle graciously this case
        if size == 0:
            if dest is not None:
                return dest
            nc = len(columns) if columns is not None else self._dset.shape[-1]
            return np.empty((0, nc), dtype=self._dset.dtype)

        # easy case, no column subset requested
        if columns is None:
            if dest is None:
                return self._dset[base:base+size]  # should we force read with [:]? add to benchmark
            else:
                self._dset.read_direct(dest, source_sel=np.s_[base:base+size])
                return dest

        # N.B.: tuple(columns) to force 2d if columns happens to be a one-element list
        # column-subset is requested
        # h5py only supports increasing order indices in fancy indexing
        #   https://github.com/h5py/h5py/issues/368
        #   https://github.com/h5py/h5py/issues/368
        # (boiling down to issues with hdf5 hyperslabs)

        if not np.any(np.diff(columns) < 1):
            if dest is not None:
                self._dset.read_direct(dest, source_sel=np.s_[base:base+size, tuple(columns)])
                return dest
            else:
                return self._dset[base:base+size, tuple(columns)]

        # better slow than unsupported...
        columns, inverse = np.unique(columns, return_inverse=True)
        if dest is not None:
            dest[:] = self._dset[base:base+size, tuple(columns)][:, inverse]
            return dest
        else:
            return self._dset[base:base+size, tuple(columns)][:, inverse]

    # --- Write

    def _open_write(self, data=None):
        if self._h5 is None:
            self._h5 = h5py.File(self._h5_path, mode='a')
            if 'data' not in self._h5:
                # http://docs.h5py.org/en/latest/high/dataset.html
                chunks = None
                if self.chunklen is not None:
                    chunks = (self.chunklen,) + (data.shape[1:] if data.ndim > 1 else ())
                self._dset = self._h5.create_dataset(self._dset_name,
                                                     dtype=data.dtype,
                                                     shape=(0, data.shape[1]),
                                                     maxshape=(None, data.shape[1]),
                                                     chunks=chunks,
                                                     compression=self.compression,
                                                     compression_opts=self.compression_opts,
                                                     shuffle=self.shuffle,
                                                     fletcher32=self.checksum)
            else:
                self._dset = self._h5[self._dset_name]

    def _append_hook(self, data):
        base = len(self)
        size = len(data)
        self._dset.resize(base + size, axis=0)
        self._dset[base:(base+size)] = data

    # --- Lifecycle

    @property
    def is_writing(self):
        return self.is_open and self._h5.mode != 'r'

    @property
    def is_reading(self):
        return self.is_open and self._h5.mode == 'r'

    @property
    def is_open(self):
        return self._h5 is not None

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

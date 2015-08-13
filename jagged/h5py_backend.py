# coding=utf-8
import os.path as op

import numpy as np
import h5py

from jagged.base import JaggedRawStore


class JaggedByH5Py(JaggedRawStore):

    def __init__(self,
                 path=None,
                 # hdf params
                 dset_name='data',
                 chunklen=None,
                 compression=None,
                 compression_opts=None,
                 shuffle=False,
                 checksum=False):
        super(JaggedByH5Py, self).__init__(path)

        self._dset_name = dset_name

        if self._path is not None:
            self._path = op.join(self._path, 'data.h5')
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
            self._h5 = h5py.File(self._path_or_fail(), mode='r')
            self._dset = self._h5[self._dset_name]

    def _get_hook(self, base, size, columns, dest):
        if dest is None:
            view = self._dset[base:base+size] if columns is None else self._dset[base:base+size, tuple(columns)]
            return view  # should we force read with [:]?
        elif size > 0:
            if columns is not None:
                if not np.any(np.diff(columns) < 1):
                    self._dset.read_direct(dest, source_sel=np.s_[base:base+size, columns])
                else:
                    # h5py only supports increasing order indices
                    #   https://github.com/h5py/h5py/issues/368
                    #   https://github.com/h5py/h5py/issues/368
                    # (boiling down to issues with hdf5 hyperslabs)
                    # better slow than unsupported...
                    columns, inverse = np.unique(columns, return_inverse=True)
                    dest[:] = self._dset[base:base+size, tuple(columns)][:, inverse]
                    # n.b.: tuple(columns) to force 2d if columns happens to be a one-element list
            else:
                self._dset.read_direct(dest, source_sel=np.s_[base:base+size])
        return dest

    # --- Write

    def _open_write(self, data=None):
        if self._h5 is None:
            self._h5 = h5py.File(self._path_or_fail(), mode='a')
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
        return base, size

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

    # --- Properties

    def _backend_attr_hook(self, attr):
        return getattr(self._dset, attr)


# From h5py docs:
# Chunking has performance implications.
# It’s recommended to keep the total size of your chunks between 10 KiB and 1 MiB,
# larger for larger datasets. Also keep in mind that when any element in a chunk is accessed,
# the entire chunk is read from disk.
#

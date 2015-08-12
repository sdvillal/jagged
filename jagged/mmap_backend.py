# coding=utf-8
"""Backend using python/numpy mmap bindings."""
from __future__ import absolute_import, unicode_literals, print_function, division
import os.path as op
import numpy as np
from jagged.base import JaggedRawStore
from jagged.misc import ensure_dir

try:
    import cPickle as pickle
except ImportError:
    import pickle


class JaggedByMemMap(JaggedRawStore):

    def __init__(self, path=None):
        super(JaggedByMemMap, self).__init__(path)

        if self._path is not None:
            self._path = ensure_dir(self._path)
            self._meta = op.join(self._path, 'meta.pkl')
            self._path = op.join(self._path, 'data.mm')

        self._mm = None  # numpy memmap for reading / file handler for writing
        self._dtype = None
        self._shape = None
        self._order = None

    def _append_hook(self, data):
        base = len(self)
        size = len(data)
        self._mm.write(str(data.data))
        self._shape = self._shape[0] + size, self._shape[1]
        return base, size  # FIXME: take into account dtype size

    def write_meta(self):
        """Writes the information about the array."""
        with open(self._meta, 'wb') as writer:
            pickle.dump((self._dtype, self._shape, self._order), writer, protocol=pickle.HIGHEST_PROTOCOL)

    def read_meta(self):
        """Reads the information about the array."""
        if self._dtype is None:
            if not op.isfile(self._meta):
                raise Exception('Meta-information has not been stored yet')
            with open(self._meta, 'rb') as reader:
                self._dtype, self._shape, self._order = pickle.load(reader)
        # TODO: growing length can be easily inferred from file size, no need to update _shape
        #       what is best?

    @property
    def is_writing(self):
        return self.is_open and not self.is_reading

    @property
    def is_reading(self):
        return isinstance(self._mm, np.memmap)

    @property
    def is_open(self):
        return self._mm is not None

    def _open_write(self, data=None):
        if not op.isfile(self._meta):
            if data is None:
                raise ValueError('data must not be None when bootstrapping storage')
            self._dtype = data.dtype
            self._order = 'F' if np.isfortran(data) else 'C'
            self._shape = (0, data.shape[1])
            self.write_meta()
        self.read_meta()
        self._mm = open(self._path, mode='a')

    def _open_read(self):
        self.read_meta()
        if self._mm is None:
            self._mm = np.memmap(self._path,
                                 dtype=self._dtype, shape=self._shape, order=self._order,
                                 mode='r')

    def close(self):
        if self.is_writing:
            self._mm.close()
        self._mm = None
        self.write_meta()

    def _get_hook(self, base, size, columns, address):
        if address is None:
            return self._mm[base:base+size]
        address[:] = self._mm[base:base+size]
        return address

    def _backend_attr_hook(self, attr):
        if attr == 'dtype':
            return self._dtype
        if attr == 'shape':
            return self._shape
        raise ValueError('Unknown attribute %s' % attr)

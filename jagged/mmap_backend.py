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
    """Provides numpy arrays as views of an underlying memmapped array."""

    def __init__(self, path=None):
        super(JaggedByMemMap, self).__init__(path)

        if self._path is not None:
            ensure_dir(self._mmpath)
            self._meta = op.join(self._path, 'meta.pkl')
            self._mmpath = op.join(self._mmpath, 'data.mm')

        self._mm = None  # numpy memmap for reading / file handler for writing
        self._dtype = None
        self._shape = None
        self._order = None

    # --- Read

    def _open_read(self):
        self._read_meta()
        if self._mm is None:
            self._mm = np.memmap(self._mmpath,
                                 dtype=self._dtype, shape=self._shape, order=self._order,
                                 mode='r')

    def _get_hook(self, base, size, columns, address):
        if address is None:
            return self._mm[base:base+size]
        address[:] = self._mm[base:base+size]
        return address

    # --- Write

    def _open_write(self, data=None):
        if not op.isfile(self._meta):
            if data is None:
                raise ValueError('data must not be None when bootstrapping storage')
            self._dtype = data.dtype
            self._order = 'F' if np.isfortran(data) else 'C'
            self._shape = (0, data.shape[1])
            self._write_meta()
        self._read_meta()
        self._mm = open(self._mmpath, mode='a')

    def _append_hook(self, data):
        base = len(self)
        size = len(data)
        self._mm.write(str(data.data))
        self._shape = self._shape[0] + size, self._shape[1]
        return base, size

    # --- Lifecycle

    @property
    def is_writing(self):
        return self.is_open and not self.is_reading

    @property
    def is_reading(self):
        return isinstance(self._mm, np.memmap)

    @property
    def is_open(self):
        return self._mm is not None

    def close(self):
        if self.is_writing:
            self._mm.close()
        self._mm = None
        self._write_meta()

    # --- Storage for underlying array shape, dtype, row/column order

    def _len_by_filelen(self):
        """Helps to check sanity of the array."""
        mmsize_bytes = op.getsize(self._mmpath)
        row_size_bytes = self.shape[1] * self.dtype.itemsize
        num_rows = mmsize_bytes // row_size_bytes
        leftovers = mmsize_bytes % row_size_bytes
        return num_rows, leftovers

    def _check_sizes(self):
        num_rows, leftovers = self._len_by_filelen()
        if 0 != leftovers:
            raise Exception('the memmap file has incomplete data (%d leftover bytes from a partially written array).'
                            '(are you missing transactions?)' % leftovers)
        if num_rows != self.shape[0]:
            raise Exception('the number or rows inferred by file size does not coincide with the length of the store '
                            '(%d != %d)' % (num_rows, self.shape[0]))

    def _write_meta(self):
        """Writes the information about the array."""
        with open(self._meta, 'wb') as writer:
            pickle.dump((self._dtype, self._shape, self._order), writer, protocol=pickle.HIGHEST_PROTOCOL)

    def _read_meta(self):
        """Reads the information about the array."""
        if self._dtype is None:
            if not op.isfile(self._meta):
                raise Exception('Meta-information has not been stored yet')
            with open(self._meta, 'rb') as reader:
                self._dtype, self._shape, self._order = pickle.load(reader)
            self._check_sizes()

    # --- Properties

    def _backend_attr_hook(self, attr):
        self._read_meta()
        if attr == 'dtype':
            return self._dtype
        if attr == 'shape':
            return self._shape
        raise ValueError('Unknown attribute %s' % attr)

#
# TODO: growing length can be easily inferred from file size, no need to update _shape
#       what is best? Probably file size allows for more robust reentrancy...
#
# Remember that resize for numpy mmap objects never resize the file under the hood.
# Fortunately we do not need it here, but was bitten by it when trying to be too clever
#
# Document that when using mode 'auto', everything returned is a view to the large
# memmapped array. TEST that there is no much memory leaking going on. Document that
# for this to work well we need 64bits, python >= 2.6
#

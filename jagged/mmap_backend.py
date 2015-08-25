# coding=utf-8
"""Backend using python/numpy mmap bindings."""
from __future__ import absolute_import, unicode_literals, print_function, division
import os.path as op
from future.utils import PY3
import numpy as np
from jagged.base import LinearRawStorage


class JaggedByMemMap(LinearRawStorage):
    """Provides numpy arrays as views of an underlying memmapped array."""

    def __init__(self, path=None, journal=None, contiguity=None, autoviews=True):
        super(JaggedByMemMap, self).__init__(path, journal=journal, contiguity=contiguity)

        if self._path is not None:
            self._mmpath = op.join(self._path, 'data.mm')

        self._mm = None  # numpy memmap for reading / file handler for writing

        self.autoviews = autoviews

    # --- Read

    def _open_read(self):
        if self._mm is None:
            self._mm = np.memmap(self._mmpath,
                                 dtype=self.dtype, shape=self.shape, order=self.order,
                                 mode='r')
        self._check_sizes()

    def _get_hook(self, base, size, columns, dest):
        view = self._mm[base:base+size]
        if columns is not None:
            view = view[:, tuple(columns)]
        if dest is None:
            return view.copy() if not self.autoviews else view
        dest[:] = view
        return dest

    # --- Write

    def _open_write(self, data=None):
        self._mm = open(self._mmpath, mode='a')

    def _append_hook(self, data):
        self._mm.buffer.write(data.data) if PY3 else self._mm.write(str(data.data))

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

    # --- Storage for underlying array shape, dtype, row/column order

    def _len_by_filelen(self):
        """Helps to check sanity of the array."""
        mmsize_bytes = op.getsize(self._mmpath)
        row_size_bytes = self.shape[1] * self.dtype.itemsize
        num_rows = mmsize_bytes // row_size_bytes
        leftovers = mmsize_bytes % row_size_bytes
        return num_rows, leftovers

    def _check_sizes(self):
        if op.isfile(self._mmpath) and self.shape is not None:
            num_rows, leftovers = self._len_by_filelen()
            if 0 != leftovers:
                raise Exception('the memmap file has incomplete data '
                                '(%d leftover bytes from a partially written array).'
                                '(are you missing transactions?)' % leftovers)
            if num_rows != self.shape[0]:
                raise Exception('the number or rows inferred by file size '
                                'does not coincide with the length of the store '
                                '(%d != %d)' % (num_rows, self.shape[0]))

#
# Remember that resize for numpy mmap objects never resize the file under the hood.
# Numpy mmap ndarray subclass code is simple and neat, can be read in no time. Get back there.
#
# Document that when using mode 'auto', everything returned is a view to the large memmapped array.
#

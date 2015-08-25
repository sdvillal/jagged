# coding=utf-8

import bcolz
from whatami import whatable
import os.path as op
from .base import LinearRawStorage


class JaggedByCarray(LinearRawStorage):
    """
    A Jagged store that uses in-disk `bcolz.carray` to store the data.

    This backend should be good for compressable data accessed sequentially or in batch range queries.
    Random access of small segments will suffer from a considerable performance degradation.

    Usually these stores are backed by many files, so access via network file systems or from spin disks can
    potentially be inefficient.

    Parameters
    ----------
    path : string
      the carray will/must reside here

    contiguity : string, default None
      see base class

    journal : must quack like JaggedJournal, default None
      see base class

    expectedlen : int, default None
      passed to the carray on creation, the expected number of rows in the store
      carray will use it to guess a good chunksize
      the actual size of each chunk will of course depend also on the number of columns
      must be None if `chunklen` is provided

    chunklen : int, default None
      passed to the carray on creation, the number of rows to store per chunk
      the actual size of each chunk will of course depend also on the number of columns
      must be None if `expectedlen` is provided

    cparams : `bcolz.cparams`, default bcolz.cparams(clevel=5, shuffle=False, cname='lz4hc')
      the compression configuration for bcolz; only used if the array is empty
    """

    def __init__(self,
                 path=None,
                 journal=None,
                 contiguity=None,
                 # bcolz params
                 expectedlen=None,
                 chunklen=1024 ** 2,
                 cparams=bcolz.cparams(clevel=5, shuffle=False, cname='lz4hc')):

        super(JaggedByCarray, self).__init__(path, journal=journal, contiguity=contiguity)

        self.expectedlen = expectedlen
        self.chunklen = chunklen
        self.cparams = whatable(cparams, add_properties=True)
        self._bcolz = None

    def _bcolz_dir(self):
        # Needs to be different than self._path or metainfo gets deleted
        return op.join(self.path_or_fail(), 'bcolz')

    # --- Write

    def _open_write(self, data=None):
        if self._bcolz is None:
            try:  # append
                self._bcolz = \
                    bcolz.carray(None,
                                 rootdir=self._bcolz_dir(),
                                 mode='a',
                                 # bcolz conf in case mode='a' semantics change to create, otherwise innocuous
                                 chunklen=self.chunklen,
                                 expectedlen=self.expectedlen,
                                 cparams=self.cparams)
            except:  # create
                self._bcolz = \
                    bcolz.carray(data[0:0],
                                 rootdir=self._bcolz_dir(),
                                 mode='w',
                                 chunklen=self.chunklen,
                                 expectedlen=self.expectedlen,
                                 cparams=self.cparams)

    def _append_hook(self, data):
        self._bcolz.append(data)

    # --- Read

    def _open_read(self):
        if self._bcolz is None:
            self._bcolz = bcolz.carray(None, rootdir=self._bcolz_dir(), mode='r')

    def _get_hook(self, base, size, columns, dest):
        if dest is not None and columns is None:
            # measure if this has any performance benefit, if so, asks for it to be public API
            self._bcolz._getrange(base, size, dest)
            return dest
        if columns is not None:
            view = self._bcolz[base:base+size, columns]
        else:
            view = self._bcolz[base:base+size]
        if dest is not None:
            dest[:] = view
            return dest
        return view

    # --- Lifecycle

    @property
    def is_writing(self):
        return self.is_open and self._bcolz.mode in ('w', 'a')

    @property
    def is_reading(self):
        return self.is_open and self._bcolz.mode == 'r'

    @property
    def is_open(self):
        return self._bcolz is not None

    def close(self):
        if self.is_writing:
            self._bcolz.flush()
        self._bcolz = None

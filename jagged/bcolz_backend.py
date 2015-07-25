# coding=utf-8
from functools import partial

import bcolz

from jagged.base import JaggedRawStoreWithContiguity
from jagged.misc import ensure_dir
from whatami import whatable


class JaggedByCarray(JaggedRawStoreWithContiguity):

    def __init__(self,
                 path=None,
                 write=False,
                 # bcolz params
                 expectedlen=None,
                 chunklen=1024**2,
                 cparams=bcolz.cparams(clevel=5, shuffle=False, cname='lz4hc')):

        super(JaggedByCarray, self).__init__()

        self._path = path
        self._write = write

        self.expectedlen = expectedlen
        self.chunklen = chunklen
        self.cparams = whatable(cparams, add_properties=True)
        self._bcolz = None

    @staticmethod
    def factory(expectedlen=None,
                chunklen=1024**2,
                cparams=bcolz.cparams(clevel=5, shuffle=False, cname='lz4hc')):
        return partial(JaggedByCarray,
                       expectedlen=expectedlen,
                       chunklen=chunklen,
                       cparams=cparams)

    def _path_or_fail(self):
        if self._path is None:
            raise Exception('In-memory ony arrays are not implemented yet')
        return self._path

    def append(self, data):

        if not self._write:
            raise Exception('Cannot write while reading data from repository %s' % self.what().id())

        if self._bcolz is None:
            try:  # try opening 'a' mode
                self._bcolz = \
                    bcolz.carray(data,
                                 rootdir=ensure_dir(self._path_or_fail()),
                                 mode='a',
                                 # bcolz conf in case mode='a' semantics change to create, otherwise innocuous
                                 chunklen=self.chunklen,
                                 expectedlen=self.expectedlen,
                                 cparams=self.cparams)
            except:  # try opening 'w' mode
                self._bcolz = \
                    bcolz.carray(data,
                                 rootdir=ensure_dir(self._path_or_fail()),
                                 mode='w',
                                 chunklen=self.chunklen,
                                 expectedlen=self.expectedlen,
                                 cparams=self.cparams)
        else:
            self._bcolz.append(data)

        return len(self._bcolz) - len(data), len(data)

    def _read_segment_to(self, base, size, columns, address):
        if columns is None:
            self._bcolz._getrange(base, size, address)
        else:
            address[:] = self._bcolz[base:base+size, columns]
            # FIXME: naive implementation, inefficient for contiguity=None

    def _open_read(self):
        # Open bcolz for reading
        if self._bcolz is None:
            self._bcolz = bcolz.carray(None, rootdir=self._path_or_fail(), mode='r')

    def _get_all(self):
        return self._bcolz[:]

    def close(self):
        if self._bcolz is not None and self._write:
            self._bcolz.flush()
        self._bcolz = None

    @property
    def is_writing(self):
        return self._write

    @property
    def shape(self):
        if self._bcolz is None:
            return None
        return self._bcolz.shape

    @property
    def dtype(self):
        if self._bcolz is None:
            return None
        return self._bcolz.dtype

#
# Can bcolz be apt for random row retrieval?
# Because of our query types and usage patterns, probably yes...
#
# Naive random access to BCOLZ in disk sucks a little on random read,
# although it is not as bad as with storing many files in an HDF5 *hierarchy*
#
# ctables are a tad slow in preliminary tests
#
# Because chunking and many files might actually hamper retrieval (esp. if looking at nfs and friends),
# look at just storing in one file, compress with blosc or bloscpack (ala blosc/castra)
#
# TODO: JaggedByCtable
# TODO: JaggedByCastra (i.e. blosc+bloscpack, can use recent c-blosc, one or more files...)
#
# We can choose between:
#   - contiguity when writing (i.e. write to adjacent positions, order in dest array on increasing base)
#   - contiguity for further reads (i.e. make order in dest array as the order of the passed segments)
# Probably contiguity for further reads is better; just for example check speeds of access (way faster)
#
# TODO: ask for carray._getrange to be public API

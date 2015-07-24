# coding=utf-8
from functools import partial
from operator import itemgetter
import numpy as np
import bcolz
from jagged.base import JaggedRawStore
from jagged.misc import ensure_dir
from whatami import whatable


@whatable(add_properties=False)
class JaggedByCarray(JaggedRawStore):

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

    def get(self, segments, columns=None, factory=None):

        if self._write:
            raise Exception('Cannot read while writing data from repository %s' % self.what.id())

        # Open bcolz for reading
        if self._bcolz is None:
            self._bcolz = bcolz.carray(None, rootdir=self._path_or_fail(), mode='r')

        # Just read all?
        if segments is None:
            return self._bcolz[:]

        ne, nc = self.shape

        # Check query sanity
        if any(((base + size) > ne) or (base < 0) for base, size in segments):
            raise Exception('Out of bounds query')

        # Prepare query and dest
        query_dest = []
        total_size = 0
        for b, l in segments:
            query_dest.append((b, total_size, l))
            total_size += l

        # Retrieve data to a single array
        dest = np.empty((total_size, nc), dtype=self._bcolz.dtype)

        # does not need to be the optimal strategy, but it usually will
        # bcolz caches 1 chunk in memory at the moment (does that mean it really reads a whole chunk?)
        #
        views = []
        for i, (base, dest_base, size) in enumerate(sorted(query_dest)):
            view = dest[dest_base:dest_base+size]
            self._bcolz._getrange(base, size, view)  # TODO: ask for getrange to be puclic API
            # dest[dest_base:dest_base+size] = self._bcolz[base:base+size, :]
            views.append((dest_base, view))

        # Unpack views
        views = [array for _, array in sorted(views, key=itemgetter(0))]
        # N.B. we must only use the first element of the tuple, this is correct because python sort is stable

        # print 'Returning...'
        return views if factory is None else map(factory, views)

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

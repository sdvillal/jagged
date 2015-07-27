# coding=utf-8
import bcolz

from jagged.base import JaggedRawStore
from jagged.misc import ensure_dir
from whatami import whatable


class JaggedByCarray(JaggedRawStore):

    def __init__(self,
                 path=None,
                 # bcolz params
                 expectedlen=None,
                 chunklen=1024**2,
                 cparams=bcolz.cparams(clevel=5, shuffle=False, cname='lz4hc')):

        super(JaggedByCarray, self).__init__(path)

        self.expectedlen = expectedlen
        self.chunklen = chunklen
        self.cparams = whatable(cparams, add_properties=True)
        self._bcolz = None

    def _append_hook(self, data):
        self._bcolz.append(data)

    def _open_write(self, data=None):
        if self._bcolz is None:
            try:  # try opening 'a' mode
                self._bcolz = \
                    bcolz.carray(None,
                                 rootdir=ensure_dir(self._path_or_fail()),
                                 mode='a',
                                 # bcolz conf in case mode='a' semantics change to create, otherwise innocuous
                                 chunklen=self.chunklen,
                                 expectedlen=self.expectedlen,
                                 cparams=self.cparams)
            except:  # try opening 'w' mode
                self._bcolz = \
                    bcolz.carray(data[0:0],
                                 rootdir=ensure_dir(self._path_or_fail()),
                                 mode='w',
                                 chunklen=self.chunklen,
                                 expectedlen=self.expectedlen,
                                 cparams=self.cparams)

    def _open_read(self):
        # Open bcolz for reading
        if self._bcolz is None:
            self._bcolz = bcolz.carray(None, rootdir=self._path_or_fail(), mode='r')

    def close(self):
        if self.is_writing:
            self._bcolz.flush()
        self._bcolz = None

    def _get_hook(self, base, size, columns, address):
        if columns is None:
            self._bcolz._getrange(base, size, address)
        else:
            address[:] = self._bcolz[base:base+size, columns]
            # FIXME: naive implementation, inefficient for contiguity=None
            #        build an extension, can almost be copied verbatim from carray_ext.pyx/__getitem__

    @property
    def is_writing(self):
        return self.is_open and self._bcolz.mode in ('w', 'a')

    @property
    def is_reading(self):
        return self.is_open and self._bcolz.mode == 'r'

    @property
    def is_open(self):
        return self._bcolz is not None

    def _backend_attr_hook(self, attr):
        return getattr(self._bcolz, attr)

#
# Can bcolz be apt for random row retrieval and range queries?
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

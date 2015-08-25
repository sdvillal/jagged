from future.builtins import range
from functools import partial
from mmap import mmap, ACCESS_READ
from operator import itemgetter
import os.path as op
from toolz import merge
from jagged.base import JaggedRawStore, JaggedJournal
from jagged.compression.compressors import BloscCompressor
from whatami import whatable


class JaggedByBlosc(JaggedRawStore):

    # Memmapped TODO should allow to just seek and read too
    # Not chunked - hope to keep using bcolz for that
    #
    # Just to check why bcolz is not giving the compression
    # we see in the benchmarks with raw python-blosc
    #
    # Also to investigate why ctable is not working as well
    # as expected after the columnar benchmarks with raw
    # python-blosc

    def __init__(self, path=None, journal=None, compressor=BloscCompressor):
        super(JaggedByBlosc, self).__init__(path, journal=journal)
        self.compressor = compressor  # for whatami, weird
        self._compressor = None
        self._mm = None
        self._writing = None
        self._bytes_journal = None

    def bytes_journal(self):
        if self._bytes_journal is None:
            self._bytes_journal = JaggedJournal(op.join(self.path_or_fail(), 'bytes_journal'))

    def copyconf(self, **params):
        conf = self.what().conf.copy()
        conf['compressor'] = self._make_compressor()
        return whatable(partial(self.__class__, **merge(conf, params)), add_properties=False)

    def _make_compressor(self):
        if self._compressor is None:
            self._compressor = self.compressor(dtype=self.dtype,
                                               shape=self.shape,
                                               order=self.order)
        return self._compressor

    # --- Write

    def _open_write(self, data=None):
        self._mm = open(op.join(self.path_or_fail(), 'data'), 'ab')
        self._writing = True

    def _append_hook(self, data):
        compressor = self._make_compressor()
        compressed = compressor.compress(data)
        self._mm.write(compressed)
        self.bytes_journal().append(compressed)

    # --- Read

    def _open_read(self):
        self._mm = open(op.join(self.path_or_fail(), 'data'), 'r')
        self._mm = mmap(self._mm.fileno(), 0, access=ACCESS_READ)
        self._writing = False

    def _get_views(self, keys, columns):

        if keys is None:
            keys = range(self.narrays)

        keys = [(key, order) for order, key in enumerate(keys)]

        compressor = self._make_compressor()
        views = []
        for key, order in sorted(keys):
            base, size = self.bytes_journal().base_size(key)  # cache these segments?
            array = compressor.decompress(self._mm[base:base+size])
            if columns is not None:
                array = array[:, tuple(columns)]
            views.append((array, order))
        views = list(map(itemgetter(0), sorted(views, key=itemgetter(1))))

        return views

    # --- Lifecycle

    @property
    def is_reading(self):
        return self.is_open and not self.is_writing

    @property
    def is_writing(self):
        return self.is_open and self._writing

    @property
    def is_open(self):
        return self._mm is not None

    def close(self):
        if self.is_open:
            self._mm.close()
        self._mm = None
        self._writing = None

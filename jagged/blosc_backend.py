from mmap import mmap, ACCESS_READ
from operator import itemgetter
import os.path as op

from future.builtins import range

from jagged.base import JaggedRawStore, JaggedJournal
from jagged.compression.compressors import BloscCompressor
from whatami import What


class JaggedByBlosc(JaggedRawStore):

    # Memmapped
    # Not chunked - hope to keep using bcolz for that

    def __init__(self, path=None, journal=None, compressor=BloscCompressor):
        super(JaggedByBlosc, self).__init__(path, journal=journal)
        self.compressor = compressor
        self._mm = None
        self._writing = None
        self._bjournal = None

    def _bytes_journal(self):
        if self._bjournal is None:
            self._bjournal = JaggedJournal(op.join(self.path_or_fail(), 'meta', 'bytes_journal'))
        return self._bjournal

    def _compressor(self):
        if not isinstance(self.compressor, BloscCompressor):
            self.compressor = self.compressor(dtype=self.dtype,
                                              shape=self.shape,
                                              order=self.order)
        return self.compressor

    # --- Custom what() to be available at any circumstance

    def what(self):
        try:
            return What(self.__class__.__name__, {'compressor': self.compressor()})
        except TypeError:
            return What(self.__class__.__name__, {'compressor': self.compressor})

    # --- Write

    def _open_write(self, data=None):
        self._mm = open(op.join(self.path_or_fail(), 'data'), 'ab')
        self._writing = True

    def _append_hook(self, data):
        compressor = self._compressor()
        compressed = compressor.compress(data)
        self._mm.write(compressed)
        self._bytes_journal().append(compressed)

    # --- Read

    def _open_read(self):
        self._mm = open(op.join(self.path_or_fail(), 'data'), 'r')
        self._mm = mmap(self._mm.fileno(), 0, access=ACCESS_READ)
        self._writing = False

    def _read_segment(self, base, size):
        return self._mm[base:base+size]

    def _get_views(self, keys, columns):

        if keys is None:
            keys = range(self.narrays)

        keys = [(key, order) for order, key in enumerate(keys)]

        compressor = self._compressor()
        views = []
        for key, order in sorted(keys):
            base, size = self._bytes_journal().base_size(key)  # cache these segments?
            array = compressor.decompress(self._read_segment(base, size))
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

# coding=utf-8
import gzip
from operator import itemgetter
from jagged.base import JaggedRawStore
import os.path as op
try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle


class JaggedByPickle(JaggedRawStore):
    """A chunked store based on pickle."""

    def __init__(self, path=None, journal=None, arrays_per_chunk=1000, compress=False):
        super(JaggedByPickle, self).__init__(path, journal)
        self.arrays_per_chunk = arrays_per_chunk
        self.compress = compress
        self._cache = []
        self._cached_pickle_num = None
        self._writing = None

    # --- Pickles

    def _pickle_num(self, index):
        return index // self.arrays_per_chunk

    def _pickle_file(self, index):
        path = op.join(self.path_or_fail(), '%d.pkl' % self._pickle_num(index))
        return (path + '.gz') if self.compress else path

    def _save_pickle(self):
        if self.is_writing:
            path = self._pickle_file(self.narrays)
            with gzip.open(path, 'wb') if self.compress else open(path, 'wb') as writer:
                pickle.dump(self._cache, writer, protocol=2)
                # protocol=2 instead of highest to maintain py2 compat of the store

    def _read_pickle(self, index):
        pickle_num = self._pickle_num(index)
        if self._cached_pickle_num != pickle_num:
            try:
                path = self._pickle_file(self.narrays)
                with gzip.open(path, 'rb') if self.compress else open(path, 'rb') as reader:
                    self._cache = pickle.load(reader)
            except IOError:
                self._cache = []
            self._cached_pickle_num = pickle_num

    # --- Cache

    def _cache_full(self):
        return self._cache is not None and len(self._cache) == self.arrays_per_chunk

    # --- Read

    def _open_read(self):
        self._writing = False

    def _get_views(self, keys, columns):
        if keys is None:
            keys = range(self.narrays)

        keys = [(key, order) for order, key in enumerate(keys)]

        views = []
        for key, order in sorted(keys):
            if not 0 <= key < self.narrays:
                raise ValueError('Key not in storage: %d' % key)
            self._read_pickle(key)
            array = self._cache[key % self.arrays_per_chunk]
            if columns is not None:
                array = array[:, tuple(columns)]
            views.append((array, order))
        views = list(map(itemgetter(0), sorted(views, key=itemgetter(1))))

        return views

    # --- Write

    def _open_write(self, data=None):
        self._writing = True
        self._read_pickle(self.narrays)

    def _append_hook(self, data):
        self._cache.append(data.copy())
        if self._cache_full():
            self._save_pickle()
            self._cache = []
            self._cached_pickle_num += 1

    # --- Lifecycle

    @property
    def is_open(self):
        return self._writing is not None

    @property
    def is_writing(self):
        return self.is_open and self._writing

    @property
    def is_reading(self):
        return self.is_open and not self._writing

    def close(self):
        self._save_pickle()
        self._cache = None
        self._cached_pickle_num = None
        self._writing = None

#
# We let pandas decide how to pickle
# In general it would use protocol 2, so pickles can be read also in py2
# We can be as clever as we want with caches: many read caches with LRU, one write cache...
#

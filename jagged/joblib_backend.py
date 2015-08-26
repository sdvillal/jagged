# coding=utf-8
import joblib
from jagged.pickle_backend import JaggedByPickle


class JaggedByJoblib(JaggedByPickle):

    def __init__(self, path=None, journal=None, arrays_per_chunk=1000, compress=False):
        super(JaggedByJoblib, self).__init__(path, journal, arrays_per_chunk, compress)

    def _load(self, path):
        self._cache = joblib.load(path)

    def _dump(self, path):
        compress = 0 if not self.compress else (5 if self.compress is True else self.compress)
        joblib.dump(self._cache, path, compress=compress)  # cache_size

# coding=utf-8
from future.builtins import range
from itertools import chain
import os.path as op
import os
import numpy as np
from jagged.base import JaggedRawStore
from jagged.misc import ensure_dir


class JaggedByNPY(JaggedRawStore):
    """Stores each array in an individual .npy file."""

    def __init__(self, path=None, journal=None):
        super(JaggedByNPY, self).__init__(path, journal=journal)
        self._shards = None
        if path is not None:
            self._all_shards()

    # We can do this memory map (see np.load)

    def _all_shards(self):
        if self._shards is None:
            self._shards = [ensure_dir(op.join(self.path_or_fail(), str(shard))) for shard in range(256)]
        return self._shards
        # random note, 256 is the last cached int in cpython

    def _dest_file(self, index):
        return op.join(self._shards[index % 256], '%d.npy' % index)

    def _infer_numarrays(self):
        numarrays = 0
        for shard in self._shards:
            numarrays = max(chain([numarrays], (int(fn[:-4]) + 1 for fn in os.listdir(shard))))
        return numarrays

    def check_numarrays(self):
        assert self._infer_numarrays() == self.journal().numarrays()

    # --- Write

    def _open_write(self, data=None):
        pass

    def _append_hook(self, data):
        np.save(self._dest_file(self._read_numarrays()), data)

    # --- Read

    def _open_read(self):
        pass

    def _read_one(self, key):
        return np.load(self._dest_file(key))

    def _get_one(self, key, columns):
        data = self._read_one(key)
        if columns is not None:
            data = data[:, tuple(columns)]
        return data

    def _get_views(self, keys, columns):
        if keys is None:
            return list(self._get_one(key, columns) for key in range(self.journal().numarrays()))
        return [self._get_one(key, columns) for key in keys]

    # --- Lifecycle

    @property
    def is_writing(self):
        return True

    @property
    def is_reading(self):
        return True

    @property
    def is_open(self):
        return True

    def close(self):
        pass

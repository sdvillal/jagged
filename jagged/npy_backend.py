# coding=utf-8
from itertools import chain
import os.path as op
import os
import numpy as np
from toolz import partition_all
from jagged.base import JaggedRawStore
from jagged.misc import ensure_dir


class JaggedByNPY(JaggedRawStore):

    def __init__(self, path=None):
        super(JaggedByNPY, self).__init__(path)
        self._writing = False
        self._shards = None
        if path is not None:
            self._all_shards()

    # We can do this memory map (see np.load)
    # We can reuse most of this stuff but drop blosc/bloscpack to the mix

    def _all_shards(self):
        if self._shards is None:
            self._shards = [ensure_dir(op.join(self._path_or_fail(), str(shard))) for shard in range(256)]
        return self._shards
        # random note, 256 is the last cached int in cpython

    def _dest_file(self, index):
        return op.join(self._shards[index % 256], '%d.npy' % index)

    def _infer_numarrays(self):
        numarrays = 0
        for shard in self._shards:
            numarrays = max(chain([numarrays], (int(fn[:-4]) + 1 for fn in os.listdir(shard))))
        return numarrays

    # --- Write

    def _open_write(self, data=None):
        self._writing = True

    def _append_hook(self, data):
        self._write_one(data)
        return self._read_numarrays()

    def _write_one(self, data):
        np.save(self._dest_file(self._read_numarrays()), data)

    # --- Read

    def _open_read(self):
        self._writing = False

    def _read_one(self, key):
        return np.load(self._dest_file(key))

    def _get_one(self, key, columns):
        data = self._read_one(key)
        if columns is not None:
            data = data[:, tuple(columns)]
        return data

    def _get_views(self, keys, columns):
        if keys is None:
            return [np.vstack([self._get_one(key, columns) for key in range(self._read_numarrays())])]
        return [self._get_one(key, columns) for key in keys]

    # --- Iterate

    def iter_segments(self, segments_per_chunk=None):
        if segments_per_chunk is None:
            for key in range(self._read_numarrays()):
                yield self.get([key])
        elif segments_per_chunk <= 0:
            raise ValueError('chunksize must be None or bigger than 0, it is %r' % segments_per_chunk)
        else:
            for segments in partition_all(segments_per_chunk, range(self._read_numarrays())):
                yield self.get(segments)

    def iter_rows(self, max_rows_per_chunk):
        raise NotImplementedError()

    # --- Lifecycle

    @property
    def is_writing(self):
        return self._writing is True

    @property
    def is_reading(self):
        return self._writing is False

    @property
    def is_open(self):
        return self._writing is not None

    def close(self):
        self._writing = None

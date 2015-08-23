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
        self._numarrays = None
        self._size = None
        self._template = None
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

    def _npy(self, index):
        return op.join(self._shards[index % 256], '%d.npy' % index)

    def _infer_numarrays(self):
        numarrays = 0
        for shard in self._shards:
            numarrays = max(chain([numarrays], (int(fn[:-4]) + 1 for fn in os.listdir(shard))))
        return numarrays

    def _read_numarrays(self):
        if self._numarrays is None:
            self._numarrays = self._infer_numarrays()
        return self._numarrays

    def _read_size(self):
        if self._size is None:
            if op.isfile(op.join(self._path_or_fail(), 'size.txt')):
                with open(op.join(self._path_or_fail(), 'size.txt')) as reader:
                    self._size = int(reader.read())
            else:
                return 0
        return self._size

    def _write_size(self):
        if self._size is not None:
            with open(op.join(self._path_or_fail(), 'size.txt'), 'w') as writer:
                writer.write('%d' % self._size)

    def _read_template(self):
        template_path = op.join(self._path_or_fail(), 'template.npy')
        if self._template is None:
            if op.isfile(template_path):
                self._template = np.load(template_path)
        return self._template

    def _write_template(self, data):
        template_path = op.join(self._path_or_fail(), 'template.npy')
        np.save(template_path, data[:0])

    def is_compatible(self, data):
        template = self._read_template()
        if template is None:
            return True
        return (template.dtype >= data.dtype and
                data.shape[-1] == template.shape[-1] and
                np.isfortran(data) == np.isfortran(data))
        # Obviously we could just store arbitrary arrays
        # But lets keep jagged contracts...

    # --- Write

    def _open_write(self, data=None):
        if self._read_template() is None:
            self._write_template(data)
        assert self.is_compatible(data)
        self._writing = True

    def _append_hook(self, data):
        assert self.is_compatible(data)
        np.save(self._npy(self._read_numarrays()), data)
        self._numarrays += 1
        if self._size is None:
            self._size = len(data)
        else:
            self._size += len(data)
        self._write_size()
        return self._numarrays - 1

    # --- Read

    def _open_read(self):
        self._writing = False

    def _get_one(self, key, columns):
        data = np.load(self._npy(key))
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
        self._write_size()
        # we could do here journaling, save stats like sizes and number of rows... later iteration

    # --- Properties

    def _backend_attr_hook(self, attr):
        if self._read_template() is None:
            return None
        if attr == 'shape':
            return self._read_size(), self._read_template().shape[1]
        return getattr(self._read_template(), attr)

if __name__ == '__main__':

    jbn = JaggedByNPY(op.join(op.expanduser('~'), 'npys-test'))
    jbn.append(np.ones((10, 10)))
    print(jbn.shape)
    print(jbn.get((5, 3, 2), columns=(2, 4)))

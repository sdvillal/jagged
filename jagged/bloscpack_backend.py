# coding=utf-8
import os.path as op
import bloscpack
from bloscpack.defaults import DEFAULT_CHUNK_SIZE
from jagged.npy_backend import JaggedByNPY
from bloscpack.numpy_io import pack_ndarray_file, unpack_ndarray_file


class JaggedByBloscpack(JaggedByNPY):

    def __init__(self,
                 path=None,
                 journal=None,
                 # blosc
                 clevel=5,
                 shuffle=True,
                 cname='lz4hc',
                 # bloscpack
                 chunk_size=DEFAULT_CHUNK_SIZE,
                 offsets=False,
                 checksum='None'):
        super(JaggedByBloscpack, self).__init__(path, journal=journal)
        self.clevel = clevel
        self.shuffle = shuffle
        self.cname = cname
        self.offsets = offsets
        self.checksum = checksum
        self.chunk_size = chunk_size
        self._bp_args = None
        self._blosc_args = None

    def _dest_file(self, index):
        return op.join(self._shards[index % 256], '%d.blp' % index)

    def _read_one(self, key):
        return unpack_ndarray_file(self._dest_file(key))

    def _append_hook(self, data):
        if self._bp_args is None:
            self._bp_args = bloscpack.BloscpackArgs(offsets=self.offsets,
                                                    checksum=self.checksum)
        if self._blosc_args is None:
            self._blosc_args = bloscpack.BloscArgs(typesize=self.dtype.itemsize,
                                                   clevel=self.clevel,
                                                   shuffle=self.shuffle,
                                                   cname=self.cname)
        pack_ndarray_file(data, self._dest_file(self._read_numarrays()),
                          chunk_size=self.chunk_size,
                          blosc_args=self._blosc_args,
                          bloscpack_args=self._bp_args)

from contextlib import contextmanager
import numpy as np

from whatami import whatable

try:
    import blosc
except ImportError:  # pragma: no cover
    blosc = None


# --- Consistent compressors API


@whatable
class Compressor(object):

    def compress(self, data):
        raise NotImplementedError()

    def decompress(self, cdata):
        raise NotImplementedError()

    def uncompress(self, cdata):  # pragma: no cover
        return self.decompress(cdata)


# --- Pimping Blosc to compress our arrays


@contextmanager
def _blosc_nthreads(n_threads):
    old_nthreads = blosc.set_nthreads(n_threads)
    yield old_nthreads
    if old_nthreads != n_threads:
        blosc.set_nthreads(old_nthreads)


class BloscCompressor(Compressor):

    # This has quite an overhead beyond compression ATM

    def __init__(self, shuffle=True, level=5, cname='lz4hc', n_threads=1, dtype=None, shape=None, order=None):
        super(BloscCompressor, self).__init__()
        self.shuffle = shuffle
        self.level = level
        self.cname = cname
        self.n_threads = n_threads
        self._dtype = dtype
        self._shape = shape
        self._order = order

    def compress(self, x):
        with _blosc_nthreads(self.n_threads):
            x = np.ascontiguousarray(x)  # LOOK AT THIS
            shape, dtype, order = x.shape, x.dtype, ('F' if np.isfortran(x) else 'C')
            if self._dtype is None:
                self._shape, self._dtype, self._order = shape, dtype, order
            else:
                assert order == self._order
                assert dtype == self._dtype
                assert len(self._shape) == 1 or shape[1] == self._shape[1]
            return blosc.compress_ptr(x.__array_interface__['data'][0],
                                      x.size, x.dtype.itemsize,
                                      shuffle=self.shuffle, cname=self.cname, clevel=self.level)

    def decompress(self, cx):
        with _blosc_nthreads(self.n_threads):
            x = blosc.decompress(cx)
            x = np.frombuffer(x, dtype=self._dtype)  # beware gets an immutable array
            if self._order == 'F':
                np.asfortranarray(x)  # correct? makes copy and screwes up?
            if len(self._shape) > 1:
                x = x.reshape(-1, self._shape[1])
            return x

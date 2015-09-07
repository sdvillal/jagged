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
    if n_threads is None:
        yield
    else:
        old_nthreads = blosc.set_nthreads(n_threads)
        yield
        if old_nthreads != n_threads:
            blosc.set_nthreads(old_nthreads)


class BloscCompressor(Compressor):

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
            order = 'F' if np.isfortran(x) else 'C'
            x = np.require(x, requirements=[order])  # beware, can make a copy
            shape, dtype = x.shape, x.dtype
            if self._dtype is None:
                self._shape, self._dtype, self._order = shape, dtype, order
            else:
                assert order == self._order
                assert dtype == self._dtype
                assert len(self._shape) == 1 or shape[1:] == self._shape[1:]
            return blosc.compress_ptr(x.__array_interface__['data'][0],
                                      x.size, x.dtype.itemsize,
                                      shuffle=self.shuffle, cname=self.cname, clevel=self.level)

    def decompress(self, cx):
        with _blosc_nthreads(self.n_threads):
            x = blosc.decompress(cx)
            x = np.frombuffer(x, dtype=self._dtype)  # beware, immutable array
            if len(self._shape) > 1:
                x = x.reshape(-1, *self._shape[1:], order=self._order)
            return x

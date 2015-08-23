import zlib

import numpy as np

from whatami import whatable

try:
    import bz2
except ImportError:
    bz2 = None

try:
    import lz4
except ImportError:
    lz4 = None

try:
    import zstd
except ImportError:
    zstd = None

try:
    import blosc
except ImportError:
    blosc = None


# --- Consistent compressors API


@whatable
class Compressor(object):

    def compress(self, data):
        raise NotImplementedError()

    def decompress(self, cdata):
        raise NotImplementedError()

    def uncompress(self, cdata):
        return self.decompress(cdata)

#
# class ZLIB(Compressor):
#     """
#     Examples
#     --------
#     >>> x = '1' * 8
#     >>> compressor = ZLIB()
#     >>> cx = compressor.compress(x)
#     >>> x == compressor.decompress(cx)
#     True
#     >>> print(compressor.what().id())
#     ZLIB(level=5)
#     """
#
#     def __init__(self, level=5):
#         self.level = level
#
#     def compress(self, data):
#         return zlib.compress(data, self.level)
#
#     def decompress(self, cdata):
#         return zlib.decompress(cdata)
#
#
# class BZ2(Compressor):
#     """
#     Examples
#     --------
#     >>> x = '1' * 8
#     >>> compressor = BZ2()
#     >>> cx = compressor.compress(x)
#     >>> x == compressor.decompress(cx)
#     True
#     >>> print(compressor.what().id())
#     BZ2(level=5)
#     """
#
#     def __init__(self, level=5):
#         self.level = level
#
#     def compress(self, data):
#         return bz2.compress(data, compresslevel=self.level)
#
#     def decompress(self, cdata):
#         return bz2.decompress(cdata)
#
#
# class LZ4(Compressor):
#     """
#     Examples
#     --------
#     >>> x = '1' * 8
#     >>> compressor = LZ4()
#     >>> cx = compressor.compress(x)
#     >>> x == compressor.decompress(cx)
#     True
#     >>> print(compressor.what().id())
#     LZ4(hc=True)
#     """
#
#     def __init__(self, hc=True):
#         if lz4 is None:
#             raise Exception('Cannot find lz4; please install python-lz4')
#         self.hc = hc
#
#     def compress(self, data):
#         if self.hc:
#             return lz4.compressHC(data)
#         return lz4.compress(data)
#
#     def decompress(self, cdata):
#         return lz4.uncompress(cdata)
#
#
# class ZSTD(Compressor):
#     """
#     Examples
#     --------
#     >>> x = '1' * 8
#     >>> compressor = ZSTD()
#     >>> cx = compressor.compress(x)
#     >>> x == compressor.decompress(cx)
#     True
#     >>> print(compressor.what().id())
#     ZSTD()
#     """
#
#     def __init__(self):
#         if zstd is None:
#             raise Exception('Cannot find zstd; please install python-zstd')
#
#     def compress(self, data):
#         return zstd.compress(data)
#
#     def decompress(self, cdata):
#         return zstd.decompress(cdata)
#
#
# class BLOSC(Compressor):
#
#     def __init__(self, typesize, level=5, shuffle=True, cname='blosclz', n_threads=1):
#         if blosc is None:
#             raise Exception('Cannot find blosc; please install python-blosc')
#         blosc.set_nthreads(n_threads)  # mmmm global
#         self.level = level
#         self.shuffle = shuffle  # 0=None, 1=ByteShuffle, 2=BitShuffle (blosc >= 1.7.0, need AVX2 to be fast)
#         self.cname = cname
#         self._typesize = typesize
#
#     def compress(self, data):
#         return blosc.compress(data,
#                               typesize=self._typesize,
#                               clevel=self.level,
#                               shuffle=self.shuffle,
#                               cname=self.cname)
#
#     def decompress(self, cdata):
#         return blosc.decompress(cdata)
#


def compress_diff(x, shuffle=True, level=5, cname='lz4hc'):
    # Diff compressor using numpy (slow) and blosc
    # Preconditions data for better compressibility / storage (should better be part of filtering pipeline)
    # Easy to speed-up with cython, try numba first maybe
    # But anyway this introduces a lot of rounding errors...
    xd = np.empty_like(x)
    xd[0] = x[0]
    xd[1:] = np.diff(x, axis=0)
    compressed = blosc.compress_ptr(xd.__array_interface__['data'][0],
                                    xd.size, xd.dtype.itemsize,
                                    shuffle=shuffle, cname=cname, clevel=level)
    return compressed, x.shape, x.dtype, ('F' if np.isfortran(x) else 'C')


def decompress_diff(xc, shape, dtype, order):
    x = np.empty(shape, dtype, order)
    blosc.decompress_ptr(xc, x.__array_interface__['data'][0])
    x[1:] = np.cumsum(x[1:], axis=0) + x[0]
    return x


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
        blosc.set_nthreads(self.n_threads)  # mmmm global, put in a context to reset
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
        blosc.set_nthreads(self.n_threads)  # mmmm global, put in a context to reset
        x = blosc.decompress(cx)
        x = np.frombuffer(x, dtype=self._dtype)  # beware gets an immutable array
        if self._order == 'F':
            np.asfortranarray(x)  # correct? makes copy and screwes up?
        if len(self._shape) > 1:
            x = x.reshape(-1, self._shape[1])
        return x


class DiffCompressor(Compressor):

    def __init__(self, shuffle=True, level=5, n_threads=1, cname='lz4hc', dtype=None, shape=None, order=None):
        super(DiffCompressor, self).__init__()
        self.shuffle = shuffle
        self.level = level
        self.cname = cname
        self.n_threads = n_threads
        self._dtype = dtype
        self._shape = shape
        self._order = order

    def compress(self, x):
        blosc.set_nthreads(self.n_threads)  # mmmm global, put in a context to reset
        shape, dtype, order = x.shape, x.dtype, ('F' if np.isfortran(x) else 'C')
        if self._dtype is None:
            self._shape, self._dtype, self._order = shape, dtype, order
        else:
            assert order == self._order
            assert dtype == self._dtype
            assert len(self._shape) == 1 or shape[1] == self._shape[1]
        return compress_diff(x, shuffle=self.shuffle, level=self.level, cname=self.cname)[0]

    def decompress(self, cx):
        blosc.set_nthreads(self.n_threads)  # mmmm global, put in a context to reset
        x = blosc.decompress(cx)
        x = np.fromstring(x, dtype=self._dtype)  # one more copy than if we use decompress_pointer
        if self._order == 'F':
            np.asfortranarray(x)  # correct? makes copy and screwes up?
        if len(self._shape) > 1:
            x = x.reshape(-1, self._shape[1])
        x[1:] = np.cumsum(x[1:], axis=0) + x[0]  # this cascades errors like crazy
        return x


def cratio(x, compressor, check_roundtrip=True):
    uncompressed = x
    compressed = compressor.compress(uncompressed)
    if check_roundtrip:
        assert uncompressed == compressor.decompress(compressed)
    return float(len(uncompressed)) / len(compressed)


def ncd(x, y, compressor=ZLIB(), cx=None, cy=None, cxy=None):
    # See:
    #   https://en.wikipedia.org/wiki/Normalized_compression_distance
    #   http://jeremykun.com/2012/12/04/information-distance-a-primer/
    #   http://complearn.org/ncd.html
    # Remember practically not commutative
    if cx is None:
        cx = compressor.compress(x)
    if cy is None:
        cy = compressor.compress(y)
    if cxy is None:
        cxy = compressor.compress(x + y)
    return (len(cxy) - min(len(cx), len(cy))) / float(max(len(cx), len(cy)))


if __name__ == '__main__':

    x = np.random.uniform(size=(10, 100))
    x = np.arange(10000)
    x = np.linspace(0, 10000, 10000, dtype=np.float32)

    compressed, shape, dtype, order = compress_diff(x)
    decompressed = decompress_diff(compressed, shape, dtype, order)

    compressor = DiffCompressor()
    compressor = BloscCompressor()
    compressed = compressor.compress(x)
    decompressed = compressor.decompress(compressed)

    print(len(x.tostring()))
    print(len(compressed))
    print('cr=%.4f' % (float(len(x.tostring())) / len(compressed)))

    assert np.allclose(x, decompressed)

#
# Try bcolz? (can we make it appear to work on bytes?)
# Something chunked could add a lot of overhead for small
# individual time series.
#
# LZ4 and zstd and blosc
#   https://code.google.com/p/lz4/
#
# Dependency installation:
#   conda install -c https://conda.binstar.org/flyem lz4
#   https://github.com/sergey-dryabzhinsky/python-zstd
#   conda install blosc
# Actually what I do is to checkout python-blosc, put there latest blosc, build and install
# (without -e, it is not devel release friendly)
#
# Eventually try the bitshuffle filter:
#  http://www.blosc.org/blog/new-bitshuffle-filter.html
#  http://nbviewer.ipython.org/gist/alimanfoo/e93a532eb6bde311ea39/genotype_bitshuffle.ipynb
# First try, it segfaulted. Remember that my only machine with AVX2 is strall.
#
# Remember fromstring copies data
# http://stackoverflow.com/questions/22236749/numpy-what-is-the-difference-between-frombuffer-and-fromstring
#

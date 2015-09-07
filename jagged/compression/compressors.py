# coding=utf-8
"""Compress and decompress collections of different length arrays."""
from __future__ import division
from contextlib import contextmanager

import numpy as np

from whatami import whatable

import zlib

try:
    import blosc
except ImportError:  # pragma: no cover
    blosc = None

try:
    import bz2
except ImportError:  # pragma: no cover
    bz2 = None

try:
    import lz4
except ImportError:  # pragma: no cover
    lz4 = None

try:
    import zstd
except ImportError:  # pragma: no cover
    zstd = None


# --- Jagged compressors API


@whatable(add_properties=False)
class JaggedCompressor(object):

    def __init__(self, dtype=None, shape=None, order=None):
        super(JaggedCompressor, self).__init__()
        self._dtype = dtype
        self._shape = shape
        self._order = order

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self._shape is None:
            self._shape = shape
        if len(self._shape) > 1 and self._shape[1:] != shape[1:]:
            raise ValueError('Cannot change shape from %r to %r' % ((-1,) + self.shape[1:],
                                                                    (-1,) + shape[1:]))

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, dtype):
        if self._dtype is None:
            self._dtype = dtype
        if self._dtype != dtype:
            raise ValueError('Cannot change dtype from %r to %r' % (self.dtype, dtype))

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        if self._order is None:
            self._order = order
        if self._order != order:
            raise ValueError('Cannot change order from %r to %r' % (self.order, order))

    def compress(self, x):
        order = 'F' if np.isfortran(x) else 'C'
        self.shape, self.dtype, self.order = x.shape, x.dtype, order
        # make data contiguous in memory; beware, can make a copy, revisit...
        x = np.require(x, requirements=[order])
        return self._compress_hook(x)

    def _compress_hook(self, x):
        raise NotImplementedError()

    def decompress(self, cx, dest=None):
        x = self._decompress_hook(cx, dest)
        if x is dest:
            return x
        x = np.frombuffer(x, dtype=self.dtype)  # beware, immutable array
        if len(self._shape) > 1:
            x = x.reshape(-1, *self.shape[1:], order=self.order)
        if dest is not None:
            dest[:] = x
            x = dest
        return x

    def _decompress_hook(self, cx, dest=None):
        raise NotImplementedError()

    def uncompress(self, cx, dest=None):  # pragma: no cover
        return self.decompress(cx, dest=dest)


# --- Blosc

if blosc:

    @contextmanager
    def _blosc_nthreads(n_threads):
        if n_threads is None:
            yield
        else:
            old_nthreads = blosc.set_nthreads(n_threads)
            yield
            if old_nthreads != n_threads:
                blosc.set_nthreads(old_nthreads)


    class JaggedCompressorByBlosc(JaggedCompressor):
        """
        A simplistic wrapper over blosc to compress and decompress numpy arrays.

        An instance of this class can only manage one type of arrays. The array type
        is set either using the constructor or on the first call to `compress`. Then all
        the compressed arrays must adhere to the specified type, except for the leading,
        dimension that can change. That is, compressing arrays of different length
        is allowed, any other change will make an exception to be raised.

        For a more featureful alternative, see bloscpack:
          https://github.com/Blosc/bloscpack

        Parameters
        ----------
        cname : string, default 'lz4hc'
          The compressor to use. As of blosc 1.7.0, this can be one of ('blosclz', 'snappy', 'lz4', 'lz4hz', 'zlib')

        level : int, default 5
          Compression level, or how aggressive the compressor is (governs, for example, the size of blocks)
          The concrete meaning is dependent on the compressor used

        shuffle : bool, default True
          Use the shuffle filter (incompatible with bitshuffle)

        bitshuffle : bool, default False
          Use the bitshuffle filter, available in c-blosc >= 1.7.0
          This is incompatible with the shuffle filter
          It is fast only if the processor supports AVX2 instructions

        n_threads : int, default 1
          The number of threads that blosc can use.
          Use None to let the global value for blosc_n_threads be used.


        Examples
        --------
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        >>> bc = JaggedCompressorByBlosc()
        >>> Xr = bc.decompress(bc.compress(X))
        >>> np.testing.assert_array_equal(X, Xr)
        >>> np.isfortran(Xr)
        True
        >>> Xd = np.empty_like(X)
        >>> Xdd = bc.decompress(bc.compress(X), dest=Xd)
        >>> Xd is Xdd
        True
        >>> np.testing.assert_array_equal(X, Xdd)
        >>> np.isfortran(Xdd)
        True
        >>> print(bc.what().id())
        JaggedCompressorByBlosc(bitshuffle=False,clevel=5,cname='lz4hc',n_threads=1,shuffle=True)
        """

        def __init__(self, cname='lz4hc', clevel=5, shuffle=True, bitshuffle=False, n_threads=1,
                     dtype=None, shape=None, order=None):
            super(JaggedCompressorByBlosc, self).__init__(dtype=dtype, shape=shape, order=order)
            if shuffle and bitshuffle:
                raise ValueError('shuffle and bitshuffle filters cannot be used at the same time')
            self.cname = cname
            self.clevel = clevel
            self.shuffle = shuffle
            self.bitshuffle = bitshuffle
            self._filter = 1 if shuffle else 2 if bitshuffle else 0
            self.n_threads = n_threads

        def _compress_hook(self, x):
            with _blosc_nthreads(self.n_threads):
                return blosc.compress_ptr(x.__array_interface__['data'][0],
                                          x.size, x.dtype.itemsize,
                                          shuffle=self._filter, cname=self.cname, clevel=self.clevel)

        def _decompress_hook(self, cx, dest=None):
            with _blosc_nthreads(self.n_threads):
                if dest is None:
                    return blosc.decompress(cx)
                else:
                    blosc.decompress_ptr(cx, dest.__array_interface__['data'][0])
                    return dest


# --- zlib

class JaggedCompressorByZLIB(JaggedCompressor):
    """
    Examples
    --------
    >>> X = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    >>> bc = JaggedCompressorByZLIB(clevel=3)
    >>> Xr = bc.decompress(bc.compress(X))
    >>> np.testing.assert_array_equal(X, Xr)
    >>> np.isfortran(Xr)
    True
    >>> Xd = np.empty_like(X)
    >>> Xdd = bc.decompress(bc.compress(X), dest=Xd)
    >>> Xd is Xdd
    True
    >>> np.testing.assert_array_equal(X, Xdd)
    >>> np.isfortran(Xdd)
    True
    >>> print(bc.what().id())
    JaggedCompressorByZLIB(clevel=3)
    """

    def __init__(self, clevel=5, dtype=None, shape=None, order=None):
        super(JaggedCompressorByZLIB, self).__init__(dtype=dtype, shape=shape, order=order)
        self.clevel = clevel

    def _compress_hook(self, x):
        return zlib.compress(np.getbuffer(x), self.clevel)

    def _decompress_hook(self, cx, dest=None):
        return zlib.decompress(cx)


# --- bzip2

if bz2:

    class JaggedCompressorByBZ2(JaggedCompressor):
        """
        Examples
        --------
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        >>> bc = JaggedCompressorByBZ2(clevel=3)
        >>> Xr = bc.decompress(bc.compress(X))
        >>> np.testing.assert_array_equal(X, Xr)
        >>> np.isfortran(Xr)
        True
        >>> Xd = np.empty_like(X)
        >>> Xdd = bc.decompress(bc.compress(X), dest=Xd)
        >>> Xd is Xdd
        True
        >>> np.testing.assert_array_equal(X, Xdd)
        >>> np.isfortran(Xdd)
        True
        >>> print(bc.what().id())
        JaggedCompressorByBZ2(clevel=3)
        """

        def __init__(self, clevel=5, dtype=None, shape=None, order=None):
            super(JaggedCompressorByBZ2, self).__init__(dtype=dtype, shape=shape, order=order)
            self.clevel = clevel

        def _compress_hook(self, x):
            return bz2.compress(np.getbuffer(x), compresslevel=self.clevel)

        def _decompress_hook(self, cx, dest=None):
            return bz2.decompress(cx)

# --- LZ4

if lz4:

    class JaggedCompressorByLZ4(JaggedCompressor):
        """
        Examples
        --------
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        >>> bc = JaggedCompressorByLZ4(hc=True)
        >>> Xr = bc.decompress(bc.compress(X))
        >>> np.testing.assert_array_equal(X, Xr)
        >>> np.isfortran(Xr)
        True
        >>> Xd = np.empty_like(X)
        >>> Xdd = bc.decompress(bc.compress(X), dest=Xd)
        >>> Xd is Xdd
        True
        >>> np.testing.assert_array_equal(X, Xdd)
        >>> np.isfortran(Xdd)
        True
        >>> print(bc.what().id())
        JaggedCompressorByLZ4(hc=True)
        """

        def __init__(self, hc=True, dtype=None, shape=None, order=None):
            super(JaggedCompressorByLZ4, self).__init__(dtype=dtype, shape=shape, order=order)
            self.hc = hc

        def _compress_hook(self, x):
            if self.hc:
                return lz4.compressHC(np.getbuffer(x))
            return lz4.compress(np.getbuffer(x))

        def _decompress_hook(self, cx, dest=None):
            return lz4.uncompress(cx)


if zstd:

    class JaggedCompressorByZSTD(JaggedCompressor):
        """
        Examples
        --------
        >>> X = np.array([[1, 2, 3], [4, 5, 6]], order='F')
        >>> bc = JaggedCompressorByZSTD()
        >>> Xr = bc.decompress(bc.compress(X))
        >>> np.testing.assert_array_equal(X, Xr)
        >>> np.isfortran(Xr)
        True
        >>> Xd = np.empty_like(X)
        >>> Xdd = bc.decompress(bc.compress(X), dest=Xd)
        >>> Xd is Xdd
        True
        >>> np.testing.assert_array_equal(X, Xdd)
        >>> np.isfortran(Xdd)
        True
        >>> print(bc.what().id())
        JaggedCompressorByZSTD()
        """

        def _compress_hook(self, x):
            return zstd.compress(x)

        def _decompress_hook(self, cx, dest=None):
            return zstd.decompress(cx)


# --- Diff compressor

class JaggedCompressorWithDiff(JaggedCompressor):

    # Proof of concept for compression preconditioning and compressor decorators
    # Uses python + numpy; slow but...
    # Kinda of delta filtering, quite stupid
    # (deltas are going to be all over the place for floats most of the time anyway)
    # This also introduces quite rounding errors... could be tamed by using "landmark each x observations"
    # Easy to speed-up with cython, try numba first maybe

    def __init__(self, compressor=JaggedCompressorByZLIB, dtype=None, shape=None, order=None):
        super(JaggedCompressorWithDiff, self).__init__(dtype=dtype, shape=shape, order=order)
        self.compressor = compressor(dtype=dtype, shape=shape, order=order)

    def compress(self, x):
        xdiff = np.empty_like(x)
        xdiff[0] = x[0]
        xdiff[1:] = np.diff(x, axis=0)
        return self.compressor.compress(xdiff)

    def _compress_hook(self, x):
        raise NotImplementedError()

    def decompress(self, cx, dest=None):
        x = self.compressor.decompress(cx, dest=dest)
        x = np.require(x, requirements=['w'])
        x[1:] = np.cumsum(x[1:], axis=0) + x[0]
        return x

    def _decompress_hook(self, cx, dest=None):
        raise NotImplementedError()


# --- Utils

def cratio(x, compressor, check_roundtrip=True):
    """Computes the compressio ratio achieved by the compressor on the array x."""
    uncompressed = x
    compressed = compressor.compress(uncompressed)
    if check_roundtrip:
        np.testing.assert_array_almost_equal(x, compressor.decompress(compressed))
    return x.nbytes / len(compressed)


# --- Entry point


if __name__ == '__main__':

    nr = 10000
    compressibility = 1
    with_noise = True

    if compressibility == 0:  # this should not be very compressible, in any case...
        x = np.random.uniform(size=nr)
    elif compressibility == 1:  # floats are not very compressible with these tools, unless we precondition...
                                # a simple linear regression would do though
                                # and then we could upper-bound by precision spec and compress the residuals...
        x = np.linspace(0, nr, nr, dtype=np.float32)
    else:  # this should compress good, specially with preconditioning
        x = np.arange(nr)
    if with_noise:
        x += np.random.RandomState(0).randn(nr)

    compressor = JaggedCompressorByBlosc(shuffle=False, bitshuffle=True, cname='lz4hc', clevel=5)
    compressed = compressor.compress(x)
    decompressed = compressor.decompress(compressed)
    print('Compression ratio: %.2f' % cratio(x, compressor, check_roundtrip=True))

    compressor = JaggedCompressorByBZ2()
    compressed = compressor.compress(x)
    decompressed = compressor.decompress(compressed)
    print('Compression ratio: %.2f' % cratio(x, compressor, check_roundtrip=True))

# --- Random thoughts

#
# Use also bcolz or bloscpack?
# Something chunked could add a lot of overhead for small arrays.
# Unless we also chunk the arrays.
#
# LZ4 and zstd and blosc
#   https://code.google.com/p/lz4/
#
# Dependency installation:
#   conda install -c https://conda.binstar.org/flyem lz4
#   https://github.com/sergey-dryabzhinsky/python-zstd
#   conda install blosc
# Actually it is better to install latest releases python-blosc + c-blosc
# (not pip -e friendly, btw)
#
# Eventually try the bitshuffle filter:
#  http://www.blosc.org/blog/new-bitshuffle-filter.html
#  http://nbviewer.ipython.org/gist/alimanfoo/e93a532eb6bde311ea39/genotype_bitshuffle.ipynb
# First try, it segfaulted. Remember that my only machine with AVX2 is strall.
#

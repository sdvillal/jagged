# coding=utf-8
"""Compress and decompress collections of same-type, different-length arrays."""
from __future__ import division
from contextlib import contextmanager
from functools import partial
from timeit import timeit

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

try:
    import bitshuffle
except ImportError:  # pragma: no cover
    bitshuffle = None


# --- Jagged compressors API


@whatable(add_properties=False)
class JaggedCompressor(object):
    """
    Common unifying API over python compressors to compress and decompress numpy arrays.

    An instance of this class can only manage arrays of given by dtype, shape and order.
    The array type is set either using the constructor or on the first call to `compress`.
    Then all the compressed arrays must adhere to the specified type, except for the leading,
    dimension length that can change. That is, compressing arrays of different length
    is allowed, any other change will lead to an exception to be raised.

    Parameters
    ----------
    dtype : numpy dtype
    shape : shape of the arrays
    order : 'C' for row major, 'F' for column major
    """

    def __init__(self, dtype=None, shape=None, order=None):
        super(JaggedCompressor, self).__init__()
        if not self.is_available():
            raise ImportError('missing dependencies for %s, please install %s' %
                              (self.__class__.__name__, ','.join(self.dependencies())))
        self._dtype = dtype
        self._shape = shape
        self._order = order

    # --- Compressed arrays type

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        """Raises a ValueError exception if `shape` is not consistent with the compressor shape."""
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
        """Raises a ValueError exception if `dtype` is not consistent with the compressor dtype."""
        if self._dtype is None:
            self._dtype = dtype
        if self._dtype != dtype:
            raise ValueError('Cannot change dtype from %r to %r' % (self.dtype, dtype))

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        """Raises a ValueError exception if `order` is not consistent with the compressor order."""
        if self._order is None:
            self._order = order
        if self._order != order:
            raise ValueError('Cannot change order from %r to %r' % (self.order, order))

    # --- Compression

    def compress(self, x):
        """Compresses numpy array `x` returning a string with its compressed buffer contents.
        If the array type of the compressor is already set, it checks that `x` complies.
        If not, it sets the array type of the compressor.
        """
        order = 'F' if np.isfortran(x) else 'C'
        self.shape, self.dtype, self.order = x.shape, x.dtype, order
        # make data contiguous in memory; beware, can make a copy, revisit...
        x = np.require(x, requirements=[order])
        return self._compress_hook(x)

    def _compress_hook(self, x):
        """Does the actual compression of `x` data, returning the compressed string."""
        pass

    # --- Decompression

    def decompress(self, cx, dest=None):
        """
        Decompress bytes `cx` into an array, possibly storing the result in `dest`.

        Parameters
        ----------
        cx : str/bytes like
          The compressed array

        dest : numpy array
          The destination memory; it must be of the same type as the arrays in the compressor,
          and have enough allocated memory.
        """
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
        """Decompresses bytes like `cx` into a buffer that can be used to instantiate an array.
        If `dest` is specified, concrete implementations might use it to avoid extra memcopy's.
        """
        pass

    def uncompress(self, cx, dest=None):  # pragma: no cover
        """`decompress` alias"""
        return self.decompress(cx, dest=dest)

    # --- Dependency checks

    @staticmethod
    def is_available():
        """Returns True iff all the dependencies needed for this compressor are available."""
        return True

    @staticmethod
    def dependencies():
        """Returns a string tuple with the names of all non-standard-library dependencies of the compressor."""
        return ()


# --- Dummy

class JaggedCompressorByDummy(JaggedCompressor):
    """The "do not compress" compressor."""

    def _compress_hook(self, x):
        return np.getbuffer(x)

    def _decompress_hook(self, cx, dest=None):
        return cx


# --- Blosc

@contextmanager
def blosc_nthreads(n_threads):
    """Context manager for restablishing blosc threading settings, variant 1."""
    if n_threads is None:
        yield
    else:
        old_nthreads = blosc.set_nthreads(n_threads)
        yield
        if old_nthreads != n_threads:
            blosc.set_nthreads(old_nthreads)


class BloscNthreads(object):
    """Context manager for restablishing blosc threading settings, variant 2."""

    def __init__(self, n_threads):
        super(BloscNthreads, self).__init__()
        self.n_threads = n_threads
        self.old_nthreads = None

    def __enter__(self):
        if self.n_threads is not None:
            self.old_nthreads = blosc.set_nthreads(self.n_threads)

    def __exit__(self, *_):
        if self.old_nthreads is not None:
            blosc.set_nthreads(self.old_nthreads)


class JaggedCompressorByBlosc(JaggedCompressor):
    """
    A simplistic wrapper over blosc to compress and decompress numpy arrays.

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
      It is faster if the processor supports AVX2 instructions

    n_threads : int, default None
      The number of threads that blosc can use.
      Use None to let the global value for blosc_n_threads be used.
      See also `blosc_nthreads` and `BloscNThreads` context managers.
    """

    def __init__(self, cname='lz4hc', clevel=5, shuffle=True, bitshuffle=False, n_threads=None,
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
        with BloscNthreads(self.n_threads):
            return blosc.compress_ptr(x.__array_interface__['data'][0],
                                      x.size, x.dtype.itemsize,
                                      shuffle=self._filter, cname=self.cname, clevel=self.clevel)

    def _decompress_hook(self, cx, dest=None):

        # N.B. try-finally much faster than context manager for thousands of calls
        # For people that like peanuts

        old_nthreads = blosc.set_nthreads(self.n_threads) if self.n_threads is not None else None
        try:
            if dest is None:
                return blosc.decompress(cx)
            else:
                blosc.decompress_ptr(cx, dest.__array_interface__['data'][0])
                return dest
        finally:
            if old_nthreads is not None:
                blosc.set_nthreads(old_nthreads)

    @staticmethod
    def is_available():
        return blosc is not None

    @staticmethod
    def dependencies():
        return 'python-blosc',


# --- zlib

class JaggedCompressorByZLIB(JaggedCompressor):
    """Jagged compressor using zlib.

    See:
     http://www.zlib.net/

    Parameters
    ----------
    clevel : int between 0 and 9
      Sets the tradeoff between compression ratio and speed.
    """

    def __init__(self, clevel=5, dtype=None, shape=None, order=None):
        super(JaggedCompressorByZLIB, self).__init__(dtype=dtype, shape=shape, order=order)
        self.clevel = clevel

    def _compress_hook(self, x):
        return zlib.compress(np.getbuffer(x), self.clevel)

    def _decompress_hook(self, cx, dest=None):
        return zlib.decompress(cx)


# --- bzip2

class JaggedCompressorByBZ2(JaggedCompressor):
    """Jagged compressor using bz2.

    See:
     http://www.bzip.org/

    Parameters
    ----------
    clevel : int between 0 and 9
      Sets the tradeoff between compression ratio and speed.
    """

    def __init__(self, clevel=5, dtype=None, shape=None, order=None):
        super(JaggedCompressorByBZ2, self).__init__(dtype=dtype, shape=shape, order=order)
        self.clevel = clevel

    def _compress_hook(self, x):
        return bz2.compress(np.getbuffer(x), compresslevel=self.clevel)

    def _decompress_hook(self, cx, dest=None):
        return bz2.decompress(cx)


# --- LZ4

class JaggedCompressorByLZ4(JaggedCompressor):
    """Jagged compressor using lz4.

    See:
      http://www.lz4.org/

    Parameters
    ----------
    hc : boolean, default True
      Use the high-compression setting of LZ4. Recommended if compression speed is
      less important than size and decompression speed.
    """

    def __init__(self, hc=True, dtype=None, shape=None, order=None):
        super(JaggedCompressorByLZ4, self).__init__(dtype=dtype, shape=shape, order=order)
        self.hc = hc

    def _compress_hook(self, x):
        return lz4.compressHC(np.getbuffer(x)) if self.hc else lz4.compress(np.getbuffer(x))

    def _decompress_hook(self, cx, dest=None):
        return lz4.uncompress(cx)

    @staticmethod
    def is_available():
        return lz4 is not None

    @staticmethod
    def dependencies():
        return 'python-lz4',


# --- zstd

class JaggedCompressorByZSTD(JaggedCompressor):
    """Jagged compressor using zstd.

    See:
     https://github.com/Cyan4973/zstd
    """

    def _compress_hook(self, x):
        return zstd.compress(x)

    def _decompress_hook(self, cx, dest=None):
        return zstd.decompress(cx)

    @staticmethod
    def is_available():
        return zstd is not None

    @staticmethod
    def dependencies():
        return 'python-zstd',

#
# --- Scale-offset preconditioning
# This is quite simple, see:
#   ftp://www.hdfgroup.uiuc.edu/pub/outgoing/ymuqun/reports/
#   scaleoffset%20filters/scaleoffset_filter_documentation-kent-2005-7-21pdf.pdf
# TODO
#

# --- Dtype (e.g. downcast to single precision)


# --- Diff preconditioning

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

    def decompress(self, cx, dest=None):
        x = self.compressor.decompress(cx, dest=dest)
        x = np.require(x, requirements=['w'])
        x[1:] = np.cumsum(x[1:], axis=0) + x[0]
        return x


# --- Bitshuffle preconditioning

class JaggedCompressorWithBitshuffle(JaggedCompressor):

    def __init__(self, compressor=JaggedCompressorByZLIB, dtype=None, shape=None, order=None):
        super(JaggedCompressorWithBitshuffle, self).__init__(dtype=dtype, shape=shape, order=order)
        self.compressor = compressor(dtype=dtype, shape=shape, order=order)

    def compress(self, x):
        return self.compressor.compress(bitshuffle.bitshuffle(x))

    def decompress(self, cx, dest=None):
        x = bitshuffle.bitunshuffle(self.compressor.decompress(cx, dest=dest))
        if dest is None:
            return x
        dest[:] = x
        return dest

    @staticmethod
    def using_SSE2():
        """Returns whether bitshuffle was compiled with SSE2 support."""
        return bitshuffle.using_SSE2()

    @staticmethod
    def using_AVX2():
        """Returns whether bitshuffle was compiled with AVX2 support.
        Still if the CPU does not support these instructions, they won't be used.
        """
        # FIXME: check this on strall, that does support AVX2
        return bitshuffle.using_AVX2()

    @staticmethod
    def is_available():
        return bitshuffle is not None

    @staticmethod
    def dependencies():
        return 'bitshuffle',


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
    dtype = np.float32

    if compressibility == 0:  # this should not be very compressible, in any case...
        x = np.random.uniform(size=nr).astype(dtype=dtype)
    elif compressibility == 1:  # floats are not very compressible with these tools, unless we precondition...
                                # a simple linear regression would do though
                                # and then we could upper-bound by precision spec and compress the residuals...
        x = np.linspace(0, nr, nr, dtype=dtype)
    else:  # this should compress good, specially with preconditioning
        x = np.arange(nr, dtype=dtype)
    if with_noise:
        x += np.random.RandomState(0).randn(nr).astype(dtype=dtype)

    compressors = (
        JaggedCompressorByBlosc(shuffle=False, bitshuffle=False, cname='lz4hc', clevel=5),
        JaggedCompressorByBlosc(shuffle=True, bitshuffle=False, cname='lz4hc', clevel=5),
        JaggedCompressorByBlosc(shuffle=False, bitshuffle=True, cname='lz4hc', clevel=5),
        JaggedCompressorByBlosc(shuffle=False, bitshuffle=True, cname='lz4hc', clevel=5),
        JaggedCompressorByBZ2(),
        JaggedCompressorWithBitshuffle(JaggedCompressorByBZ2),
        JaggedCompressorWithBitshuffle(JaggedCompressorByLZ4),
        JaggedCompressorWithBitshuffle(JaggedCompressorByZSTD),
        JaggedCompressorWithBitshuffle(JaggedCompressorByZLIB),
        JaggedCompressorWithDiff(partial(JaggedCompressorWithBitshuffle, compressor=JaggedCompressorByLZ4)),
    )

    for compressor in compressors:
        cr = cratio(x, compressor, check_roundtrip=True)
        c_time = timeit(lambda: compressor.compress(x), number=1000)
        compressed = compressor.compress(x)
        d_time = timeit(lambda: compressor.decompress(compressed), number=1000)
        print('%s: cr=%.2f, ct=%.2f, dt=%.2f' % (compressor.what().id(), cr, c_time, d_time))

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
# Missing delta filter, scale-offset (lossy) and the like
#

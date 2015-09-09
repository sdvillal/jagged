import blosc
import numpy as np
from jagged.compressors import JaggedCompressorByBlosc


def set_nthreads(n=1):
    # So that it is catched by yappi
    blosc.set_nthreads(n)

if __name__ == '__main__':
    nr = 1000
    ncalls = 400000
    dtype = np.float32
    x = np.linspace(0, nr, nr, dtype=dtype)
    x += np.random.RandomState(0).randn(nr).astype(dtype=dtype)
    # c = JaggedCompressorByBlosc(cname='blosclz', n_threads=12)
    c = JaggedCompressorByBlosc(cname='blosclz', n_threads=None)
    blosc.set_nthreads(1)  # Otherwise it is the number of cores
    # blosc.set_nthreads(12)  # Otherwise it is the number of cores
    for i in range(ncalls):
        c.decompress(c.compress(x))

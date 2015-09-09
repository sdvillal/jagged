# coding=utf-8
from __future__ import print_function, absolute_import

import numpy as np
import pytest

from jagged.compressors import JaggedCompressorByBlosc

compressors = [
    (JaggedCompressorByBlosc,
     "JaggedCompressorByBlosc(bitshuffle=False,clevel=5,cname='lz4hc',n_threads=1,shuffle=True)")
]


@pytest.fixture(params=compressors)
def compressor(request):
    pass


def test_compressor(compressor):
    X = np.array([[1, 2, 3], [4, 5, 6]], order='F')
    bc = ()
    Xr = bc.decompress(bc.compress(X))
    np.testing.assert_array_equal(X, Xr)
    np.isfortran(Xr)
    Xd = np.empty_like(X)
    Xdd = bc.decompress(bc.compress(X), dest=Xd)
    Xd is Xdd
    True
    np.testing.assert_array_equal(X, Xdd)
    np.isfortran(Xdd)
    True
    print(bc.what().id())

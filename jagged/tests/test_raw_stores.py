# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from functools import partial
from operator import itemgetter
import bcolz
import numpy as np
import pytest
from jagged.bcolz_backend import JaggedByCarray
from jagged.h5py_backend import JaggedByH5Py


@pytest.yield_fixture(params=[JaggedByCarray, JaggedByH5Py])
def jagged_raw(request, tmpdir):
    jr = request.param
    dest = tmpdir.join(jr().what().id()).ensure_dir()
    try:
        yield jr, str(dest)
    finally:
        dest.remove(ignore_errors=True)


def test_roundtrip(jagged_raw):
    jagged_raw, path = jagged_raw
    jagged_raw = partial(jagged_raw, path=path)
    rng = np.random.RandomState(0)
    sizes = range(0, 1000, 100)
    ncol = 10
    originals = [rng.rand(size, ncol) for size in sizes]

    # Write
    segments = []
    with jagged_raw(write=True) as jr:
        total = 0
        assert jr.dtype is None
        assert jr.shape is None
        assert jr.is_writing
        for original in originals:
            base, size = jr.append(original)
            assert base == total
            assert size == len(original)
            total += size
            assert len(jr) == total
            segments.append((base, size))
        assert jr.dtype == originals[0].dtype
        assert jr.shape == (sum(map(itemgetter(1), segments)), ncol)

    # Read
    def test_read(originals, segments):
        # test read, one by one
        with jagged_raw(write=False) as jr:
            for original, segment in zip(originals, segments):
                roundtripped = jr.get([segment])[0]
                assert np.allclose(roundtripped, original)

        # test read, in a batch
        with jagged_raw(write=False) as jr:
            for original, roundtripped in zip(originals, jr.get(segments)):
                assert np.allclose(roundtripped, original)

    # read all
    with jagged_raw(write=False) as jr:
        assert np.allclose(np.vstack(originals), jr.get())

    # read in insertion order
    test_read(originals, segments)

    # read in random order
    or_s = list(zip(originals, segments))
    rng.shuffle(or_s)
    originals, segments = zip(*or_s)
    test_read(originals, segments)


def test_whatid():
    # TODO: add this to the yield fixtures, with expectations...
    #       check pytest docs as it might be a preferred way of doing this
    assert "JaggedByCarray(chunklen=1000," \
           "cparams=cparams(clevel=3,cname='zlib',shuffle=False),expectedlen=None)" \
           == JaggedByCarray(chunklen=1000,
                             cparams=bcolz.cparams(clevel=3, cname='zlib', shuffle=False),
                             expectedlen=None).what().id()


def test_factory(jagged_raw):
    jagged_raw, path = jagged_raw
    # factory without parameters should give the same config as the constructor
    assert jagged_raw().what().id() == jagged_raw.factory()().what().id()

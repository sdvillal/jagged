# coding=utf-8
"""Tests the raw storers."""
from __future__ import print_function, absolute_import, unicode_literals
from functools import partial
from operator import itemgetter
import bcolz
from .fixtures import *


@pytest.mark.xfail(reason='Needs to be implemented')
def test_empty_read():
    raise NotImplementedError()


@pytest.mark.xfail(reason='Needs to be implemented')
def test_0ncol():
    raise NotImplementedError()


def test_roundtrip(jagged_raw, dataset, columns, contiguity):
    jagged_raw, path = jagged_raw
    jagged_raw = partial(jagged_raw, path=path)
    rng, originals, ncol = dataset
    columns = columns(ncol)

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

        if columns is not None:
            originals = [o[:, columns] for o in originals]

        # test read, one by one
        with jagged_raw(write=False) as jr:
            for original, segment in zip(originals, segments):
                roundtripped = jr.get([segment], contiguity=contiguity, columns=columns)[0]
                assert np.allclose(original, roundtripped)

        # test read, in a batch
        with jagged_raw(write=False) as jr:
            for original, roundtripped in zip(originals, jr.get(segments, contiguity=contiguity, columns=columns)):
                assert np.allclose(original, roundtripped)

    # read all
    # with jagged_raw(write=False) as jr:
    #     assert np.allclose(np.vstack(originals) if columns is None else np.vstack(originals)[:, columns],
    #                        jr.get(contiguity=contiguity, columns=columns)[0])

    # read in insertion order
    # test_read(originals, segments)

    # read in random order
    or_s = list(zip(originals, segments))
    rng.shuffle(or_s)
    originals, segments = zip(*or_s)
    test_read(originals, segments)


def test_whatid():
    # TODO: add this to the yield fixtures, with expectations...
    #       check pytest docs as it might be a preferred way of doing this
    assert "JaggedByCarray" \
           "(chunklen=1000,cparams=cparams(clevel=3,cname='zlib',shuffle=False),expectedlen=None)" \
           == JaggedByCarray(chunklen=1000,
                             cparams=bcolz.cparams(clevel=3, cname='zlib', shuffle=False),
                             expectedlen=None).what().id()


def test_factory(jagged_raw):
    jagged_raw, path = jagged_raw
    assert (jagged_raw().what().id() == jagged_raw.factory()().what().id(),
            'factory without parameters should give the same config as the constructor')


# we should really use hypothesis

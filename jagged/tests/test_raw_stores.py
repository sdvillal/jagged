# coding=utf-8
"""Tests the raw storers."""
from __future__ import print_function, absolute_import, unicode_literals
from functools import partial
from operator import itemgetter
import os.path as op
import bcolz
from .fixtures import *
from jagged.base import retrieve_contiguous
from jagged.misc import ensure_dir


@pytest.mark.xfail(reason='Needs to be implemented')
def test_empty_read():
    raise NotImplementedError()


@pytest.mark.xfail(reason='Needs to be implemented')
def test_0ncol():
    raise NotImplementedError()


# -- lifecycle tests

def test_interleaved_appending_and_reading(jagged_raw):
    jagged_raw, path = jagged_raw
    data0 = np.zeros((2, 10))
    data1 = np.ones((3, 10))
    expected = np.vstack((data0, data1))
    with jagged_raw(path=path) as jr:
        # before writing, everything is unknown
        assert jr.shape is None
        assert jr.ndim is None
        assert jr.dtype is None
        # first write-up
        jr.append(data0)
        assert jr.shape == data0.shape
        assert jr.dtype == data0.dtype
        assert jr.ndim == data0.ndim
        # first read
        assert np.allclose(data0, jr.get()[0])
        # even if we close it...
        jr.close()
        # we can now know shapes and the like
        assert jr.shape == data0.shape
        assert jr.dtype == data0.dtype
        assert jr.ndim == data0.ndim
        # we can reread...
        assert np.allclose(data0, jr.get()[0])
        # we can know shapes and the like
        assert jr.shape == data0.shape
        assert jr.dtype == data0.dtype
        assert jr.ndim == data0.ndim
        # we can append more
        jr.append(data1)
        assert jr.shape == expected.shape
        assert jr.dtype == expected.dtype
        assert jr.ndim == expected.ndim
        # and the data will be properlly appended
        assert np.allclose(expected, jr.get()[0])


# -- Tests retrieve contiguous

def test_retrieve_contiguous(mock_jagged_raw, contiguity, columns):

    originals, ne, nc, dtype, segments, reader, rng = mock_jagged_raw

    if columns is not None:
        originals = [o[:, tuple(columns)] for o in originals]

    # sanity checks for wrong inputs
    with pytest.raises(ValueError) as excinfo:
        retrieve_contiguous(segments, columns, reader, dtype, ne, nc, 'wrong')
    assert 'Unknown contiguity scheme:' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        retrieve_contiguous([(-1, 1)], columns, reader, dtype, ne, nc, contiguity)
    assert 'Out of bounds query (base=-1, size=1' in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        retrieve_contiguous([(0, 100000)], columns, reader, dtype, ne, nc, contiguity)
    assert 'Out of bounds query (base=0, size=100000' in str(excinfo.value)

    # insertion order
    views = retrieve_contiguous(segments, columns, reader, dtype, ne, nc, contiguity)
    for o, v in zip(originals, views):
        assert np.allclose(o, v)

    # random order
    o_s = list(zip(originals, segments))
    rng.shuffle(o_s)
    originals, segments = zip(*o_s)
    views = retrieve_contiguous(segments, columns, reader, dtype, ne, nc, contiguity)
    for o, v in zip(originals, views):
        assert np.allclose(o, v)


# -- roundtrip tests

def test_roundtrip(jagged_raw, dataset, columns, contiguity):
    jagged_raw, path = jagged_raw
    jagged_raw = partial(jagged_raw, path=path)
    rng, originals, ncol = dataset

    # Write
    segments = []
    with jagged_raw() as jr:
        total = 0
        assert jr.dtype is None
        assert jr.shape is None
        for original in originals:
            base, size = jr.append(original)
            assert base == total
            assert size == len(original)
            assert jr.is_writing
            segments.append((base, size))
            total += size
            assert len(jr) == total
        assert jr.dtype == originals[0].dtype
        assert jr.shape == (sum(map(itemgetter(1), segments)), ncol)

    # Read
    def test_read(originals, segments):

        if columns is not None:
            originals = [o[:, columns] for o in originals]

        # test read, one by one
        with jagged_raw() as jr:
            for original, segment in zip(originals, segments):
                roundtripped = jr.get([segment], contiguity=contiguity, columns=columns)[0]
                assert np.allclose(original, roundtripped)

        # test read, in a batch
        with jagged_raw() as jr:
            for original, roundtripped in zip(originals, jr.get(segments, contiguity=contiguity, columns=columns)):
                assert np.allclose(original, roundtripped)

    # read all
    with jagged_raw() as jr:
        assert np.allclose(np.vstack(originals) if columns is None else np.vstack(originals)[:, columns],
                           jr.get(contiguity=contiguity, columns=columns)[0])

    # read in insertion order
    test_read(originals, segments)

    # read in random order
    or_s = list(zip(originals, segments))
    rng.shuffle(or_s)
    originals, segments = zip(*or_s)
    test_read(originals, segments)


# --- Test self-identification

def test_whatid():
    # TODO: add this to the yield fixtures, with expectations...
    #       check pytest docs as it might be a preferred way of doing this
    assert "JaggedByCarray" \
           "(chunklen=1000,cparams=cparams(clevel=3,cname='zlib',shuffle=False),expectedlen=None)" \
           == JaggedByCarray(chunklen=1000,
                             cparams=bcolz.cparams(clevel=3, cname='zlib', shuffle=False),
                             expectedlen=None).what().id()


# --- Test factories

def test_copyconf(jagged_raw):
    jagged_raw, path = jagged_raw
    assert jagged_raw().what().id() == jagged_raw().copyconf()().what().id(), \
        'factory without parameters should give the same config as the constructor'


# --- Misc tests

def test_nonvalid_appends(jagged_raw):
    jagged_raw, path = jagged_raw
    with jagged_raw(path=path) as jr:
        with pytest.raises(Exception) as excinfo:
            jr.append(np.zeros((10, 0)))
        assert 'Cannot append data with sizes 0 in non-leading dimension' in str(excinfo.value)


def test_no_inmemory_storage(jagged_raw):
    # maybe one day we allow these...
    jagged_raw, path = jagged_raw
    with jagged_raw(path=None) as jr:
        with pytest.raises(Exception) as excinfo:
            jr.append(np.zeros((1, 1)))
        assert 'In-memory only arrays are not implemented' in str(excinfo.value)


def test_copy_from(jagged_raw):
    jagged_raw, path = jagged_raw
    path0 = ensure_dir(op.join(path, 'test0'))
    path1 = ensure_dir(op.join(path, 'test1'))
    with jagged_raw(path0) as jr0, jagged_raw(path1) as jr1:
        jr0.append(np.zeros((2, 10)))
        jr0.append(np.ones((3, 10)))
        jr1.append_from(jr0)
        assert np.allclose(jr0.get()[0], jr1.get()[0])


def test_chunked_copy_from(jagged_raw):
    jagged_raw, path = jagged_raw
    path0 = ensure_dir(op.join(path, 'test0'))
    path1 = ensure_dir(op.join(path, 'test1'))
    with jagged_raw(path0) as jr0, jagged_raw(path1) as jr1:
        jr0.append(np.zeros((2, 10)))
        jr0.append(np.ones((3, 10)))
        jr1.append_from(jr0, chunksize=2)
        assert np.allclose(jr0.get()[0], jr1.get()[0])

# We should really have a look at using hypothesis

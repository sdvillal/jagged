# coding=utf-8
"""Tests the raw storers."""
from __future__ import print_function, absolute_import, unicode_literals
import os.path as op
import bcolz
from .fixtures import *
from jagged.base import retrieve_contiguous
from jagged.misc import ensure_dir


# -- lifecycle tests

def test_lifecycle(jagged_raw):
    jagged_raw, path = jagged_raw
    data0 = np.zeros((2, 10))
    data1 = np.ones((3, 10))
    expected = np.vstack((data0, data1))
    with jagged_raw(path=path) as jr:
        # before writing, everything is unknown
        assert jr.shape is None
        assert jr.ndims is None
        assert jr.dtype is None
        # first write-up
        jr.append(data0)
        assert jr.shape == data0.shape
        assert jr.dtype == data0.dtype
        assert jr.ndims == data0.ndim
        assert jr.bases() == [(0, len(data0))]
        # first read
        assert np.allclose(data0, jr.get()[0])
        # even if we close it...
        jr.close()
        # we can now know shapes and the like
        assert jr.shape == data0.shape
        assert jr.dtype == data0.dtype
        assert jr.ndims == data0.ndim
        # we can reread...
        assert np.allclose(data0, jr.get()[0])
        # we can know shapes and the like
        assert jr.shape == data0.shape
        assert jr.dtype == data0.dtype
        assert jr.ndims == data0.ndim
        # we can append more
        jr.append(data1)
        assert jr.shape == expected.shape
        assert jr.dtype == expected.dtype
        assert jr.ndims == expected.ndim
        assert jr.bases() == [(0, len(data0)), (len(data0), len(data1))]
        # and the data will be properlly appended
        assert np.allclose(expected, jr.get()[0])


# -- Tests retrieve contiguous

def test_retrieve_contiguous(mock_jagged_raw, columns, contiguity):

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

def test_roundtrip(jagged_raw, dataset, columns):
    jagged_raw, path = jagged_raw
    jagged_raw = partial(jagged_raw, path=path)
    rng, originals, ncol = dataset

    # Write
    keys = []
    with jagged_raw() as jr:
        total = 0
        assert jr.dtype is None
        assert jr.shape is None
        for original in originals:
            key = jr.append(original)
            assert jr.is_writing
            keys.append(key)
            total += len(original)
            assert len(jr) == total
        assert jr.dtype == originals[0].dtype
        assert jr.shape == (total, ncol)

    # Read
    def test_read(originals, keys):

        if columns is not None:
            originals = [o[:, columns] for o in originals]

        # test read, one by one
        with jagged_raw() as jr:
            for original, key in zip(originals, keys):
                roundtripped = jr.get([key], columns=columns)[0]
                assert np.allclose(original, roundtripped)

        # test read, in a batch
        with jagged_raw() as jr:
            for original, roundtripped in zip(originals, jr.get(keys, columns=columns)):
                assert np.allclose(original, roundtripped)

    # read all
    with jagged_raw() as jr:
        assert np.allclose(np.vstack(originals) if columns is None else np.vstack(originals)[:, columns],
                           jr.get(columns=columns)[0])

    # read in insertion order
    test_read(originals, keys)

    # read in random order
    or_s = list(zip(originals, keys))
    rng.shuffle(or_s)
    originals, keys = zip(*or_s)
    test_read(originals, keys)


# --- Test self-identification

def test_whatid():
    assert "JaggedByCarray(chunklen=1000," \
           "contiguity=None," \
           "cparams=cparams(clevel=3,cname='zlib',shuffle=False)," \
           "expectedlen=None)" \
           == JaggedByCarray(chunklen=1000,
                             cparams=bcolz.cparams(clevel=3, cname='zlib', shuffle=False),
                             expectedlen=None).what().id()
    assert "JaggedByH5Py(checksum=False," \
           "chunklen=1000," \
           "compression='lzf'," \
           "compression_opts=0," \
           "contiguity=None," \
           "shuffle=True)" \
           == JaggedByH5Py(chunklen=1000,
                           compression='lzf',
                           compression_opts=0,
                           shuffle=True).what().id()


# --- Test factories / curries/ partials

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
        for _ in range(10):
            jr0.append(np.zeros((2, 10)))
            jr0.append(np.ones((3, 10)))
        jr1.append_from(jr0, chunksize=2)
        assert np.allclose(jr0.get()[0], jr1.get()[0])
        with pytest.raises(ValueError) as excinfo:
            jr1.append_from(jr0, chunksize=-1)
        assert 'chunksize must be None or bigger than 0, it is -1' in str(excinfo.value)


def test_mmap_check_sizes(tmpdir):
    dest = str(tmpdir)
    x = np.empty((5, 2), dtype=np.int32)
    with JaggedByMemMap(dest) as jbm:
        jbm.append(x)
        mmf = jbm._mmpath
    # write row-sized junk
    with open(mmf, 'a') as writer:
        writer.write('junk' * 10)
    with JaggedByMemMap(dest) as jbm:
        with pytest.raises(Exception) as excinfo:
            jbm.get([(0, 2)])
        assert 'the number or rows inferred by file size does not coincide' in str(excinfo.value)
    # write junk that look like leftovers of an aborted write
    with open(mmf, 'a') as writer:
        writer.write('jagged')
    with JaggedByMemMap(dest) as jbm:
        with pytest.raises(Exception) as excinfo:
            jbm.get([(0, 2)])
        assert 'the memmap file has incomplete data' in str(excinfo.value)
    # make the memmap way too small
    with open(mmf, 'w') as writer:
        writer.write('jagged')
    with pytest.raises(Exception) as excinfo:
        jbm.get([(0, 2)])
    assert 'mmap length is greater than file size' in str(excinfo.value)

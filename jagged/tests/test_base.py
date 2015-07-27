# coding=utf-8
from jagged.base import retrieve_contiguous
from .fixtures import *


@pytest.fixture
def mock(dataset):

    # unbox the fixture
    rng, originals, ncol = dataset

    # reader
    jagged = np.vstack(originals)

    def reader(base, size, columns, dest):
        if columns is None:
            dest[:] = jagged[base:(base+size)]
        else:
            dest[:] = jagged[base:(base+size), tuple(columns)]

    # shape
    ne, nc = jagged.shape

    # segments
    base = 0
    segments = []
    for o in originals:
        segments.append((base, len(o)))
        base += len(o)

    return originals, ne, nc, originals[0].dtype, segments, reader, rng


def test_retrieve_contiguous(mock, contiguity, columns):

    originals, ne, nc, dtype, segments, reader, rng = mock

    columns = columns(nc)
    if columns is not None:
        originals = [o[:, tuple(columns)] for o in originals]

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

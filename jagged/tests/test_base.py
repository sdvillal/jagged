# coding=utf-8
from operator import itemgetter
from jagged.base import retrieve_contiguous
import pytest
import numpy as np
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


def test_retrieve_contiguous(mock, contiguity):

    originals, ne, nc, dtype, segments, reader, rng = mock

    # insertion order
    views = retrieve_contiguous(segments, None, reader, dtype, ne, nc, 'read')
    for o, v in zip(originals, views):
        assert np.allclose(o, v)

    # random order
    o_s = zip(originals, segments)
    rng.shuffle(o_s)
    originals, segments = zip(*o_s)

    views = retrieve_contiguous(segments, None, reader, dtype, ne, nc, contiguity)
    for o, v in zip(originals, views):
        assert np.allclose(o, v)


if __name__ == '__main__':
    pytest.main()

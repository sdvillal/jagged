# coding=utf-8
from .fixtures import *


def test_index(index):
    index, path = index
    with index(path=path) as idx:
        idx.add((0, 100), 'key1')
        assert idx.segments() == [(0, 100)]
        assert idx.keys() == {'key1': 0}

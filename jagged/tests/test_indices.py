# coding=utf-8
from __future__ import absolute_import, print_function, unicode_literals
from .fixtures import *


def test_index(index):
    index, path = index
    with index(path=path) as idx:
        idx.add((0, 100), 'key1')
        idx.add((33, 12), 'key2')
        idx.add((1000, 42))
        assert idx.segments() == [(0, 100), (33, 12), (1000, 42)]
        assert idx.keys() == {'key1': 0, 'key2': 1}
        assert idx.segment(0) == (0, 100)
        assert idx.segment(1) == (33, 12)
        assert idx.num_keys() == 2
        assert idx.num_segments() == 3
        assert not idx.can_add('key1')
        assert idx.can_add('key_not_in_there')
        with pytest.raises(KeyError) as excinfo:
            assert idx.add((0, 1), 'key1')
        assert 'Cannot insert key' in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            assert idx.add('not_a_segment', 'key1')
        assert 'not a valid segment specification' in str(excinfo.value)
        segments = idx.get(('key1', 'key2', 'key1'))
        assert segments == [(0, 100), (33, 12), (0, 100)]
        sss = idx.subsegments('key2', (3, 3), (5, 6))
        assert sss == [(36, 3), (38, 6)]
        sss = idx.subsegments(1, (3, 3), (5, 6))
        assert sss == [(36, 3), (38, 6)]

    # open, add again...
    with index(path=path) as idx:
        idx.add((57, 123), 'oddkey')
        assert idx.segments() == [(0, 100), (33, 12), (1000, 42), (57, 123)]
        assert idx.keys() == {'key1': 0, 'key2': 1, 'oddkey': 3}
        assert idx.segment(0) == (0, 100)
        assert idx.segment(1) == (33, 12)
        assert idx.num_keys() == 3
        assert idx.num_segments() == 4
        assert not idx.can_add('oddkey')
        assert idx.can_add('key_not_in_there')
        with pytest.raises(KeyError) as excinfo:
            assert idx.add((0, 1), 'key1')
        assert 'Cannot insert key' in str(excinfo.value)
        segments = idx.get(('key1', 'key2', 'key1', 'oddkey'))
        assert segments == [(0, 100), (33, 12), (0, 100), (57, 123)]
        sss = idx.subsegments('key2', (3, 3), (5, 6))
        assert sss == [(36, 3), (38, 6)]
        sss = idx.subsegments(1, (3, 3), (5, 6))
        assert sss == [(36, 3), (38, 6)]

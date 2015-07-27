# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from jagged.tests.fixtures import *


def test_store(store):
    store, path = store
    assert store is not None
    with store(path) as js:
        js.add(np.zeros((100, 10)), 'mola')
        assert np.allclose(np.zeros((100, 10)), js.get(['mola'])[0])

# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals

import numpy as np
import pytest

from jagged.bcolz_backend import JaggedByCarray
from jagged.h5py_backend import JaggedByH5Py


@pytest.yield_fixture(params=[JaggedByCarray, JaggedByH5Py],
                      ids=['jr=carray', 'jr=h5py'])
def jagged_raw(request, tmpdir):
    jr = request.param
    dest = tmpdir.join(jr().what().id()).ensure_dir()
    try:
        yield jr, str(dest)
    finally:
        dest.remove(ignore_errors=True)


@pytest.fixture(params=(1, 2, 10),
                ids=('ncol=1', 'ncol=2', 'ncol=10'))
def ncol(request):
    return request.param


@pytest.fixture(params=(0, 1), ids=['rng=0', 'rng=1'])
def rng(request):
    return np.random.RandomState(request.param)


@pytest.fixture
def dataset(ncol, rng):
    sizes = [0] + rng.randint(low=0, high=500, size=10).tolist()
    rng.shuffle(sizes)
    originals = [rng.rand(size, ncol) for size in sizes]
    return rng, originals, ncol


@pytest.fixture(params=('read', 'write', None),
                ids=('cont=read', 'cont=write', 'cont=none'))
def contiguity(request):
    return request.param


@pytest.fixture(params=(lambda nc: None,  # all
                        lambda nc: range(nc),  # all, explicit
                        lambda nc: [0],  # first
                        lambda nc: [nc - 1],  # last (we should support python -col syntax)
                        lambda nc: range(0, nc, 2),  # even
                        lambda nc: range(nc)[::-1],  # inverse
                        lambda nc: list(range(nc)[::-1]) + list(range(0, nc, 2))  # mixed
                        ),
                ids=('cols=all', 'cols=all-exp', 'cols=first', 'cols=last', 'cols=even', 'cols=inverse', 'cols=mixed'))
def columns(request):
    return request.param
# FIXME: probably this could take ncol and not return a function

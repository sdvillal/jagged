# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals

import numpy as np
import pytest

from functools import partial

from jagged.bcolz_backend import JaggedByCarray
from jagged.blosc_backend import JaggedByBlosc
from jagged.h5py_backend import JaggedByH5Py
from jagged.joblib_backend import JaggedByJoblib
from jagged.mmap_backend import JaggedByMemMap
from jagged.npy_backend import JaggedByNPY
from jagged.bloscpack_backend import JaggedByBloscpack
from jagged.pickle_backend import JaggedByPickle

RAW_STORES = []

LINEAR_RAW_STORES = (
    ('jr=carray', JaggedByCarray),
    ('jr=carraychunks', partial(JaggedByCarray, chunklen=100)),
    ('jr=h5py', JaggedByH5Py),
    ('jr=h5pychunks', partial(JaggedByH5Py, chunklen=100)),
    ('jr=mmap', JaggedByMemMap),
)

for contiguity in ('read', 'write', None, 'auto'):
    for name, store in LINEAR_RAW_STORES:
        RAW_STORES.append((name + '#' + 'cont=%s' % contiguity, partial(store, contiguity=contiguity)))

RAW_STORES.extend([
    ('jr=npy', JaggedByNPY),
    ('jr=blp', JaggedByBloscpack),
    ('jr=blosc-mm', JaggedByBlosc),
    ('jr=pickle', JaggedByPickle),
    ('jr=joblib', JaggedByJoblib),
])

stores = [store for _, store in RAW_STORES]
names = [name for name, _ in RAW_STORES]


@pytest.yield_fixture(params=stores, ids=names)
def jagged_raw(request, tmpdir):
    jr = request.param
    dest = tmpdir.join(jr().what().id()).ensure_dir()
    try:
        yield jr, str(dest)
    finally:
        dest.remove(ignore_errors=True)


@pytest.fixture(params=(1, 2, 10), ids=('ncol=1', 'ncol=2', 'ncol=10'))
def ncol(request):
    return request.param


@pytest.fixture(params=('cols=all', 'cols=all-exp', 'cols=first', 'cols=last',
                        'cols=even', 'cols=inverse', 'cols=mixed'),
                ids=('cols=all', 'cols=all-exp', 'cols=first', 'cols=last',
                     'cols=even', 'cols=inverse', 'cols=mixed'))
def columns(request, ncol):
    if request.param == 'cols=all':
        return None
    elif request.param == 'cols=all-exp':
        return range(ncol)
    elif request.param == 'cols=first':
        return [0]
    elif request.param == 'cols=last':
        return [ncol - 1]  # (we should support python negative indexing syntax)
    elif request.param == 'cols=even':
        return list(range(0, ncol, 2))
    elif request.param == 'cols=inverse':
        return list(range(ncol)[::-1])
    elif request.param == 'cols=mixed':
        list(range(ncol)[::-1]) + list(range(0, ncol, 2))
    else:  # pragma: no cover
        raise ValueError('Unknows column spec %r' % request.param)


@pytest.fixture(params=(0, 1), ids=['rng=0', 'rng=1'])
def rng(request):
    return np.random.RandomState(request.param)


@pytest.fixture
def dataset(ncol, rng):
    sizes = [0] + rng.randint(low=0, high=500, size=10).tolist()
    rng.shuffle(sizes)
    originals = [rng.rand(size, ncol) for size in sizes]
    return rng, originals, ncol


@pytest.fixture
def mock_jagged_raw(dataset):

    # unbox the fixture
    rng, originals, ncol = dataset

    # reader
    jagged = np.vstack(originals)

    def reader(base, size, columns, dest):
        view = jagged[base:(base+size)]
        if columns is not None:
            view = view[:, tuple(columns)]
        if dest is None:
            return view
        dest[:] = view
        return dest

    # shape
    ne, nc = jagged.shape

    # segments
    base = 0
    segments = []
    for o in originals:
        segments.append((base, len(o)))
        base += len(o)

    return originals, ne, nc, originals[0].dtype, segments, reader, rng


@pytest.fixture(params=('read', 'write', None, 'auto'),
                ids=('cont=read', 'cont=write', 'cont=none', 'cont=auto'))
def contiguity(request):
    return request.param

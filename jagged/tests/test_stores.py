# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from operator import itemgetter
from jagged.tests.fixtures import *


@pytest.fixture
def keyed_dataset(rng, ncol):
    return [('key%d' % i, rng.uniform(size=(size, ncol)))
            for i, size in enumerate([0] + rng.randint(0, 50, 50).tolist())]


def test_store(store, keyed_dataset):
    store, path = store
    dset = keyed_dataset
    dset0 = dset[:int(len(dset)/2)]
    dset0stacked = np.vstack(map(itemgetter(1), dset0))
    dset1 = dset[int(len(dset)/2):]
    dset1stacked = np.vstack(map(itemgetter(1), dset1))
    dsetstacked = np.vstack((dset0stacked, dset1stacked))
    colnames = ['col%d' % col for col in range(dsetstacked.shape[1])]
    units = ['m/s²'] * dsetstacked.shape[1]
    with store(path) as js:
        # insert, get, check that we cannot insert again
        for k, data in dset0:
            js.add(data, key=k)
            assert np.allclose(data, js.get([k])[0])
            with pytest.raises(KeyError) as excinfo:
                js.add(data, k)
            assert 'Cannot add key' in str(excinfo.value)
        # check that batch retrieval works
        for original, got in zip(js.get([k for k, _ in dset0]), [d for _, d in dset0]):
            assert np.allclose(original, got)
        for original, got in zip(js.get(), [d for _, d in dset0]):
            assert np.allclose(original, got)
        # check that iter works
        for original, got in zip(js.iter(), [d for _, d in dset0]):
            assert np.allclose(original, got)
        for original, got in zip(js.iter([k for k, _ in dset0]), [d for _, d in dset0]):
            assert np.allclose(original, got)
        for original, got in zip(js.iter([k for k, _ in dset0][::2]), [d for _, d in dset0][::2]):
            assert np.allclose(original, got)
        # check shapes and the like
        assert js.shape == dset0stacked.shape
        assert js.ndim == dset0stacked.ndim
        assert js.dtype == dset0stacked.dtype
        assert len(js) == len(dset0stacked)
        # check meta-info
        js.set_colnames(colnames)
        js.set_units(units)
        assert js.colnames() == colnames
        assert js.units() == units

    # "close" and reopen
    with store(path) as js:
        # check that batch retrieval works
        for original, got in zip(js.get([k for k, _ in dset0]), [d for _, d in dset0]):
            assert np.allclose(original, got)
        # check that batch retrieval works
        for original, got in zip(js.get(), [d for _, d in dset0]):
            assert np.allclose(original, got)
        # check shapes and the like
        assert js.shape == dset0stacked.shape
        assert js.ndim == dset0stacked.ndim
        assert js.dtype == dset0stacked.dtype
        assert len(js) == len(dset0stacked)
        # check meta-info
        assert js.colnames() == colnames
        assert js.units() == units

    # "close", write more, read
    with store(path) as js:
        # insert, get, check that we cannot insert again
        for k, data in dset1:
            js.add(data, key=k)
            assert np.allclose(data, js.get([k])[0])
            with pytest.raises(KeyError) as excinfo:
                js.add(data, k)
            assert 'Cannot add key' in str(excinfo.value)
        # check that batch retrieval works
        for original, got in zip(js.get([k for k, _ in dset]), [d for _, d in dset]):
            assert np.allclose(original, got)
        # check that batch retrieval works
        for original, got in zip(js.get(), [d for _, d in dset]):
            assert np.allclose(original, got)
        # check shapes and the like
        assert js.shape == dsetstacked.shape
        assert js.ndim == dsetstacked.ndim
        assert js.dtype == dsetstacked.dtype
        assert len(js) == len(dsetstacked)
        # check meta-info
        assert js.colnames() == colnames
        assert js.units() == units
    # we miss things like:
    #   columns=
    #   factory=

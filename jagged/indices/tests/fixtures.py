# coding=utf-8
from jagged.indices.indices import JaggedSimpleIndex, JaggedStore
import pytest


@pytest.yield_fixture(params=(JaggedSimpleIndex,), ids=('idx=simple',))
def index(request, tmpdir):
    idx = request.param
    dest = tmpdir.join(idx().what().id()).ensure_dir()
    try:
        yield idx, str(dest)
    finally:
        dest.remove(ignore_errors=True)


@pytest.yield_fixture(params=(JaggedStore,), ids=('store=simple',))
def store(request, tmpdir):
    store = request.param
    dest = tmpdir.join('store').ensure_dir()
    try:
        yield store, str(dest)
    finally:
        dest.remove(ignore_errors=True)

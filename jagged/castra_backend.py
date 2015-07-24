# coding=utf-8
from __future__ import absolute_import, unicode_literals, print_function
from jagged.base import JaggedRawStore


class JaggedByCastra(JaggedRawStore):
    def __init__(self, path=None, write=False,):
        super(JaggedByCastra, self).__init__()

#
# After playing around for a few minutes, Castra is not ready (or even gearing towards) this.
# Keep an eye on it for another nifty backend.
# Some ideas that we must move:
#   - Of course use always categoricals when/if we support something else than numeric types
#   - Use blosc/bloscpack should be easy; use msgpack for object serialisation is a good trick
#   - Columnar!
#

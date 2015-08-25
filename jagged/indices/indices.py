# coding=utf-8
"""Convenient (and somehow performing) storage of objects with homogeneous types but different lengths.

These three base abstractions allow a slightly pluggable architecture for `jagged`:
 - `JaggedRawStore`: stores elements, agnostic of any meaningful partition
 - `JaggedIndex`: tells how to split the data provided by a jagged raw store into meaningful segments
 - `JaggedStore`: puts together `JaggedRawStore` and `JaggedIndex` with a convenient, higher-level API

All these classes are `whatable`.

`jagged` data providers have very simple, low level contracts:
 - Focus on reading performance, append only store.
 - May or may not restrict the type of the stored elements
 - Retrieve only by providing indices *collections*
   No explicit support for slice notation
"""

from __future__ import absolute_import, unicode_literals, print_function
import os.path as op
from operator import itemgetter

from future.builtins import map

from jagged.misc import ensure_dir, subsegments, is_valid_segment
from whatami import whatable

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle


# --- Index stores

@whatable(add_properties=False)
class JaggedIndex(object):
    """Maps keys to index in a jagged store."""

    def keys(self):
        """Returns a dictionary mapping keys to array indices.
        May be smaller than the number of known segments.
        """
        raise NotImplementedError()

    def sorted_keys(self):
        """Returns a list of tuples (key, index) sorted by index.
        Usually this will correspond to "insertion order".
        """
        return sorted(self.keys().items(), key=itemgetter(1))

    def num_keys(self):
        """Returns the number of keys in the index."""
        return len(self.keys())

    def can_add(self, key):
        """Returns True iff the `key` can be added to the index.
        In the default implementation repeated keys are not allowed.
        """
        return key not in self.keys()

    def add(self, segment, key=None):
        """Adds a segment to the index, possibly linking it to a key."""
        # This default implementation assumes the index is all in memory using python lists and dicts
        # We should move one step ahead and use pandas-like indices (and in general be also pandas-friendly)
        if not is_valid_segment(segment):
            raise ValueError('%r is not a valid segment specification' % (segment,))
        if key is not None:
            if not self.can_add(key):
                raise KeyError('Cannot insert key %r' % key)
            self.segments().append(segment)
            self.keys()[key] = self.num_segments() - 1
        else:
            self.segments().append(segment)

    def get(self, keys):
        """Returns the list of segments associated with the keys."""
        return [self.segments()[self.keys()[k]] for k in keys]

    def close(self):
        """Flushes buffers to permanent storage and closes the underlying backend, if this is necessary."""
        raise NotImplementedError()

    def subsegments(self, segment_key, *subs):
        """Returns a list of subsegments relative to the segment pointed to by `segment_key` and `subs` specs."""
        return subsegments(self.segment(segment_key), *subs)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class JaggedSimpleIndex(JaggedIndex):
    """Simplemost implementation, index in-memory and persistence using pickle"""

    def __init__(self, path=None):
        self._keys = None
        self._segments = None
        self._path = path
        # populate indices
        self.segments()
        self.keys()

    def keys(self):
        if self._keys is None:
            try:
                with open(self._keys_file, 'rb') as reader:
                    self._keys = pickle.load(reader)
            except:  # FIX Too broad
                self._keys = {}
        return self._keys

    def close(self):
        if self._segments is not None:
            with open(self._segments_file, 'wb') as writer:
                pickle.dump(self._segments, writer, protocol=pickle.HIGHEST_PROTOCOL)
        if self._keys is not None:
            with open(self._keys_file, 'wb') as writer:
                pickle.dump(self._keys, writer, protocol=pickle.HIGHEST_PROTOCOL)



# --- Jagged stores


@whatable
class JaggedStore(object):

    def __init__(self,
                 path,
                 jagged_factory=None,
                 index_factory=None):

        super(JaggedStore, self).__init__()

        self._path = path

        # Raw storage
        if jagged_factory is None:
            from .bcolz_backend import JaggedByCarray
            jagged_factory = JaggedByCarray
        self._jagged_factory = jagged_factory
        self._jagged_root = ensure_dir(op.join(self._path,
                                               'raw',
                                               jagged_factory().what().id()))
        self._jagged = jagged_factory(path=self._jagged_root)

        # Meta-information
        self._meta = None
        self._meta_file = op.join(ensure_dir(op.join(self._path, 'meta')), 'meta.pkl')

        # Indices
        if index_factory is None:
            index_factory = JaggedSimpleIndex
        self._index_factory = index_factory
        self._indices = {}

    def index(self, name='main'):
        if name not in self._indices:
            self._indices[name] = self._index_factory(path=ensure_dir(op.join(self._path, 'indices', name)))
        return self._indices[name]

    def add(self, data, key=None):
        index = self.index()
        if index.can_add(key):
            base, length = self._jagged.append(data)
            index.add(segment=(base, length), key=key)
        else:
            raise KeyError('Cannot add key %r' % (key,))

    def get(self, keys=None, columns=None, factory=None, index='main'):
        index = self.index(index)
        if keys is None:
            keys = map(itemgetter(0), index.sorted_keys())
            # N.B. slow, since we already have the indices of the segments we could save us more searchs
        return self._jagged.get(index.get(keys), columns=columns, factory=factory)

    def iter(self, keys=None, columns=None, factory=None, index='main'):
        index = self.index(index)
        if keys is None:
            keys = map(itemgetter(0), index.sorted_keys())
        return (self._jagged.get((segment,), columns=columns, factory=factory)[0]
                for segment in index.get(keys))
        # we should also add a chunksize parameter and allow to retrieve by chunks
        # of course now chunks have a logical meaning (= number of segments)
        # so imagining memory consumption would now be a bit more indirect task

    # --- Meta information

    def meta(self):
        if self._meta is None:
            try:
                with open(self._meta_file, 'rb') as reader:
                    self._meta = pickle.load(reader)
            except:
                self._meta = {}
        return self._meta

    def add_meta(self, key, value):
        self.meta()[key] = value

    def get_meta(self, key):
        return self.meta().get(key, None)

    def set_colnames(self, colnames):
        self.add_meta('colnames', colnames)  # check consistency with dimensionality

    def colnames(self):
        return self.get_meta('colnames')

    def set_units(self, units):
        self.add_meta('units', units)

    def units(self):
        return self.get_meta('units')

    # --- Closing, context manager

    def close(self):
        if self._jagged is not None:
            self._jagged.close()
        for index in self._indices.values():
            index.close()
        self._indices = {}
        if self._meta is not None:
            with open(self._meta_file, 'wb') as writer:
                pickle.dump(self._meta, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        """Returns the size of the leading dimension."""
        return self.shape[0]

    @property
    def shape(self):
        return self._jagged.shape

    @property
    def ndim(self):
        return self._jagged.ndims

    @property
    def dtype(self):
        return self._jagged.dtype

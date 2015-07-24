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
 - Retrieve only by providing base + size *collections*
   Retrieve only by contiguous blocks (i.e. no explicit support for slice notation)
"""
from __future__ import absolute_import, unicode_literals, print_function
import os.path as op
from jagged.misc import ensure_dir
from whatami import whatable
try:
    import cPickle as pickle
except ImportError:
    import pickle


# --- Raw stores

@whatable(add_properties=False)
class JaggedRawStore(object):
    """Persistent storage of objects of the same type but different length."""

    def append(self, data):
        """Appends new data to this storage.

        If the storage is empty, this will define the dtype of the store.

        Parameters
        ----------
        data : numpy-array like
          The data to append, must have a compatible dtype with what was already added to the store.

        Returns
        -------
        A tuple (base, size) that addresses the appended data in the storage.
        """
        raise NotImplementedError()

    def get(self, segments, columns=None, factory=None):
        """Returns a list with the data specified in `segments` (and `columns`), possibly transformed by `factory`.

        Concrete implementations may warrant things like "all segments actually lie in congiguous regions in memory".

        Parameters
        ----------
        segments : list of tuples (base, size)
          specifies which elements to retrieve; if None, the hole data is returned

        columns : list of integers, default None
          specifies which columns to retrieve; if None, retrieve all columns

        factory : factory(ndarray)->desired type, default None
          transforms each of the returned elements into a desired type (for example, a pandas DataFrame)
          another use can be to apply summary statistics

        Returns
        -------
        A list with the retrieved elements, possibly transformed by factory.
        """
        raise NotImplementedError()

    def consolidate(self):
        """Perform post-append optimisations, possibly disabling writing."""
        return self

    def close(self):
        """Flushes buffers to permanent storage and closes the underlying backend."""
        raise NotImplementedError()

    def iterchunks(self, chunksize):
        """Reads `chunksize` elements at a time until all is read."""
        base = 0
        total = len(self)
        while base < total:
            size = min(chunksize, total - base)
            yield self.get([(base, size)])[0]

    def append_from(self, jagged, chunksize=None):
        """Appends all the contens of `jagged`."""
        if chunksize <= 0:
            self.append(jagged.get())
        else:
            for chunk in jagged.iterchunks(chunksize):
                self.append(chunk)

    @staticmethod
    def factory(**kwargs):
        """Returns a factory for a concrete configuration of this store.
        The factory is a method that should accept (also both Nones):
          - path: the address for the data of the store (usually a directory)
          - write: a boolean indicating if we are opening the store in read or write mode
        """
        raise NotImplementedError()

    @property
    def is_writing(self):
        """Returns whether we can append more data using this jagged instance."""
        raise NotImplementedError()

    @property
    def ndim(self):
        """Returns the number of dimensions."""
        return len(self.shape)

    @property
    def shape(self):
        """Returns a tuple with the current size of the storage in each dimension."""
        raise NotImplementedError()

    @property
    def dtype(self):
        """Returns the data type of the store."""
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        """Returns the size of the leading dimension."""
        return self.shape[0]

    # Also consider registry to atexit etc.


# --- Index stores

@whatable
class JaggedIndex(object):
    """Maps keys to segments that can address elements in `JaggedRawStore` instances.
    Segments can be addressed by key and insertion index.
    """

    def segments(self):
        """Returns the list of known segments.
        May be larger than the number of known keys.
        """
        raise NotImplementedError()

    def segment(self, i):
        """Returns the ith segment in the index."""
        return self.segments()[i]

    def num_segments(self):
        """Returns the number of segments in the index."""
        return len(self.segments())

    def keys(self):
        """Returns a dictionary mapping keys to segment indices.
        May be smaller than the number of known keys.
        """
        raise NotImplementedError()

    def num_keys(self):
        """Returns the number of keys in the index."""
        return len(self.keys())

    def can_add(self, key):
        """Returns True iff the `key` can be added to the index."""
        # This default implementation disallow repeated keys
        return key not in self.keys()

    def add(self, segment, key=None):
        """Adds a segment to the index, possibly linking it to a key."""
        # This default implementation assumes the index is all in memory using python lists and dicts
        # Maybe we should move one step ahead and use pandas-like indices
        if key is not None:
            if not self.can_add(key):
                raise Exception('Cannot insert key %r' % key)
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

    def subsegment(self, segment_id, segment_spec):
        # a tuple (TODO a bool array...)
        start, size = segment_spec
        base_start, base_size = self.segment(segment_id)
        if start < 0:
            raise Exception()
        if base_size < (start + size):
            raise Exception()
        return base_start + start, size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class JaggedSimpleIndex(JaggedIndex):
    """Simplemost implementation, index in-memory and persistence using pickle"""

    def __init__(self, root):
        self._keys = None
        self._segments = None
        self._root = ensure_dir(op.join(root, self.what().id()))
        self._keys_file = op.join(self._root, 'keys.pkl')
        self._segments_file = op.join(self._root, 'segments.pkl')

    def segments(self):
        if self._segments is None:
            try:
                with open(self._segments_file) as reader:
                    self._segments = pickle.load(reader)
            except:  # FIX Too broad
                self._segments = []
        return self._segments

    def keys(self):
        if self._keys is None:
            try:
                with open(self._keys_file) as reader:
                    self._keys = pickle.load(reader)
            except:  # FIX Too broad
                self._keys = {}
        return self._keys

    def close(self):
        if self._segments is not None:
            with open(self._segments_file, 'w') as writer:
                pickle.dump(self._segments, writer, protocol=pickle.HIGHEST_PROTOCOL)
        if self._keys is not None:
            with open(self._keys_file, 'w') as writer:
                pickle.dump(self._keys, writer, protocol=pickle.HIGHEST_PROTOCOL)


# --- Jagged stores


@whatable
class JaggedStore(object):

    #
    # Only one JaggedRawStore is allowed to be used.
    # This restriction is in place to free me from thinking
    # about consistency of data accross backends, and may be
    # removed in the future.
    #
    # Allow for multiple views on the segments; for example:
    #  - original trajectories
    #  - saccades
    #  - when the animal is behaving / collaborating / engaging
    #  - perturbations
    #  - where a filter flags a bad trajectory
    # This can be done by:
    #   - switching indices (e.g. an index for saccades, an index for perturbs...)
    #   - single-index, taking care that the keys in the index indicate well the type os
    #     (and allowing partial key matching, pandas amazing indices can be helpful here)
    # We must think of a mechanism to share indices accross different raw-jagged instances
    # Let's do when all this have been properly used in production
    #
    # How to share indices?
    #  - Instantiate a new JaggedStore, same root, different jagged_name.
    #  - Copy the jagged stuff without caring about keys
    #
    # Index <-> Jagged is actually a many-to-many relationship
    # What about .register(index, jagged) to make this explicit?
    #
    # TODO: the usual weakref to tame memory consumption
    #

    def __init__(self,
                 path,
                 jagged_name='main',
                 jagged_factory=None,
                 index_factory=None):

        super(JaggedStore, self).__init__()

        self._path = path

        # Raw storage
        if jagged_factory is None:
            from .bcolz_backend import JaggedByCarray
            jagged_factory = JaggedByCarray.factory()
        self._jagged_name = jagged_name
        self._jagged_factory = jagged_factory
        self._jagged_root = ensure_dir(op.join(self._path,
                                               'raw',
                                               jagged_name,
                                               jagged_factory().what().id()))
        self._jagged = None

        # Meta-information
        self._meta = None
        self._meta_file = op.join(ensure_dir(op.join(self._path, 'meta', jagged_name)), 'meta.pkl')

        # Indices
        if index_factory is None:
            index_factory = JaggedSimpleIndex
        self._index_factory = index_factory
        self._indices = {}

    def jagged(self, write=False):
        if self._jagged is None:
            self._jagged = self._jagged_factory(self._jagged_root, write=write)
        elif self._jagged.is_writing != write:
            self._jagged.close()
            self._jagged = self._jagged_factory(self._jagged_root, write=write)
        return self._jagged

    def index(self, name='main'):
        if name not in self._indices:
            self._indices['name'] = self._index_factory(path=ensure_dir(op.join(self._path, 'indices', name)))
        return self._indices[name]

    def add(self, data, key=None):
        index = self.index()
        if index.can_add(key):
            jagged = self.jagged(write=True)
            base, length = jagged.append(data)
            index.add(segment=(base, length), key=key)
        else:
            raise Exception('Cannot add key %r' % (key,))

    def get(self, keys=None, factory=None, index='main'):
        index = self.index(index)
        if keys is None:
            keys = index.keys()
        return self.jagged().get(index.get(keys), factory=factory)

    def iter(self, keys=None, factory=None, index='main'):
        index = self.index(index)
        if keys is None:
            return (self.jagged().get((segment,), factory=factory)[0]
                    for segment in index.segments())
        return (self.jagged().get((segment,), factory=factory)[0]
                for segment in index.get(keys))

    # --- Meta information

    def meta(self):
        if self._meta is None:
            try:
                with open(self._meta_file) as reader:
                    self._meta = pickle.load(reader)
            except:
                self._meta = {}
        return self._meta

    def add_meta(self, key, value):
        self.meta()[key] = value

    def get_meta(self, key):
        return self.meta().get(key, None)

    def add_colnames(self, colnames):
        self.add_meta('colnames', colnames)  # check consistency with dimensionality

    def get_colnames(self):
        return self.get_meta('colnames')

    def add_units(self, units):
        self.add_meta('units', units)

    def get_units(self):
        return self.get_meta('units')

    # --- Closing, context manager

    def close(self):
        if self._jagged is not None:
            self._jagged.close()
        for index in self._indices.values():
            index.close()
        self._indices = {}
        if self._meta is not None:
            with open(self._meta_file, 'w') as writer:
                pickle.dump(self._meta, writer, protocol=pickle.HIGHEST_PROTOCOL)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


#
# TODO: put a lock and make clear this is not useful in multithreading
# File-based locking for catching multiprocessing races and other problems...
#   http://tilde.town/~cristo/file-locking-in-python.html
# import portalocker
# def lock(self):
#     # very simple concurrency model
#     if self._write:
#         lock_method = portalocker.LOCK_EX | portalocker.LOCK_NB
#     else:
#         lock_method = portalocker.LOCK_SH | portalocker.LOCK_NB
#     self._lockfile = portalocker.Lock(op.join(path, 'lockfile'), lock_method=lock_method, timeout=0)
#     (Requires my branch of portalocker
# Do later and program well in the meantime...
#
# TODO: database backend, castra/blosc/bloscpack backend, pytables backend, memmap backend, scidb backend...
#       npy backend, one traj per file backend, one hdf5 dataset per traj backend...
#
# FIXME: too broad exception catchings
#
# TODO: make this fault tolerant; it actually kind of is already, but we should write keys incrementally and
#       coordinate flushes with raw blah...
#

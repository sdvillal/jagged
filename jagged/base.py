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
from functools import partial
import os.path as op
from operator import itemgetter

from toolz import merge, partition_all
import numpy as np

from jagged.misc import ensure_dir, subsegments, is_valid_segment
from whatami import whatable

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle


# --- Raw stores

class JaggedJournal(object):
    """Keeps track and persists needed details about the added arrays to a jagged instance."""

    def __init__(self, jagged, path=None):
        super(JaggedJournal, self).__init__()
        # the (master) jagged instance
        self.jagged = jagged
        # a journal must be instantiated only when jagged knows its location
        # a journal can be shared by many jagged instances (e.g. when storing different columns by different jaggeds)
        if path is None:
            path = op.join(jagged.path_or_fail(), 'journal')
        self._path = path
        # keeped info, essentially counts of arrays and rows
        self._numarrays = None   # total number of arrays
        self._lengths = None     # lenght of each added array
        self._numrows = None     # total number or rows (sum of self._lengths)
        self._segments = None    # list of (base, size) segments (inferred from self._lengths)

    def added(self, data):
        """Informs of a new added array."""
        self._write_numarrays()
        self._lengths.append(len(data))
        self._save_segment_info(data)

        if self._numrows is None:
            self._numrows = len(data)
        else:
            self._numrows += len(data)
        self._write_numrows()

        if self._numarrays is None:
            self._numarrays = 1
        else:
            self._numarrays += 1
        self._write_numarrays()

    def _read_numrows(self):
        if self._numrows is None:
            if op.isfile(op.join(self._path, 'numrows.txt')):
                with open(op.join(self.jagged.path_or_fail(), 'numrows.txt')) as reader:
                    self._numrows = int(reader.read())
            else:
                return 0
        return self._numrows

    def _write_numrows(self):
        if self._numrows is not None:
            with open(op.join(self.jagged.path_or_fail(), 'size.txt'), 'w') as writer:
                writer.write(str('%d' % self._numrows))

    def numarrays(self):
        """Returns the number of arrays in the jagged instance."""
        if self._numarrays is None:
            if op.isfile(op.join(self.jagged.path_or_fail(), 'numarrays.txt')):
                with open(op.join(self.jagged.path_or_fail(), 'numarrays.txt'), 'r') as reader:
                    self._numarrays = int(reader.read())
            else:
                self._numarrays = 0
                self._write_numarrays()
        return self._numarrays

    def _write_numarrays(self):
        if self._numarrays is not None:
            with open(op.join(self.jagged.path_or_fail(), 'numarrays.txt'), 'w') as writer:
                writer.write(str('%d' % self._numarrays))

    def _save_segment_info(self, data):
        with open(op.join(self.jagged.path_or_fail(), 'segment_sizes.csv'), 'a') as writer:
            writer.write(str('%d\n' % len(data)))

    def read_segments(self):
        sizes = np.atleast_1d(np.loadtxt(op.join(self.jagged.path_or_fail(), 'segment_sizes.csv'), dtype=int))
        bases = np.hstack(([0], np.cumsum(sizes)))
        return list(zip(bases, sizes))


@whatable(add_properties=False)
class JaggedRawStore(object):
    """Persistent storage of objects of the same type but different length."""

    def __init__(self, path):
        super(JaggedRawStore, self).__init__()
        self._path = path
        if self._path is not None:
            ensure_dir(self._path)
        self._template = None   # how the saved arrays look like
        self._journal = None

    def journal(self):
        if self._journal is None:
            self._journal = JaggedJournal(self)
        return self._journal

    # --- Where this storage resides

    def path_or_fail(self):
        """Returns the path if set, otherwise raises an exception."""
        if self._path is None:
            raise Exception('In-memory only arrays are not implemented for %s.' % self.what().id())
        return self._path

    # --- Template

    def template(self):
        template_path = op.join(self.path_or_fail(), 'template.npy')
        if self._template is None:
            if op.isfile(template_path):
                self._template = np.load(template_path)
        return self._template

    def _write_template(self, data):
        template_path = op.join(self.path_or_fail(), 'template.npy')
        np.save(template_path, data[:0])

    def is_compatible(self, data):
        """Returns True iff data can be stored.
        This usually means it is of the same kind as previously stored arrays.
        """
        # Obviously we could just store arbitrary arrays in some implementations (e.g. NPY)
        # But lets keep jagged contracts...
        template = self.template()
        if template is None:
            return True
        return (template.dtype >= data.dtype and
                data.shape[-1] == template.shape[-1] and
                np.isfortran(data) == np.isfortran(data))

    # --- Lifecycle

    # N.B. at the moment, to make things simple, we only want write or read
    # We should ensure concurrency does not break this rule (only one writer xor many readers)
    # Of course we could use stuff like SWMR from hdf5 or role our own, more featureful and compled concurrency
    # Not a priority

    @property
    def is_writing(self):
        """Returns whether we can append more data using this jagged instance."""
        raise NotImplementedError()

    @property
    def is_reading(self):
        """Returns whether we can append more data using this jagged instance."""
        raise NotImplementedError()

    @property
    def is_open(self):
        """Returns whether we are currently open in any mode."""
        raise NotImplementedError()

    def close(self):
        """Flushes buffers to permanent storage and closes the underlying backend."""
        raise NotImplementedError()

    # --- Writing data

    def _open_write(self, data=None):
        """Opens in writing mode, returns None.

        Parameters
        ----------
        data : numpy array like, default None
          data schema to use by the storage, needed if this is the first opening of the repository
        """
        raise NotImplementedError()

    def append(self, data):
        """Appends new data to this storage.

        If the storage is empty, this will define the dtype of the store.

        Parameters
        ----------
        data : numpy-array like
          The data to append, must have a compatible dtype with what was already added to the store.

        Returns
        -------
        An id (usually a tuple) that addresses the appended data in the storage.
        """

        # at the moment we do not allow coordinate-less stores
        self.path_or_fail()

        # check data validity
        if any(s < 1 for s in data.shape[1:]):
            raise Exception('Cannot append data with sizes 0 in non-leading dimension (%s, %r)' %
                            (self.what().id(), data.shape))

        # check we can write
        if self.is_reading and not self.is_writing:
            self.close()

        # template
        if self.template() is None:
            self._write_template(data)
        assert self.is_compatible(data)

        # open
        self._open_write(data)

        # write
        coords = self._append_hook(data)

        # bookkeping
        self._journal.added(data)

        # done
        return coords

    def iter_segments(self, segments_per_chunk=None):
        if segments_per_chunk is None:
            for segment in self.read_segments():
                yield self.get([segment])
        elif segments_per_chunk <= 0:
            raise ValueError('chunksize must be None or bigger than 0, it is %r' % segments_per_chunk)
        else:
            for segments in partition_all(segments_per_chunk, self.read_segments()):
                yield self.get(segments)

    def iter_rows(self, max_rows_per_chunk):
        # Iterates segments in chunks with max_rows_per_chunk as upper bound
        # (but will give at least one segment at a time)
        # This can be more (e.g. SegmentRawStorage) or less involved (e.g. JaggedByNumpy)
        # Useful to iterate with really controlled amount of memory
        raise NotImplementedError()

    def _append_hook(self, data):
        raise NotImplementedError()

    def append_from(self, jagged, chunksize=None):
        """Appends all the contents of `jagged`."""
        for chunk in jagged.iter_segments(chunksize):
            for data in chunk:
                self.append(data)

    # --- Reading data

    def _open_read(self):
        """Opens in reading mode, returns None."""
        raise NotImplementedError()

    def _get_views(self, keys, columns):
        """Returns a list of arrays corresponding to the provided keys and columns."""
        raise NotImplementedError()

    def get(self, keys=None, columns=None, factory=None):
        """Returns a list with the data specified in `keys` (and `columns`), possibly transformed by `factory`.

        Concrete implementations may warrant things like "all segments actually lie in congiguous regions in memory".

        Parameters
        ----------
        keys : list keys as returned by append
          specifies which elements to retrieve; if None, the whole data is returned

        columns : list of integers, default None
          specifies which columns to retrieve; if None, retrieve all columns

        factory : factory(ndarray)->desired type, default None
          transforms each of the returned elements into a desired type (for example, a pandas DataFrame)
          another use can be to apply summary statistics

        Returns
        -------
        A list with the retrieved elements, possibly transformed by factory.
        """

        # at the moment we do not allow coordinate-less stores
        self.path_or_fail()

        # flush if needed
        if self.is_writing and not self.is_reading:
            self.close()

        # open
        self._open_read()

        # read
        views = self._get_views(keys, columns)

        return views if factory is None else map(factory, views)

    # --- Factories / curries / partials

    def copyconf(self, **params):
        """Returns a partial function that instantiates this type of store
        with changed default parameters.

        N.B. this default implementation is based on being able to retrieve all default parameters
        using the `what` method; override if that is not the case.

        Parameters
        ----------
        params: **dict
          The parameters that will be fixed in the returned factory function.
        """
        return whatable(partial(self.__class__, **merge(self.what().conf, params)), add_properties=False)

    # --- Shape and dtype

    def _backend_attr(self, attr):
        # TODO: probably this can be simplified:
        #  - removed in favor of _backend_attr_hook
        #  - remove reimplementations in subclasses
        if not self.is_open:
            with self:
                try:
                    self._open_read()
                    return self._backend_attr_hook(attr)
                except IOError:
                    return None
        return self._backend_attr_hook(attr)

    def _backend_attr_hook(self, attr):
        if self.template() is None:
            return None
        if attr == 'shape':
            return self._read_numrows(), self.template().shape[1]
        return getattr(self.template(), attr)

    @property
    def shape(self):
        """Returns a tuple with the current size of the storage in each dimension."""
        return self._backend_attr('shape')

    @property
    def dtype(self):
        """Returns the data type of the store."""
        return self._backend_attr('dtype')

    @property
    def ndim(self):
        """Returns the number of dimensions."""
        shape = self.shape
        return len(self.shape) if shape is not None else None

    # --- Context manager and other magics...

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        """Returns the size of the leading dimension."""
        return self.shape[0] if self.shape is not None else 0

    # Also consider register to atexit


class LinearRawStorage(JaggedRawStore):

    def __init__(self, path, contiguity=None):
        """
        A segment raw storage can access arbitrary rows using base (row index in the storage) and size
        (number of rows to retrieve).

        Parameters
        ----------
        contiguity : string or None, default None
           indicates the type of contiguity sought for the results; for performance segments retrieval
           does not need to be done in any order
             - 'read': a best effort should be done to leave retrieved segments order-contiguous in memory;
                       this can potentially speed up operations reading these data in the order specified by segments
             - 'write': a best effort should be done to write segments sequentially in memory;
                        this can potentially speed up retrieval
             - 'auto': allow the backend to decide the return flavor;
                       using this the backends can return "lazy" or "cached" arrays
                       (for example, views on memmapped arrays or hdf5 datasets)
             - None: do not force any contiguity nor allow any strange return, just plain numpy arrays
                     owning their own data; this is safest and usually well performing
           usually 'read' can be a good idea for analysis, and 'auto' can have memory saving benefits
           beware that forcing contiguity for speed might lead to memory leaks
           (the whole retrieved segments won't be released while any of them is reacheable)
        """
        super(LinearRawStorage, self).__init__(path)
        self.contiguity = contiguity

    def _get_views(self, keys, columns):
        # get one segment with all if segments is None
        if keys is None:
            keys = [(0, len(self))]

        # retrieve data
        ne, nc = self.shape
        views = retrieve_contiguous(keys, columns, self._get_hook, self.dtype, ne, nc, self.contiguity)

        return views

    def _get_hook(self, base, size, columns, dest):
        raise NotImplementedError()

    def iterchunks(self, chunksize):
        """Reads `chunksize` rows at a time until all is read.
        As opposed to iter_rows, this warrants equal sized, never bigger than specified, chunks.
        """
        base = 0
        total = len(self)
        while base < total:
            size = min(chunksize, total - base)
            yield self.get([(base, size)])[0]
            base += size


def retrieve_contiguous(segments, columns, reader, dtype, ne, nc, contiguity):

    # Check for valid contiguity
    if contiguity not in ('read', 'write', 'auto', None):
        raise ValueError('Unknown contiguity scheme: %r' % contiguity)

    # Check query sanity and prepare contiguous query
    # dest_base tells where each query must go to in case of contiguity='read'
    # note that dest_base is not useful for unsorting in the presence of 0-length items (so we explicitly store order)
    dest_base = 0
    query_dest = []
    for order, (base, size) in enumerate(segments):
        if (base + size) > ne or base < 0:
            raise ValueError('Out of bounds query (base=%d, size=%d, maxsize=%d)' % (base, size, ne))
        query_dest.append((order, base, dest_base, size))
        dest_base += size
    total_size = dest_base

    nc = len(columns) if columns is not None else nc

    # Retrieve
    views = []
    if contiguity == 'read':
        # Hope for one-malloc only, but beware of memory leaks
        dest = np.empty((total_size, nc), dtype=dtype)
        # Populate
        for order, base, dest_base, size in sorted(query_dest):
            view = dest[dest_base:dest_base+size]
            view = reader(base, size, columns, view)
            views.append((order, view))
    elif contiguity == 'write':
        # Hope for one-malloc only, but beware of memory leaks
        dest = np.empty((total_size, nc), dtype=dtype)
        # Populate
        dest_base = 0
        for order, base, _, size in sorted(query_dest):
            view = dest[dest_base:dest_base+size]
            view = reader(base, size, columns, view)
            dest_base += size
            views.append((order, view))
    elif contiguity == 'auto':
        for order, base, _, size in sorted(query_dest):
            view = reader(base, size, columns, None)
            views.append((order, view))
    else:
        for order, base, _, size in sorted(query_dest):
            view = np.empty((size, nc), dtype=dtype)
            view = reader(base, size, columns, view)
            views.append((order, view))

    # Unpack views while restoring original order
    return list(map(itemgetter(1), sorted(views, key=itemgetter(0))))


# --- Index stores

@whatable(add_properties=False)
class JaggedIndex(object):
    """Maps keys to segments that can address elements in `JaggedRawStore` instances.
    Segments can be addressed by key and insertion index.
    """

    def segments(self):
        """Returns the list of known segments.
        May be larger than the number of known keys.
        """
        raise NotImplementedError()

    def segment(self, key_over_index):
        """Returns the segment in the index addressed by key_over_index.

        Parameters
        ----------
        key_over_index : hashable
          A key in the key -> index dictionary or an index itself to a a segment.
          N.B. if a key_over_index can address both the dictionary of keys and the list of segments
          (that is, an int key has been inserted in the dictionary), then the dictionary address take
          precedence and the respective segment is returned.

        Returns
        -------
        The segment (base, size)

        Raises
        ------
        TypeError: if `key_over_index` is not a hashable object or if it cannot index a list
        IndexError: if `key_over_index` is an integer not in the keys dictionary
          and cannot address in the segments list
        """
        try:
            return self.segments()[self.keys()[key_over_index]]
        except KeyError:
            return self.segments()[key_over_index]

    def num_segments(self):
        """Returns the number of segments in the index."""
        return len(self.segments())

    def keys(self):
        """Returns a dictionary mapping keys to segment indices.
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class JaggedSimpleIndex(JaggedIndex):
    """Simplemost implementation, index in-memory and persistence using pickle"""

    def __init__(self, path=None):
        self._keys = None
        self._segments = None
        self._path = path
        # populate indices
        # N.B. at the moment things are brittle and we can loose index data under certain conditions
        # FIXME asap
        self.segments()
        self.keys()

    @property
    def _keys_file(self):
        return op.join(ensure_dir(self._path), 'keys.pkl')

    @property
    def _segments_file(self):
        return op.join(ensure_dir(self._path), 'segments.pkl')

    def segments(self):
        if self._segments is None:
            try:
                with open(self._segments_file, 'rb') as reader:
                    self._segments = pickle.load(reader)
            except:  # FIX Too broad
                self._segments = []
        return self._segments

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
        return self._jagged.ndim

    @property
    def dtype(self):
        return self._jagged.dtype

#
# TODO: concurrency on reading
# Put a lock and make clear write is not possible in multithreading
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
# Are we reinventing the wheel?
#
# Look at blaze, bioinformatics approaches for intervals and the like.
#
# ---- About JaggedStore
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
# TODO: versioning mechanism (simply define versions for everything and store them somewhere)
#

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
from future.builtins import range, map
from array import array
from functools import partial
import os.path as op
from operator import itemgetter

from toolz import merge, partition_all
import numpy as np

from jagged.misc import ensure_dir, subsegments, is_valid_segment
from whatami import whatable
import json

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle


# --- Raw stores


def _int_or_0(v):
    if v is None:
        return 0
    return int(v)


def _read_full_file(x, path):
    """Reads the full contentes of file path into array x."""
    with open(path, 'rb') as reader:
        reader.seek(0, 2)
        size = reader.tell()
        reader.seek(0, 0)
        if size % x.itemsize != 0:
            raise Exception('Truncated file')
        x.fromfile(reader, size // x.itemsize)
        return x


class JaggedJournal(object):
    """Keeps track and persists information about the sizes of added arrays."""

    # a journal must be instantiated only when jagged knows its location
    # a journal can be shared by many jagged instances (e.g. when storing different columns by different jaggeds)

    def __init__(self, path):
        super(JaggedJournal, self).__init__()
        self._path = path
        # base and length of each added array
        self._lengths_file = op.join(self._path, 'lengths.array')
        self._lengths = self._read_lengths()
        self._bases = None
        # total number of rows and arrays
        self._sizes_file = op.join(self._path, 'size.json')
        self._numrows, self._numarrays = self._read_sizes()

    def append(self, data):
        """Appends the data array to the journal."""
        self._add_length(data)
        self._add_sizes(data)

    # --- Num rows, num arrays (redundant with lengths, light and good for redundancy)

    def _add_sizes(self, data):
        """Adds to numrows and numarrays the sizes of the array data and immediatly persists them."""
        self._numrows += len(data)
        self._numarrays += 1
        with open(self._sizes_file, 'w') as writer:
            json.dump({'numrows': self._numrows, 'numarrays': self._numarrays},
                      writer, encoding='utf-8', indent=2)

    def _read_sizes(self):
        """Reads the current numrows and numarrays values from persistent storage.
        If there is no info stored, makes them 0.
        """
        if op.isfile(self._sizes_file):
            with open(op.join(self._sizes_file, 'sizes.json'), 'r') as reader:
                sizes = json.load(reader)
                return _int_or_0(sizes['numrows']), _int_or_0(sizes['numarrays'])
        return 0, 0

    def numrows(self):
        """Returns the total number of rows in the jagged instance."""
        return self._numrows

    def numarrays(self):
        """Returns the number of arrays in the jagged instance."""
        return self._numarrays

    # --- Base and size of each array

    def _add_length(self, data):
        """Adds the length to the journal and immediatly persists it."""
        self._lengths.append(len(data))
        with open(self._lengths_file, 'ab') as writer:
            array[-1:].tofile(writer)

    def _read_lengths(self):
        """Reads the lengths from persistent storage, if it does not exist, returns an empty array."""
        lengths = array(b'l')
        if op.isfile(self._lengths_file):
            _read_full_file(lengths, self._lengths_file)
        return lengths

    def lengths(self):
        """Returns an array with the length of each array added to the journal."""
        return self._read_lengths()

    def bases(self):
        """Returns where each array would start if the storage is linear."""
        if self._bases is None or len(self._bases) < len(self._lengths):
            self._bases = np.hstack(([0], np.cumsum(self._lengths)))
        return self._bases

    def start_end(self, index):
        """Returns the start and end of the array at index."""
        base, size = self.base_size(index)
        return base, base + size

    def base_size(self, index):
        """Returns the base and size of the array at index."""
        return self.bases()[index], self.lengths()[index]

    # --- Sanity checks

    def check_consistency(self):
        """Checks the internal consistency of the journal."""
        assert len(self.lengths()) == len(self.bases())
        assert len(self.lengths()) == self.numarrays()
        assert len(np.sum(self.lengths())) == self.numrows()


@whatable(add_properties=False)
class JaggedRawStore(object):
    """Persistent storage of objects of the same type but different length."""

    def __init__(self, path, journal=None):
        super(JaggedRawStore, self).__init__()
        self._path = path
        if self._path is not None:
            ensure_dir(self._path)
        self._template = None    # how the saved arrays look like
        self._journal = journal  # sizes of the added arrays

    # --- Where this storage resides

    def path_or_fail(self):
        """Returns the path if set, otherwise raises an exception."""
        if self._path is None:
            raise Exception('In-memory only arrays are not implemented for %s.' % self.what().id())
        return self._path

    # --- Journal

    def journal(self):
        if self._journal is None:
            self._journal = JaggedJournal(op.join(self.path_or_fail(), 'journal'))
        return self._journal

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

    def can_add(self, data):
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
        An integer addressing the added array in the storage
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
        assert self.can_add(data)

        # open
        self._open_write(data)

        # write
        self._append_hook(data)

        # bookkeeping
        index = self.journal().numarrays()
        self.journal().append(data)

        # done
        return index

    def _append_hook(self, data):
        """Saves the data, returns nothing."""
        raise NotImplementedError()

    def append_from(self, jagged, arrays_per_chunk=None):
        """Appends all the contents of jagged."""
        for chunk in jagged.iter_arrays(arrays_per_chunk):
            for data in chunk:
                self.append(data)

    # --- Reading

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
        keys : list of keys
          specifies which elements to retrieve; if None, all arrays are returned

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

    # -- Iteration

    def iter_arrays(self, arrays_per_chunk=None):
        """Iterates over the arrays in this store."""
        if arrays_per_chunk is None:
            for key in range(self.journal().numarrays()):
                yield self.get([key])
        elif arrays_per_chunk <= 0:
            raise ValueError('arrays_per_chunk must be None or bigger than 0, it is %r' % arrays_per_chunk)
        else:
            for segments in partition_all(arrays_per_chunk, self.journal().numarrays()):
                yield self.get(segments)

    def __iter__(self, arrays_per_chunk=None):
        """Alias to iter_arrays."""
        return self.iter_arrays(arrays_per_chunk=arrays_per_chunk)

    # def iter_rows(self, max_rows_per_chunk):
    #     # Iterates segments in chunks with max_rows_per_chunk as upper bound
    #     # (but will give at least one segment at a time)
    #     # This can be more (e.g. SegmentRawStorage) or less involved (e.g. JaggedByNumpy)
    #     # Useful to iterate with really controlled amount of memory
    #     raise NotImplementedError()

    #  --- Factories / curries / partials

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

    # --- shape, dtype, order

    @property
    def shape(self):
        """Returns a tuple with the current size of the storage in each dimension."""
        ncol = self.ncols
        return None if ncol is None else (self.nrows, ncol)

    @property
    def dtype(self):
        """Returns the data type of the store."""
        template = self.template()
        return None if template is None else template.dtype

    @property
    def ndims(self):
        """Returns the number of dimensions."""
        # Actually at the moment we only support ndims == 2
        shape = self.shape
        return len(self.shape) if shape is not None else None

    @property
    def ncols(self):
        """Returns the number of columns."""
        template = self.template()
        return None if template is None else template.shape[1]

    @property
    def nrows(self):
        """Returns the number of rows in the store."""
        return self.journal().numrows()

    @property
    def narrays(self):
        """Returns the number or arrays in the store."""
        return self.journal().numarrays()

    @property
    def order(self):
        """Returns 'C' for row major, 'F' for column major."""
        template = self.template()
        if template is None:
            return None
        return 'F' if np.isfortran(template) else 'C'


    # --- Context manager and other magics...

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        """Returns the size of the leading dimension."""
        return self.shape[0] if self.shape is not None else 0

    # Also consider register to atexit as a parameter to the constructor


class LinearRawStorage(JaggedRawStore):

    def __init__(self, path, journal=None, contiguity=None):
        """
        A segment raw storage can access arbitrary rows using base (row index in the storage) and size
        (number of rows to retrieve).

        Parameters
        ----------
        journal : must quack like JaggedJournal, default None
          see base class

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
        super(LinearRawStorage, self).__init__(path, journal=journal)
        self.contiguity = contiguity

    def _get_views(self, keys, columns):
        # get all segments if segments is None
        if keys is None:
            keys = range(self.journal().numarrays())
        keys = [self.journal().base_size(key) if isinstance(key, int) else key for key in keys]

        # retrieve data
        ne, nc = self.shape
        views = retrieve_contiguous(keys, columns, self._get_hook, self.dtype, ne, nc, self.contiguity)

        return views

    def _get_hook(self, base, size, columns, dest):
        raise NotImplementedError()

    def iter_rows(self, rows_per_chunk):
        """Reads rows_per_chunk rows at a time until all is read."""
        base = 0
        total = len(self)
        while base < total:
            size = min(rows_per_chunk, total - base)
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
        return self._jagged.ndims

    @property
    def dtype(self):
        return self._jagged.dtype

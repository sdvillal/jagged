# coding=utf-8
"""Convenient (and somehow performing) storage of objects with homogeneous types but different lengths.

"jagged" array providers have very simple, low level contracts:
 - Focus on reading performance, append only store.
 - Use numpy arrays as the canonical data carriers
 - May or may not restrict the type of the stored elements
 - Retrieve only by providing indices *collections*
   No explicit support for slice notation
 - All clases are whatami whatables
"""
from __future__ import absolute_import, unicode_literals, print_function
from future.utils import bytes_to_native_str
from abc import ABCMeta
from array import array
from functools import partial
import os.path as op
from operator import itemgetter
import json

from future.builtins import range, map
from toolz import merge, partition_all
import numpy as np

from jagged.misc import ensure_dir
from whatami import whatable

try:
    import cPickle as pickle
except ImportError:  # pragma: no cover
    import pickle


# --- Journals (persitence of array lengths)


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
        self._path = ensure_dir(path)
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
            json.dump({'numrows': self._numrows, 'numarrays': self._numarrays}, writer, indent=2)

    def _read_sizes(self):
        """Reads the current numrows and numarrays values from persistent storage.
        If there is no info stored, makes them 0.
        """
        if op.isfile(self._sizes_file):
            with open(self._sizes_file, 'r') as reader:
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
            self._lengths[-1:].tofile(writer)

    def _read_lengths(self):
        """Reads the lengths from persistent storage, if it does not exist, returns an empty array."""
        lengths = array(bytes_to_native_str(b'l'))
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

# --- Raw stores


@whatable(add_properties=False)
class JaggedRawStore(object):
    """Persistent storage of objects of the same type but different length."""

    __metaclass__ = ABCMeta

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
            self._journal = JaggedJournal(op.join(self.path_or_fail(), 'meta', 'journal'))
        return self._journal

    # --- Template

    def template(self):
        template_dir = ensure_dir(op.join(self.path_or_fail(), 'meta', 'template'))
        template_path = op.join(template_dir, 'template.npy')
        if self._template is None:
            if op.isfile(template_path):
                self._template = np.load(template_path)
        return self._template

    def _write_template(self, data):
        template_dir = ensure_dir(op.join(self.path_or_fail(), 'meta', 'template'))
        template_path = op.join(template_dir, 'template.npy')
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

        # id log
        if not op.isfile(op.join(self.path_or_fail(),  'meta', 'whatid.txt')):
            ensure_dir(op.join(self.path_or_fail(),  'meta'))
            with open(op.join(self.path_or_fail(), 'meta', 'whatid.txt'), 'w') as writer:
                writer.write(self.what().id())

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
        for chunk in jagged.iter_arrays(arrays_per_chunk=arrays_per_chunk):
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
            for segments in partition_all(arrays_per_chunk, range(self.journal().numarrays())):
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
        ncols = self.ncols
        return None if ncols is None else (self.nrows, ncols)

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


# --- Linear stores (can address arbitrary row segments)


class LinearRawStorage(JaggedRawStore):

    __metaclass__ = ABCMeta  # no harm, lint stops complaining

    def __init__(self, path, journal=None, contiguity=None):
        """
        A linear raw storage can access arbitrary rows using base (row index in the storage) and size
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

        if 0 == len(keys):
            return []

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

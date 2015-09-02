# coding=utf-8
"""Examples copied verbatim in the readme.
This should probably be a notebook.
"""
from __future__ import print_function
import os.path as op
import shutil
import numpy as np
import pandas as pd
import tempfile
from jagged.mmap_backend import JaggedByMemMap
from jagged.blosc_backend import JaggedByBlosc

# A Jagged instance is all you need
mmap_dir = tempfile.mkdtemp('mmap')
jagged = JaggedByMemMap(op.expanduser(path=mmap_dir))
# You can drop here any JaggedRawStore implementation you want to

# Generate a random dataset
print('Creating a random dataset...')
rng = np.random.RandomState(0)
num_arrays = 1000
max_length = 2000
num_columns = 100
originals = [rng.randn(rng.randint(0, max_length), num_columns)
             for _ in range(num_arrays)]

# Add these to the store ("with" context is usually optional, but recommended)
print('Populating the jagged store...')
with jagged:
    indices = list(map(jagged.append, originals))

# Some jagged stores optimize queries retrieving arrays by their insertion order
# Retrieval speed should not suffer much even with random queries
shuffled_indices = rng.permutation(indices).tolist()
shuffled_originals = [originals[i] for i in shuffled_indices]

# What do we have in store?
print('Number of arrays: %d, number of rows: %d' % (jagged.narrays, jagged.nrows))
# Out: Number of arrays: 200, number of rows: 193732
print('Jagged shape=%r, dtype=%r, order=%r' %
      (jagged.shape, jagged.dtype, jagged.order))
# Out: Jagged shape=(193732, 50), dtype=dtype('float64'), order='C'

# Check roundtrip
roundtrippeds = jagged.get(shuffled_indices)
for original, roundtripped in zip(shuffled_originals, roundtrippeds):
    assert np.array_equal(original, roundtripped)
print('Roundtrip checks pass')

# Jagged stores self-identified themselves (using whatami)
print(jagged.what().id())
# Out: JaggedByMemMap(autoviews=True,contiguity=None)

# Jagged stores can be iterated in chunks (see iter)
for original, roundtripped in zip(originals, jagged):
    assert np.array_equal(original, roundtripped[0])
print('Roundtrip checks for iteration pass')

# Some jagged stores allow to retrieve arbitrary rows without penalty
# (i.e. without retrieving the whole containing array).
# These are marked as "linear" in the store feature matrix.
# You do so by passing a list of (base, size) segments.
some_rows = jagged.get([[3, 22], [45, 1000]])
assert len(some_rows[1]) == 1000
assert np.array_equal(some_rows[0], originals[0][3:25])
print('Roundtrip checks for row retrieval pass')

# Some jagged stores allow to be lazy retrieving the arrays.
# On top of that, the MemMap implementation allow memmapped arrays.
# Can be handy to have long lists of views in memory
# while letting the OS managing memory fetching and eviction for us.
jbmm = JaggedByMemMap(op.expanduser(path=mmap_dir),
                      autoviews=True,
                      contiguity='auto')
print('Retrieving %d arrays...' % (len(shuffled_indices) * 100))
many_arrays = jbmm.get(shuffled_indices * 100)
# This will work also for pandas DataFrames as long as
# "copy=True" is honored by the pandas constructor
# that is, the dtype of the arrays is simple),
print('Making %d dataframes...' % (len(shuffled_indices) * 100))
columns = pd.Index(np.arange(num_columns))
dfs = [pd.DataFrame(data=array, columns=columns, copy=False)
       for array in many_arrays]
print('Checking roundtrip...')
for original, roundtripped in zip(shuffled_originals * 100, dfs):
    assert np.array_equal(original, roundtripped)
print('Roundtrip checks for lazy dataframes pass')

# Jagged stores can be populated from other jagged stores
blosc_dir = tempfile.mkdtemp('mmap')
jbb = JaggedByBlosc(path=blosc_dir)
print('Saving compressed (although these data are not compressable)...')
jbb.append_from(jagged)
for a_from_mmap, a_from_blosc in zip(jbb, jagged):
    assert np.array_equal(a_from_mmap, a_from_blosc)
print(jbb.what().id())
print('Roundtrip checks for compressed arrays pass')
# Out: JaggedByBlosc(compressor=BloscCompressor(cname='lz4hc',
#                                               level=5,
#                                               n_threads=1,
#                                               shuffle=True))

# We are done, cleanup
shutil.rmtree(mmap_dir, ignore_errors=True)
shutil.rmtree(blosc_dir, ignore_errors=True)

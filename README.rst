jagged
======

Efficient storage of same-type, uneven-size arrays
--------------------------------------------------

|Pypi Version| |Build Status| |Coverage Status| |Scrutinizer Status|

Jagged_ is an ongoing amateur project exploring the storage panorama
for datasets containing (large amounts of) arrays with the same type
and number of columns, but varying number of rows. Examples of such
datasets for which *jagged* has been used are collections of multivariate
timeseries (short animal behaviour snippets) and collections of molecules
(represented as varying length strings).

Jagged aims to help analyzing data in the laptop and the cluster, in batch
or interactively, providing a very lightweight store. Jagged provides fast
retrieval of array subsets for many-GB datasets containing millions of rows.

Requirements
------------

All the requirements are pip-installable and listed in in pypi.

Jagged needs numpy_, whatami_ and python-future_.

Jagged stores build on top of several optional high quality python libraries: c-blosc_, python-blosc_,
bloscpack_, bcolz_ and joblib_. Testing relies on pytest_.

Getting the right combination for blosc, python-blosc, bcolz and bloscpack can be a challenge
(but worth the effort). At the moment (2015/09/02), we recommend using the latest released
versions of c-blosc (1.7.0) in combination with the latest releases of python-blosc (1.2.7)
and bloscpack (0.9.0).

Jagged runs in python 2.7+ and 3.4+. At the moment it has been tested only on linux, but it should
work on mac and windows as well.


Installation
------------

It should suffice to use pip::

    pip install jagged

Showcase
--------

Using jagged is simple. There are different implementations that provide
two basic methods: **append** adds a new array to the store, **get** retrieves
collections of arrays identified by their insertion order in the store. Usually
the lifecycle of a jagged store is also simple: there is no explicit open,
append and get calls can be interleaved at will and the only needed action
to warrant consistency is to close after write, which can be achieved by calling
**close**, by calling *get* or by using a with statement with the provided
context manager.

This is a `real life`_ small example combining jagged with indices and queries
over real data.

Another synthetic example follows:

.. code:: python

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


Backends
--------

Although rapidly changing, *jagged* already provides the following storage backends
that can be considered as working and stable. Other backends are planned.

+-------------------+------+-------+--------+------+-----+------+------+
| Backend           | comp | chunk | column | mmap | lin | lazy | cont |
+===================+======+=======+========+======+=====+======+======+
| JaggedByBlosc     | X    |       |        | X    |     |      |      |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByCarray    | X    | X     |        |      | X   |      | X    |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByH5Py      | X    | X     |        |      | X   | X    | X    |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByJoblib    | X    | X     |        |      |     |      |      |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByMemMap    |      |       |        | X    | X   | X    | X    |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByNPY       |      |       |        |      |     |      |      |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByBloscpack | X    |       |        |      |     |      |      |
+-------------------+------+-------+--------+------+-----+------+------+
| JaggedByPickle    | X    | X     |        |      |     |      |      |
+-------------------+------+-------+--------+------+-----+------+------+


- comp:
  can be compressed
- chunk:
  can be chunked
- column:
  stores columns of the array contiguously (can be easily implemented by using a store per column)
- mmap:
  can open a memmap to the data
- lin:
  can retrieve any row without the need to retrieve the whole array it contains it
- lazy:
  the arrays are not fetched immediatly; this can mean also that they can be managed
  as virtual-memory by the OS (JaggedByMemMap only)
- cont:
  retrieved arrays can be forced to lie in contiguous memory segments


Benchmarks
----------

What backend and parameters work best depends on whether the data is compressible or not, the
sizes of the arrays and the kind of queries. We have a good idea of what works best for our data
and query types and are working at providing a benchmarking framework, that can be useful if
you can get a good sample of the data to store. Find here a preview_, results will be soon posted here.


By-design constraints
---------------------

Jagged would like to be simple: conceptually, to deploy and to use.

Jagged is about retrieving full arrays.
Focus is on fast retrieval of arbitrary batch queries.
Batch queries over arrays appended closeby should be faster.
Jagged is good for local caches or reducing the burden of
network file systems.

Jagged stores are append only.

There is no transaction, replication or distribution or...
It is all files in your local or network disks, written once, read many times.
If you have complex data or requirements, there are many better options.
If you have simple numerical arrays you want to load fast and store light,
jagged might be worth a try.

Not important efforts have been given yet to optimize
(although some backends work quite smoothly).
At the moment, everything is simple algorithms implemented in pure python.


Links
-----

This neat blogpost_ from Matthew Rocklin is highly recommended, as it delivers
the promised *"vocabulary to talk about efficient tabular storage"*. Add perhaps
"blocked" (as in "compression is done in cache-friendly sized blocks") and
"chunked" (as in "retrieval is done in I/O-friendly sized chunks") to the lexicon.
The castra_ project is worth a look.


.. _Jagged: https://github.com/sdvillal/jagged
.. |Pypi Version| image:: https://badge.fury.io/py/jagged.svg
   :target: http://badge.fury.io/py/jagged
.. |Build Status| image:: https://travis-ci.org/sdvillal/jagged.svg?branch=master
   :target: https://travis-ci.org/sdvillal/jagged/branches
.. |Coverage Status| image:: http://codecov.io/github/sdvillal/jagged/coverage.svg?branch=master
   :target: http://codecov.io/github/sdvillal/jagged?branch=master
.. |Scrutinizer Status| image:: https://scrutinizer-ci.com/g/sdvillal/jagged/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/sdvillal/jagged/?branch=master
.. _real life: https://github.com/strawlab/strawlab-examples/blob/master/strawlab_examples/euroscipy/euroscipy_example.py
.. _preview: https://github.com/sdvillal/strawlab-examples/tree/master/strawlab_examples/benchmarks
.. _numpy: http://www.numpy.org/
.. _whatami: http://www.github.com/sdvillal/whatami
.. _python-future: http://python-future.org/
.. _c-blosc: https://github.com/Blosc/c-blosc
.. _python-blosc: https://github.com/Blosc/python-blosc
.. _bloscpack: https://github.com/Blosc/bloscpack
.. _bcolz: https://github.com/Blosc/bcolz
.. _joblib: https://pythonhosted.org/joblib/
.. _pytest: http://pytest.org
.. _blogpost: http://matthewrocklin.com/blog/work/2015/08/28/Storage/
.. _castra: https://github.com/blaze/castra

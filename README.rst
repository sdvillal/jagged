jagged
======

Efficient storage of same-type, uneven-size arrays
--------------------------------------------------

|Pypi Version| |Build Status| |Coverage Status| |Scrutinizer Status|

*Jagged* is an ongoing amateur project exploring the storage panorama
for datasets containing (large amounts of) arrays with the same type
and number of columns, but varying number of rows. Examples of such
datasets for which *jagged* has been used are collections of multivariate
timeseries (short animal behaviour snippets) and collections of molecules
(represented as varying length strings).

Jagged aims to help analyzing data in the laptop and the cluster, in batch
or interactively, providing a very lightweight store. Jagged provides fast
retrieval of array subsets for many-GB datasets containing millions of rows.

By-design constraints
---------------------

Focus is on fast retrieval of arbitrary batch queries.

Jagged stores are append only.

There is no transaction, replication or distribution.
It is all files in your local or network disks.

Not important efforts have been given yet to optimize
(although some backends work quite smoothly).

At the moment, everything is simple algorithms implemented in pure python.

Installation
------------

It should suffice to use pip::

    pip install jagged

Jagged stores builds on top of several high quality python libraries: numpy, blosc,
bloscpack, bcolz and joblib. It also needs whatami and python-future.
Testing relies on pytest (you need to install all dependencies to test at the moment,
this will change soon).


Showcase
--------

Using jagged is simple. There are different implementations that provide
two basic methods: *append* adds a new array to the store, *get* retrieves
collections of arrays identified by their insertion order in the store.

.. code:: python

    import os.path as op
    import numpy as np
    from jagged.mmap_backend import JaggedByMemmap

    # A Jagged instance is all you need
    jagged = JaggedByMemmap(op.expanduser(path='~/jagged-example/mmap'))
    # You can drop here any you want to

    # Generate a random dataset
    rng = np.random.RandomState(0)
    max_length = 2000
    num_arrays = 100
    originals = [rng.randn(rng.randint(0, max_length), 50)
                 for _ in range(num_arrays))

    # Add these to the store (context is usually optional but recommended)
    with jagged:
        indices = map(jagged.append, originals)

    # What do we have in store?
    print('Number of arrays: %d, number of rows: %d' % (jbmm.narrays, jbmm.nrows))
    print('Jagged shape=%r, dtype=%r, order=%r' %
          (jagged.shape, jagged.dtype, jagged.order))

    # Check roundtrip
    roundtripped = jagged.get(indices)
    print('The store has %d arrays')

    # Jagged stores self-identified themselves (using whatami)
    print(jagged.what().id())

    # Jagged stores can be iterated in chunks
    # See iter

    # Jagged stores can be populated from other jagged stores

    # Some jagged stores allow to retrieve arbitrary rows as fast
    # as arbitrary arrays.


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


- comp: can be compressed
- chunk: can be chunked
- column: stores columns of the array contiguously (can be easily implemented by using a store per column)
- mmap: can open a memmap to the data
- lin: can retrieve any row without the need to retrieve the whole
       array it contains it
- lazy: the arrays are not fetched immediatly; this can mean also that they can be managed
        as virtual-memory by the OS (JaggedByMemMap only)
- cont: retrieved arrays can be forced to lie in contiguous memory segments

Benchmarks
----------

What backend and parameters work best depends on whether your data is compressible or not and the
sizes of the arrays. We have a good idea of what works best for our data and are working at
providing a benchmarking framework. Find here a preview_.


.. |Pypi Version| image:: https://badge.fury.io/py/jagged.svg
   :target: http://badge.fury.io/py/jagged
.. |Build Status| image:: https://travis-ci.org/sdvillal/jagged.svg?branch=master
   :target: https://travis-ci.org/sdvillal/jagged
.. |Coverage Status| image:: http://codecov.io/github/sdvillal/jagged/coverage.svg?branch=master
   :target: http://codecov.io/github/sdvillal/jagged?branch=master
.. |Scrutinizer Status| image:: https://scrutinizer-ci.com/g/sdvillal/jagged/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/sdvillal/jagged/?branch=master
.. _preview: https://github.com/sdvillal/strawlab-examples/tree/master/strawlab_examples/benchmarks

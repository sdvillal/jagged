jagged
======

Efficient storage of same-type, uneven-size arrays
--------------------------------------------------

|Pypi Version| |Build Status| |Coverage Status| |Scrutinizer Status|

*Jagged* is an ongoing project exploring the storage panorama for datasets
containing (large amounts of) arrays with the same type and number of
columns but varying number of rows. Examples of such datasets for which
*jagged* has been used are collections of multivariate timeseries (short
animal behavior snippets) and large collections of molecules (represented
as varying length strings).

Showcase
--------

.. code:: python

    import os.path as op
    import numpy as np
    from jagged.mmap_backend import JaggedByMemmap

    # A Jagged instance is all you need
    jagged = JaggedByMemmap(op.expanduser(path='~/jagged-example/mmap'))

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
    print('Jagged shape=%r, dtype=%r, order=%r' % (jagged.shape, jagged.dtype, jagged.order))

    # Check roundtrip
    roundtripped = jagged.get(indices)
    print('The store has %d arrays')




Although rapidly changing, *jagged* already provides the following storage backends
(all accessed by a simple unifying API) that can be considered as working
and stable.

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


Installation
------------

Jagged stores builds on top of several high quality python libraries: numpy, blosc,
bloscpack, bcolz and joblib. It also needs pyopy, future. Testing relies on pytest.


---------------------

.. |Pypi Version| image:: https://badge.fury.io/py/jagged.svg
   :target: http://badge.fury.io/py/jagged
.. |Build Status| image:: https://travis-ci.org/sdvillal/jagged.svg?branch=master
   :target: https://travis-ci.org/sdvillal/jagged
.. |Coverage Status| image:: http://codecov.io/github/sdvillal/jagged/coverage.svg?branch=master
   :target: http://codecov.io/github/sdvillal/jagged?branch=master
.. |Scrutinizer Status| image:: https://scrutinizer-ci.com/g/sdvillal/jagged/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/sdvillal/jagged/?branch=master

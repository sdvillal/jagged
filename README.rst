jagged: tricks for efficient loading of data pieces
===================================================

|Pypi Version| |Build Status| |Coverage Status|

Goodies and exploration of storage backends to get easy on memory and time.

Light storage of elements with different sizes. Current applications:
 - Serialised rdkit molecules
 - Cool storage for ML experiments results
 - Uneven length time-series (ala strawlab)

Goals:
  - Easy on storage requirements
  - Fast retrieval
  - Metadata can be all loaded, actual series/molecules can then be loaded on demand
  - Allow partial retrieval
  - Allow easy spec of intervals

Design limitations:
  - No removal - just rewrite
  - Costly updates
  - Hard to change schema

Are we reinventing the wheel?

Look at blaze, bioinformatics approaches for intervals and the like.

.. |Build Status| image:: https://travis-ci.org/sdvillal/jagged.svg?branch=master
   :target: https://travis-ci.org/sdvillal/jagged
.. |Coverage Status| image:: https://img.shields.io/coveralls/sdvillal/jagged.svg
   :target: https://coveralls.io/r/sdvillal/jagged
.. |Pypi Version| image:: https://badge.fury.io/py/jagged.svg
   :target: http://badge.fury.io/py/jagged


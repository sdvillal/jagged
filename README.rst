jagged
======

tricks for efficient loading of same-type, uneven-size data
-----------------------------------------------------------

|Pypi Version| |Build Status| |Coverage Status| |Scrutinizer Status|

Light storage of elements with different sizes. Current applications:
 - Serialised rdkit molecules
 - Large-scale ML experiments results
 - Uneven length multivariate time-series

Goals:
  - Easy on storage requirements
  - Fast retrieval
  - All metadata can be loaded, actual series/molecules can then be loaded on demand
  - Allow partial retrieval and easy interval specification

Design limitations:
  - No removal - just rewrite
  - Costly updates
  - Hard to change schema

.. |Pypi Version| image:: https://badge.fury.io/py/jagged.svg
   :target: http://badge.fury.io/py/jagged
.. |Build Status| image:: https://travis-ci.org/sdvillal/jagged.svg?branch=master
   :target: https://travis-ci.org/sdvillal/jagged
.. |Coverage Status| image:: http://codecov.io/github/sdvillal/jagged/coverage.svg?branch=master
   :target: http://codecov.io/github/sdvillal/jagged?branch=master
.. |Scrutinizer Status| image:: https://scrutinizer-ci.com/g/sdvillal/jagged/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/sdvillal/jagged/?branch=master

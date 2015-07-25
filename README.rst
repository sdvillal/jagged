jagged
======

tricks for efficient loading of same-type, uneven-size data
-----------------------------------------------------------

|Pypi Version| |Pypi Downloads| |Build Status| |Scrutinizer Status| |Coverage Status|

|Coverage History|

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

.. |Build Status| image:: https://travis-ci.org/sdvillal/jagged.svg?branch=master
   :target: https://travis-ci.org/sdvillal/jagged
.. |Coverage Status| image:: http://codecov.io/github/sdvillal/jagged/coverage.svg?branch=master
   :target: http://codecov.io/github/sdvillal/jagged?branch=master
.. |Coverage History| image:: http://codecov.io/github/sdvillal/jagged/branch.svg?branch=master
.. |Scrutinizer Status| image:: https://scrutinizer-ci.com/g/sdvillal/jagged/badges/quality-score.png?b=master
   :target: https://scrutinizer-ci.com/g/sdvillal/jagged/?branch=master
.. |Pypi Version| image:: https://badge.fury.io/py/jagged.svg
   :target: http://badge.fury.io/py/jagged
.. |Pypi Downloads| image:: https://pypip.in/d/$REPO/badge.png
   :target: https://crate.io/packages/$REPO/
   :alt: Number of PyPI downloads

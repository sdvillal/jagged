# coding=utf-8
from __future__ import absolute_import

# --- Backend imports

from .mmap_backend import JaggedByMemMap

try:
    from .compressed_raw_backend import JaggedByCompression
except ImportError:  # pragma: no cover
    JaggedByBlosc = None

try:
    from .bcolz_backend import JaggedByCarray
except ImportError:  # pragma: no cover
    JaggedByCarray = None

try:
    from .h5py_backend import JaggedByH5Py
except ImportError:  # pragma: no cover
    JaggedByH5Py = None

from .npy_backend import JaggedByNPY

try:
    from .bloscpack_backend import JaggedByBloscpack
except ImportError:  # pragma: no cover
    JaggedByBloscpack = None

from .pickle_backend import JaggedByPickle

try:
    from .joblib_backend import JaggedByJoblib
except ImportError:  # pragma: no cover
    JaggedByJoblib = None


# --- Version

__version__ = '0.1.1-dev0'

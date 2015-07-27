# coding=utf-8
"""A jumble of seemingly useful stuff."""
from __future__ import unicode_literals
from itertools import chain
import numbers
import os
import os.path as op
import numpy as np


def home():  # pragma: no cover
    """Returns current user home dir."""
    return op.expanduser('~')  # Valid in both py2 and py3


def ensure_writable_dir(path):  # pragma: no cover
    """Ensures that a path is a writable directory."""
    def check_path(path):
        if not op.isdir(path):
            raise Exception('%s exists but it is not a directory' % path)
        if not os.access(path, os.W_OK):
            raise Exception('%s is a directory but it is not writable' % path)
    if op.exists(path):
        check_path(path)
    else:
        try:
            os.makedirs(path)
        except Exception:
            if op.exists(path):  # Simpler than using a file lock to work on multithreading...
                check_path(path)
            else:
                raise
    return path


def ensure_dir(path):  # pragma: no cover
    return ensure_writable_dir(path)


# --- Intervals

def crossings(x, threshold=0, after=False):
    """Returns the indices of the elements before or after crossing a threshold.

    N.B. touching the threshold itself is considered a cross.

    Parameters
    ----------
    x: array
    The data

    threshold: float, default 0
    Where crossing happens.

    after: bool, default False
    If True, the indices represent the elements after the cross, if False the elements before the cross.

    Returns
    -------
    The indices where crosses happen.

    Examples
    --------

    >>> print(crossings(np.array([0, 1, -1, -1, 1, -1])))
    [0 1 3 4]
    >>> print(crossings(np.array([0, 1, -1, -1, 1, -1]), after=True))
    [1 2 4 5]
    >>> print(crossings(np.array([0, 0, 0])))
    []
    >>> print(crossings(np.array([0, 3, -3, -3, 1]), threshold=1))
    [0 1 3]
    >>> print(crossings(np.array([0, 3, -3, -3]), threshold=-2.5))
    [1]
    >>> print(crossings(np.array([[0, 3], [-3, -3]]), threshold=-2.5))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    Exception: Only 1D arrays, please (you gave me 2 dimensions)
    """
    if len(x.shape) > 1:
        raise Exception('Only 1D arrays, please (you gave me %d dimensions)' % len(x.shape))
    where_crosses = np.where(np.diff(np.sign(x - threshold)))[0]
    if after:
        return where_crosses + 1
    return where_crosses


def find_intervals(x):
    """
    Finds the intervals in which x is True or non-zero.


    Returns
    -------
    Pairs of indices representing the intervals in which x is True or nonzero.
    The pairs represent valid python intervals, lower point included, upper point excluded.


    Examples
    --------
    >>> find_intervals([])
    []
    >>> find_intervals([1])
    [(0, 1)]
    >>> find_intervals([0, 1])
    [(1, 2)]
    >>> find_intervals([0, 0, 1, 1, 0, 0, 1, 1, 0])
    [(2, 4), (6, 8)]
    >>> find_intervals([0, 0, 0])
    []
    >>> find_intervals([1, 1, 1])
    [(0, 3)]
    >>> find_intervals([True, True, True])
    [(0, 3)]
    >>> find_intervals([1, 1, 1, 0])
    [(0, 3)]
    """
    # This ugly 6 lines are here because:
    #   - we allow to pass lists but we need numpy arrays
    #   - we want to allow both boolean (True, False) arrays and numeric arrays
    #   - we want to use the crossings function which only accepts numeric arrays
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not x.dtype == np.bool:
        x = x != 0
    zeros_ones = np.zeros_like(x, dtype=np.int)
    zeros_ones[x] = 1

    # Find where we change from being in an interval to not being in an interval
    starts_ends = list(crossings(zeros_ones, after=True))

    # Do we start already in an interval?
    if len(zeros_ones) > 0 and 1 == zeros_ones[0]:
        starts_ends = [0] + starts_ends

    # Do we end in an interval?
    if len(zeros_ones) > 0 and 1 == zeros_ones[-1]:
        starts_ends = starts_ends + [len(x)]

    assert len(starts_ends) % 2 == 0

    starts = starts_ends[0::2]
    ends = starts_ends[1::2]
    return list(zip(starts, ends))


def subsegments(segment, *subsegments):
    """Make subsegments relative to the start of a base segment, checking for boundaries.

    Parameters
    ----------
    segment : tuple (base, size)
      The segment to which relative subsegments are being specified

    subsegments : list of (base, size) boolean arrays specifying subsegments
      These can be either something like (3, 8) (ss_base, ss_size), or boolean lists/arrays
      It is assumed that ss_base is here is offset from `segment` base

    Returns
    -------
    A list of subsegments [(base, size)], each lying within the boundaries of `segment`.

    Examples
    --------
    >>> subsegments((5, 100))
    []
    >>> subsegments((5, 100), (11, 14))
    [(16, 14)]
    >>> subsegments((5, 100), (11, 14), (3, 88))
    [(16, 14), (8, 88)]
    >>> subsegments((0, 5), [True, True, False, True, True])
    [(0, 2), (3, 2)]
    >>> subsegments((0, 5), [False] * 5)
    []
    >>> subsegments((0, 5), [True] * 5)
    [(0, 5)]
    >>> subsegments((0, 5), np.array([True] * 5))
    [(0, 5)]
    >>> subsegments((0, 5), [True, True, False, True, True], (2, 2))
    [(0, 2), (3, 2), (2, 2)]
    >>> subsegments((0, 100), (90, 11))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: (90, 11) is not a valid subsegment specification for (0, 100)
    >>> subsegments((0, 100), (-3, 8))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: (90, 11) is not a valid subsegment specification for (0, 100)
    >>> subsegments((0, 100), ('a', 8))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: ('a', 8) is not a valid subsegment specification for (0, 100)
    >>> subsegments((0, 100), 'crazyyou')  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: 'crazyyou' is not a valid subsegment specification for (0, 100)
    """

    # This implementation is slow, but seemingly correct; I do not think it will bottleneck

    base, size = segment

    def bool2segments(ss):
        ssa = np.array(ss)
        if ssa.dtype.kind == 'b' and ssa.ndim == 1 and len(ssa) == size:
            return [(start, end - start) for start, end in find_intervals(ssa)]
        return None

    def is_valid_segment(ss):
        if not isinstance(ss, (tuple, list)):
            return False
        if not len(ss) == 2:
            return False
        ss_base, ss_size = ss
        if not isinstance(ss_base, numbers.Integral) and isinstance(ss_size, numbers.Integral):
            return False
        if ss_base < 0 or (base + ss_base + ss_size) > (base + size):
            return False
        return True

    def bool_and_valid(ss):
        if is_valid_segment(ss):
            return [ss]
        ss_from_bool = bool2segments(ss)
        if ss_from_bool is not None and all(map(is_valid_segment, ss_from_bool)):
            return ss_from_bool
        raise ValueError('%r is not a valid subsegment specification for %r' % (ss, segment))

    return [(base + ss_base, ss_size) for ss_base, ss_size in chain(*[bool_and_valid(ss) for ss in subsegments])]

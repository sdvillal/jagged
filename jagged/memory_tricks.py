# coding=utf-8
import numbers
import numpy as np
import pandas as pd


def reduce_while_freeing(xs,
                         reduce_function=pd.concat,
                         initializer=None,
                         batch_size=0.2):
    """
    A kind of reduce, but:
      - xs must be a list-like supporting len and pop (no iterator support)
      - the elements in xs must not be referenced anywhere else
      - the reduce function takes a list, the first element is the accumulator

    Examples
    --------
    This will concatenate 1000 pandas dataframes taking only memory for 1200 at a time.
    It trades not doubling memory with more reallocs / mallocs + memcopys
    >>> dfs = [pd.DataFrame(np.ones((100, 10))) for _ in xrange(1000)]
    >>> cdf = reduce_while_freeing(dfs, batch_size=0.2)

    Returns
    -------
    The reduced value

    Side effects
    ------------
    xs is emptied
    """

    # TODO: same function over iterators, batched reduce. Use e.g. cytoolz take + similar API to this

    if isinstance(batch_size, numbers.Real):
        if not 0 < batch_size <= 1:
            raise Exception('If batch size is a real number, it must be in (0, 1]')
        batch_size = int(round(len(xs) * batch_size))

    # Check refcounts
    from sys import getrefcount
    if any(getrefcount(df) > 3 for df in xs):  # 3 references expected: in list, in loop and in function call
        raise Exception('For this function to be worthy, no dataframe must be referenced outside the passed list')

    reduced = initializer
    while len(xs) > 0:
        batch = [xs.pop(0) for _ in xrange(min(len(xs), batch_size))]
        reduced = reduce_function(batch if reduced is None else [reduced] + batch)

    return reduced


#
# Concatenate without doubling memory usage brain dump
# ----------------------------------------------------
# Having the convenience of pandas dataframes is great, but it requires that you have all your data in memory.
# If data does not fit in memory, an option to keep convenience would be to use any of the many out-of-core
# (DataFrame-like) data analysis libraries These are gaining momentum and have different scopes but a common
# denominator: allow to analyze data that does not fit in main memory while actually being light on memory usage.
# Some randomly chosen examples for the python world are: apache spark dataframes,
# http://ddf.io/, https://github.com/sparklingpandas/sparklingpandas, bcolz, blaze...
#
# Out-of-core pandas is debated at least in these issues:
# https://github.com/pydata/pandas/issues/3202
# https://github.com/pydata/pandas/issues/5902
# https://github.com/pydata/pandas/issues/2305
# (see also related tools like the HDFStore that allow to load the data selectively)
#
# Sometimes we want to create big dataframes out of many (not so) small pieces
# that we have been collected incrementally and/or in parallel (simplemost ETL ever).
# (examples bothering: fish data in max laptop, small feature matrices in poor Katja's VM,
# large molecular datasets in manysources for Flo, or the results of the large bootstrapping
# machine learning experiments in manysources).
#
# Pandas DataFrames use numpy arrays to carry their data, one array per dtype.
# Numpy is not apt for these tasks: each enlarging operation requires malloc and memcpy.
# Therefore, pandas internal representation, requiring everything of the same dtype to be packed
# in the very same numpy array, makes data aggregation operations like concat, adding a column
# and others will always require double memory allocation and copy (of the data with the
# same dtype involved, to be precise).
#
# Options:
#   - Find the total used data a priory (e.g. concat will make a 300000x200000 dataframe),
#     malloc once, load different parts at a time (we can dump data into disk first if needed).
#
#   - We can just dump everythn to disk, free memory, reload the aggregated dataframe.
#     Do it with bcolz or pytables... anything that allows incremental updates in disk.
#
#   - In the same spirit as 1) but without needing disk dump, we can just trade memory by memory copy.
#     For example, if we had 1000 dataframes to concatenate we could concatenate the first 100,
#     then free the memory for these, then concatenate the next 100, then free...
#
# bcolz is one important example trying to alleviate these problems. In memory bcolz arrays could
# anyway copy everything first to a numpy array, so bcolz in memory is not a solution. But bcolz in disk,
# with its compressed chunking strategy at 0 API adaptation, can indeed make this easy: store the data in a
# (temporary) bcolz array in disk, then create the dataframe out of the numpy array created from bcolz.
#  This is equivalent, but arquably more convenient, to storing the data anywhere in disk
# (e.g. in a pyttables container).
#

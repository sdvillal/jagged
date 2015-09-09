# coding=utf-8
from __future__ import print_function
from collections import OrderedDict
from functools import partial
from glob import glob
import os.path as op
import shutil
from time import time

import humanize
import joblib
import numpy as np
import pandas as pd

import bcolz
from jagged.bcolz_backend import JaggedByCarray
from jagged.benchmarks.utils import sync, available_ram, timestr, drop_caches, du, hostname
from jagged.compressed_raw_backend import JaggedByCompression
from jagged.bloscpack_backend import JaggedByBloscpack
from jagged.compressors import JaggedCompressorByBlosc
from jagged.h5py_backend import JaggedByH5Py
from jagged.misc import ensure_dir
from jagged.mmap_backend import JaggedByMemMap
from jagged.npy_backend import JaggedByNPY
from jagged.pickle_backend import JaggedByPickle



# --- Free{flight/swim} datasets
from strawlab_examples.minifly import split_df

MITFA_RELEASE_PATH = op.join(op.expanduser('~'), 'data-analysis', 'strawlab', 'mitfa', 'release')
RNAi_RELEASE_PATH = op.join(op.expanduser('~'), 'data-analysis', 'strawlab', 'rnai', 'release')
NEUROPEPTIDE_DEGRADATION_PATH = op.expanduser('~/np-degradation')

DEFAULT_BENCHMARKS_DEST = ensure_dir(op.join(op.expanduser('~'), 'benchs-jagged'))


# --- Numpy

def array_nan_equal(a, b):
    try:
        np.testing.assert_equal(a, b)
    except AssertionError:
        return False
    return True


# --- Others

def merge_ordered_dicts(*dicts):
    result = OrderedDict()
    for d in dicts:
        for k, v in d.items():
            result[k] = v
    return result


# --- Benchmarks

def populate(hub, tdf, jagged, arrays_per_chunk=1000, redo=False):

    measurements = OrderedDict()

    DONE_FILE = op.join(jagged.path_or_fail(), 'DONE')
    if op.isfile(DONE_FILE) and not redo:
        print('Already done, skipping...')
        return (np.nan,) * 4

    measurements['released_read_time'] = 0
    measurements['write_time'] = 0
    measurements['reread_time'] = 0
    measurements['roundtrip_check_time'] = 0

    for tdf in split_df(tdf, chunk_size=arrays_per_chunk):

        # Read from released
        start = time()
        series = hub.series_df(tdf)
        measurements['released_read_time'] += time() - start

        # Write to the store
        start = time()
        indices = []
        for df in series.series:
            indices.append(jagged.append(df.values))
        sync()
        measurements['write_time'] += time() - start

        # Read back
        measurements['before_reread_mem'] = available_ram()
        start = time()
        roundtripped = jagged.get(indices)
        measurements['reread_time'] += time() - start
        measurements['after_reread_mem'] = available_ram()

        # Roundtrip check
        start = time()
        for rt, df in zip(roundtripped, series.series):
            assert array_nan_equal(rt, df.values)
        measurements['roundtrip_check_time'] += time() - start

    with open(DONE_FILE, 'w') as writer:
        writer.write(timestr())

    return measurements


def query(tdf, jagged, seed=None, query_size=None):

    measurements = OrderedDict()

    # warmup
    start = time()
    jagged.get([0])
    measurements['warmup_time'] = time() - start

    # contiguous or random query
    indices = np.arange(len(tdf)) if seed is None else np.random.RandomState(0).permutation(len(tdf))
    if query_size is not None:
        total_size = int(len(indices) * query_size)
        indices = indices[:total_size]
    else:
        total_size = len(indices)
    measurements['query_percent'] = query_size
    measurements['total_size'] = total_size  # number of rows

    # drop caches, read
    measurements['before_read_mem'] = available_ram()
    try:
        drop_caches(jagged.path_or_fail())
        start = time()
        jagged.get(indices)
        measurements['cold_read_time'] = (time() - start)
    except RuntimeError:
        jagged.get(indices)
        measurements['cold_read_time'] = np.nan
    measurements['after_read_mem'] = available_ram()

    # read, maybe hot caches
    start = time()
    roundtripped = jagged.get(indices)
    measurements['read_time'] = (time() - start)

    # pandify
    start = time()
    columns = pd.Index('%d' % col for col in range(jagged.shape[1]))
    roundtripped = [pd.DataFrame(data, copy=False, columns=columns) for data in roundtripped]
    measurements['pandify_time'] = (time() - start)

    # sum (will give a small idea on the overhead of laziness / mmap)
    measurements['before_sum_mem'] = available_ram()
    start = time()
    measurements['suma'] = float(np.sum([np.nansum(df['6']) for df in roundtripped]))
    measurements['sum_time'] = time() - start
    measurements['after_sum_mem'] = available_ram()

    # get a checksum from the whole collection
    start = time()
    measurements['checksum'] = joblib.hash(tuple(joblib.hash(df) for df in roundtripped))
    measurements['checksum_time'] = time() - start

    return measurements


def run_bench(hub,
              jaggeds,
              take_each,
              seed,
              dest=None,
              delete_before=False,
              delete_after=False):

    tdf = hub.trials_df()
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(tdf))[::take_each]
    # These will go contiguous to the generated stores
    tdf = tdf.iloc[permutation]
    # We try 4 query sizes, relative to the size of the dataset

    if dest is None:
        dest = DEFAULT_BENCHMARKS_DEST
    dest_jagged = op.join(dest, 'each=%d' % take_each, 'seed=%r' % seed)

    for jagged in jaggeds:
        print(jagged().what().id())
        path = op.join(dest_jagged, jagged().what().id())
        if delete_before:
            shutil.rmtree(path, ignore_errors=True)
        jagged = jagged(path=path)
        with jagged:
            populate_stats = populate(hub, tdf, jagged)
        with jagged:
            query_stats = [query(tdf,
                                 jagged=jagged,
                                 seed=seed,
                                 query_size=qs)
                           for qs in (0.2, 0.4, 0.8, 1.0)]
        populate_stats['disk_usage'] = du(path)
        populate_stats['take_each'] = take_each
        populate_stats['host'] = hostname()
        populate_stats['path'] = op.abspath(jagged.path_or_fail())
        populate_stats['seed'] = seed
        populate_stats['jagged'] = jagged.what().id()
        populate_stats['narrays'] = jagged.narrays
        populate_stats['nrows'] = jagged.shape[0]
        populate_stats['ncols'] = jagged.shape[0]
        for name, value in populate_stats.items():
            if 'disk_usage' == name or name.endswith('_mem'):
                print('\t%s: %s' % (name, humanize.naturalsize(value)))
            elif isinstance(value, int):
                print('\t%s: %d' % (name, value))
            elif isinstance(value, float):
                print('\t%s: %.2f' % (name, value))
            else:
                print('\t%s: %r' % (name, value))
        if delete_after:
            shutil.rmtree(path, ignore_errors=True)
        stats = [merge_ordered_dicts(populate_stats, qs)
                 for qs in query_stats]
        pd.to_pickle(stats,
                     op.join(dest, joblib.hash(stats)))


if __name__ == '__main__':

    take_eachs = [1000, 100, 10, 2, 1]
    hubs = (
        FreeflightHub(NEUROPEPTIDE_DEGRADATION_PATH),
        # FreeflightHub(RNAi_RELEASE_PATH),
        # FreeflightHub(MITFA_RELEASE_PATH)
    )
    seeds = [None, 0, 1, 2]
    jaggeds = (
        # pickle
        partial(JaggedByPickle, compress=False, arrays_per_chunk=500),
        partial(JaggedByPickle, compress=False, arrays_per_chunk=2000),
        partial(JaggedByPickle, compress=True, arrays_per_chunk=500),
        partial(JaggedByPickle, compress=True, arrays_per_chunk=2000),

        # npy
        JaggedByNPY,

        # bloscpack
        JaggedByBloscpack,

        # blosc
        partial(JaggedByCompression, compressor=JaggedCompressorByBlosc(cname='lz4hc', clevel=5, shuffle=True)),
        partial(JaggedByCompression, compressor=JaggedCompressorByBlosc(cname='lz4hc', clevel=5, shuffle=False)),

        # mmap
        partial(JaggedByMemMap, contiguity='auto', autoviews=True),
        partial(JaggedByMemMap, contiguity=None, autoviews=False),

        # carray
        partial(JaggedByCarray, cparams=bcolz.cparams(clevel=5, shuffle=True, cname='lz4hc')),
        partial(JaggedByCarray, cparams=bcolz.cparams(clevel=5, shuffle=False, cname='lz4hc')),

        # h5py
        JaggedByH5Py,
    )

    # Run the benchmarks
    # take_eachs = [1000, 100, 10, 5]  # 2, 1 take forever, even with small datasets
    # for hub, take_each, seed in product(hubs, take_eachs, seeds):
    #     run_bench(hub=hub,
    #               jaggeds=jaggeds,
    #               take_each=take_each,
    #               seed=seed,
    #               delete_before=True,
    #               delete_after=True)

    def duhfy(df):
        df['duh'] = df.disk_usage.map(humanize.naturalsize)
        return df

    def read_results(src=None):
        if src is None:
            src = DEFAULT_BENCHMARKS_DEST
        results = []
        for pkl in glob(op.join(src, '*')):
            try:
                results.extend(pd.read_pickle(pkl))
            except:
                pass
        df = pd.DataFrame(results, columns=results[0].keys())
        return duhfy(df)
    
    BENCH_COLS = [
        'released_read_time', 'write_time', 'reread_time',
        'roundtrip_check_time', 'before_reread_mem', 'after_reread_mem',
        'disk_usage', 'host', 'path', 'seed', 'jagged', 'narrays',
        'nrows', 'ncols', 'warmup_time', 'query_percent', 'total_size',
        'before_read_mem', 'cold_read_time', 'after_read_mem', 'read_time',
        'pandify_time', 'before_sum_mem', 'suma', 'sum_time',
        'after_sum_mem', 'checksum', 'checksum_time', 'duh'
    ]

    results = read_results()
    # print(results.columns)
    # exit(22)

    groupers = (
        'jagged',
        ('jagged', 'query_percent'),
        ('jagged', 'query_percent', 'nrows'),
    )
    meaned = results.groupby(groupers[2])[['disk_usage',
                                           'pandify_time',
                                           'cold_read_time',
                                           'read_time',
                                           'nrows']].mean()

    meaned = duhfy(meaned)
    print(meaned)
    meaned.to_html('/home/santi/meaned.html')

# Just one pickle shines, but it is quite irreal if memory is really an issue
# Implement JaggedByChunkedPickles
# Obviously, the best would be as we had until now, a pickle per uuid

# Talk about data arrangement in jaggeds, important insertion to have slow
# keys together (i.e. first this uuid, then condition...)
# Most queries would benefit from it

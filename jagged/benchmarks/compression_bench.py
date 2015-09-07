# coding=utf-8
from __future__ import print_function, unicode_literals, absolute_import, division
from future.builtins import str
from collections import defaultdict
from copy import copy
from glob import glob
from time import time
from array import array
import os.path as op
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import humanize

from jagged.benchmarks.utils import hostname, drop_caches
from jagged.compression.compressors import JaggedCompressorByBlosc
from jagged.misc import ensure_dir
from whatami import whatable, what2id, What, id2what, whatid2columns


@whatable
class CompressionStatsCollector(object):
    """Collects the performance on compression of dataframe collections."""

    def __init__(self,
                 compressor,
                 item_getter=None,
                 size_getter=None,
                 comparer=None):
        super(CompressionStatsCollector, self).__init__()
        # compressor(s)
        self.compressor = compressor
        self._compressors = {}
        # roundtrip passes
        self._rts = defaultdict(lambda: array(b'b'))
        # lengths
        self._lengths = defaultdict(lambda: array(b'L'))
        # uncompressed size
        self._usizes = defaultdict(lambda: array(b'L'))
        # compressed size
        self._csizes = defaultdict(lambda: array(b'L'))
        # time taken in compression
        self._ctimes = defaultdict(lambda: array(b'f'))
        # time taken in decompression
        self._dtimes = defaultdict(lambda: array(b'f'))
        # time taken in roundtrip checks
        self._rtimes = defaultdict(lambda: array(b'f'))
        # item kinds
        if item_getter is None:
            def item_getter(x):
                return x.values
        if size_getter is None:
            def size_getter(x):
                return x.nbytes
        if comparer is None:
            comparer = array_nan_equal
        self._item_getter = item_getter
        self._size_getter = size_getter
        self._comparer = comparer

    def _compressor(self, name):
        if name not in self._compressors:
            # copy because compressors can be stateful
            # (e.g. if they are meant to compress only one kind of dataframes)
            self._compressors[name] = copy(self.compressor)
        return self._compressors[name]

    def add(self, df, name='whole_df', fail_with_rt=False, columnar_too=True):

        # compressor
        compressor = self._compressor(name)

        # new length
        self._lengths[name].append(len(df))

        # get the compressible original
        original = self._item_getter(df)

        # compress
        start = time()
        compressed = compressor.compress(original)
        self._ctimes[name].append(time() - start)

        # sizes
        self._csizes[name].append(len(compressed))
        self._usizes[name].append(self._size_getter(original))

        # decompress
        start = time()
        decompressed = compressor.decompress(compressed)
        self._dtimes[name].append(time() - start)

        # roundtrip check
        start = time()
        rtok = self._comparer(original, decompressed)
        self._rtimes[name].append(time() - start)
        self._rts[name].append(rtok)
        if fail_with_rt and not rtok:
            raise Exception('Roundtrip failed')

        # do also for columns?
        if columnar_too:
            for col in df.columns:
                self.add(df[col], col, fail_with_rt=fail_with_rt, columnar_too=False)

    def names(self):
        return sorted(self._csizes.keys())

    def weights(self, name, use_length=True):
        # Does length influence compression? (compare to total cr)
        # Light read on the inspection paradox:
        #   http://allendowney.blogspot.co.at/2015/08/the-inspection-paradox-is-everywhere.html
        w = self.lengths(name) if use_length else self.usizes(name)
        return w / np.sum(w)

    def lengths(self, name):
        return np.array(self._lengths[name])

    def length(self, name):
        return self.lengths(name).sum()

    def usizes(self, name):
        return np.array(self._usizes[name])

    def usize(self, name, human=False):
        dsize = self.usizes(name).sum()
        return humanize.naturalsize(dsize) if human else dsize

    def csizes(self, name):
        return np.array(self._csizes[name])

    def csize(self, name, human=False):
        csize = self.csizes(name).sum()
        return humanize.naturalsize(csize) if human else csize

    def crs(self, name, weight_by=None):
        crs = self.usizes(name) / self.csizes(name)
        if weight_by is None:
            return crs
        elif weight_by == 'length':
            return crs / self.weights(name, use_length=True)
        elif weight_by == 'usize':
            return crs / self.weights(name, use_length=False)
        else:
            raise ValueError('Unknown weighting policy %r, must be one of [None, "length", "usize"]' % weight_by)

    def cr(self, name, weight_by='length'):
        valid_weights = {'total', 'length', 'usize', None}
        if weight_by not in valid_weights:
            raise ValueError('Unknown weighting policy %r, must be one of %r' % (weight_by, sorted(valid_weights)))
        if weight_by == 'total':
            return (self.usize(name) / self.csize(name)), 0
        crs = self.crs(name, weight_by=weight_by).mean()
        return crs.mean(), crs.std()

    def roundtrips_ok(self, name):
        return np.array(self._rts[name], dtype=np.bool)

    def roundtrip_ok(self, name):
        return self.roundtrips_ok(name).all()

    def ctimes(self, name):
        return np.array(self._ctimes[name])

    def ctime(self, name):
        return self.ctimes(name).sum()

    def dtimes(self, name):
        return np.array(self._dtimes[name])

    def dtime(self, name):
        return self.dtimes(name).sum()

    def rtimes(self, name):
        return np.array(self._rtimes[name])

    def rtime(self, name):
        return self.rtimes(name).sum()

    @staticmethod
    def summary_measures():
        return (
            ('name', 'the name of the dataframe (usually "whole_df" or the name of a column)'),
            ('rtok', 'True iff all compress/decompress roundtrips succeed in reconstructing the original'),
            ('usize', 'the uncompressed size, in bytes'),
            ('csize', 'the compressed size, in bytes'),
            ('usizeh', 'the uncompressed size, as a human friendly string'),
            ('csizeh', 'the compressed size, as a human friendly string'),
            ('cr', 'the compression ratio (usize / csize); >1 means compression was possible'),
            ('lcrm', 'the mean of length-normalised compression ratios'),
            ('lcrs', 'the standard deviation of length-normalised compression ratios'),
            ('ctime', 'total compression time, in seconds'),
            ('dtime', 'total decompression time, in seconds'),
            ('rtime', 'total roundtripping time, in seconds'),
        )

    def summary_dict(self, name):
        cr, _ = self.cr(name, weight_by='total')
        lcrm, lcrs = self.cr(name, weight_by='length')
        return {
            'name': name,
            'rtok': self.roundtrip_ok(name),
            'usize': self.usize(name),
            'csize': self.csize(name),
            'usizeh': self.usize(name, human=True),
            'csizeh': self.csize(name, human=True),
            'cr': cr,
            'lcrm': lcrm,
            'lcrs': lcrs,
            'ctime': self.ctime(name),
            'dtime': self.dtime(name),
            'rtime': self.rtime(name),
        }

    def summary_df(self, names=None, all_but=False, fixed_values=None):

        # Get the names
        if names is None:
            names = self.names()
        elif isinstance(names, str):
            names = [names]

        # Invert the names set
        if all_but:
            names = [name for name in self.names() if name not in names]

        # Create the dataframe
        df = pd.DataFrame(map(self.summary_dict, names))

        # Tidying: add fixed values to the dataframe
        fv_cols = []
        if fixed_values is not None:
            for fv_name, fv_value in fixed_values:
                df[fv_name] = fv_value
                fv_cols.append(fv_name)

        # Tidying: reorder columns
        scols = [scol for scol, _ in self.summary_measures()]
        df = df[fv_cols + scols]

        return df.sort('cr')


def bench_compressibility(sdfs,
                          chunk_size=100,
                          compressor=JaggedCompressorByBlosc(cname='lz4hc', shuffle=True)):

    # Collect stats for individual vs larger blocks compressability
    individual_stats = CompressionStatsCollector(compressor)
    pooled_stats = CompressionStatsCollector(compressor)

    for batch in split_df(sdfs, chunk_size=chunk_size):
        # Compress each trial individually
        for df in batch.series:
            individual_stats.add(df, columnar_too=True)
        # Compress all trials pooled
        pooled = pd.concat(batch.series.tolist())
        pooled_stats.add(pooled, columnar_too=True)

    return individual_stats, pooled_stats


def timestr():
    return datetime.now().strftime('%Y/%m/%d %H:%M:%S')


class CompressionBenchManager(object):

    def __init__(self, path=None):
        if path is None:
            path = op.expanduser(op.join('~', 'free-compression-bench'))
        self.path = path
        ensure_dir(self.path)

    def run_bench(self):
        compressors = (
            JaggedCompressorByBlosc(cname='lz4hc', shuffle=True),
            JaggedCompressorByBlosc(cname='lz4hc', shuffle=False),
            JaggedCompressorByBlosc(cname='blosclz', shuffle=True),
            JaggedCompressorByBlosc(cname='blosclz', shuffle=False),
        )

        dataset_paths = MITFA_RELEASE_PATH, RNAi_RELEASE_PATH

        sizes_chunks = (500, 100), (2000, 500), (2000, 1000)

        for dataset_path in dataset_paths:
            hub = FreeflightHub(dataset_path)
            tdf = hub.trials_df()

            for num_trials, chunk_size in sizes_chunks:

                rng = np.random.RandomState(0)
                trials = tdf.iloc[rng.permutation(len(tdf))[:num_trials]]
                start = time()
                sdfs = hub.series_df(trials, groups=hub.series_group_names())
                time_read_s = time() - start
                n_cols = len(sdfs.series.iloc[0].columns)

                for compressor in compressors:

                    start = timestr()

                    # Run id
                    dataset = 'mitfa' if 'mitfa' in dataset_path else 'rnai'
                    fixed_values = OrderedDict((
                        ('dataset', dataset),
                        ('compressor', compressor),
                        ('num_trials', num_trials),
                        ('chunk_size', chunk_size),  # num trials per chunk
                        ('time_read_s', time_read_s),
                        ('n_cols', n_cols),
                        ('host', hostname())
                    ))
                    what_run = What('benchrun', conf=dict(fixed_values))
                    print(what2id(what_run))

                    # Run benchmark
                    cindiv, cpooled = bench_compressibility(sdfs,
                                                            chunk_size=chunk_size,
                                                            compressor=copy(compressor))

                    # Override compressor value with its id string
                    fixed_values['compressor'] = what2id(compressor)
                    # Get the summary dataframes
                    indiv_df = cindiv.summary_df(fixed_values=fixed_values.items())
                    indiv_df['indiv'] = True
                    pooled_df = cpooled.summary_df(fixed_values=fixed_values.items())
                    pooled_df['indiv'] = False
                    result_df = pd.concat((indiv_df, pooled_df))

                    # Store the result
                    id_sha = what_run.id(maxlength=1)
                    run_dir = ensure_dir(op.join(self.path, id_sha))
                    result_df.to_pickle(op.join(run_dir, 'results.pkl'))
                    with open(op.join(run_dir, 'whatid.txt'), 'w') as writer:
                        writer.write(what_run.id() + '\n\n')
                        writer.write(start + '\n\n')
                        writer.write(timestr())

    def load_bench_results(self):

        dfs = []
        for bench_result_dir in glob(op.join(self.path, '*')):
            if not op.isdir(bench_result_dir):
                continue
            if not op.isfile(op.join(bench_result_dir, 'whatid.txt')):
                continue
            dfs.append(pd.read_pickle(op.join(bench_result_dir, 'results.pkl')))
        results_df = pd.concat(dfs)
        # make sizes to be GB
        results_df['usize'] /= 1024 ** 3
        results_df['csize'] /= 1024 ** 3
        # compute speed columns (GB/s)
        results_df['r_speed'] = results_df['usize'] / results_df['time_read_s']
        results_df['c_speed'] = results_df['usize'] / results_df['ctime']
        results_df['d_speed'] = results_df['usize'] / results_df['dtime']
        # compute relative times to read if compressed
        results_df['taken_raw_hot'] = results_df['usize'] / 3.7
        results_df['taken_c_hot'] = (results_df['csize'] / 3.7 +
                                     results_df['usize'] * results_df['d_speed'])
        results_df['taken_hot_ratio'] = results_df['taken_c_hot'] / results_df['taken_raw_hot']

        results_df['taken_raw_cold'] = results_df['usize'] / 0.5
        results_df['taken_c_cold'] = (results_df['csize'] / 0.5 +
                                      results_df['usize'] * results_df['d_speed'])

        results_df['taken_cold_ratio'] = results_df['taken_c_cold'] / results_df['taken_raw_cold']

        return results_df


def raw_ssd_speed(path='/home/santi/ssdtests/randomfile'):
    # dd if=/dev/random iflag=fullblock of=randomfile bs=100M count=1
    # dd if=/dev/zero of=zerofile bs=100M count=1
    size = op.getsize(path)
    drop_caches(path)
    start = time()
    with open(path, 'r') as reader:
        read = len(reader.read())
        assert read == size
    taken_cold = time() - start
    start = time()
    with open(path, 'r') as reader:
        read = len(reader.read())
        assert read == size
    taken_hot = time() - start
    size_GB = size / 1024 ** 3
    print('Cold: %.2f GB/s' % (size_GB / taken_cold))
    print('Hot: %.2f GB/s' % (size_GB / taken_hot))
# raw_ssd_speed()
# A 10GB zeros file
# Cold: 0.50 GB/s
# Hot: 3.70 GB/s
# exit(22)


if __name__ == '__main__':

    cbm = CompressionBenchManager()

    # Run the benchmark
    # cbm.run_bench()
    # print('Done')
    # exit(22)

    from whatami.registry import WhatamiRegistry
    nicknames = WhatamiRegistry()

    results_df = cbm.load_bench_results()

    # Make a short-id column for the compressor name
    for compressor_id in results_df.compressor.unique():
        what = id2what(compressor_id)
        nicknames.register(what, what.positional_id())
    results_df['compressor_nick'] = results_df.compressor.apply(nicknames.id2nick)

    # Put the compressor parameters as columns
    whatid2columns(results_df, 'compressor', ('cname', 'level', 'shuffle'))

    # Get only best configuration results
    # results_df = results_df.query('cname in ["lz4hc"] and shuffle')

    def chunk_does_not_matter(rdf):
        # of curse it matters when fetching,
        # to do the less amount of work possible
        max_crs = rdf.groupby(['name', 'chunk_size'])['cr'].max()
        print(max_crs)
    chunk_does_not_matter(results_df)

    def columnar_best(rdf):
        # drop slow decompressors
        rdf = rdf.copy()
        rdf['dspeed'] = (rdf['usize'] / 1024 ** 3) / rdf['dtime']  # GB / s
        # Let's define fast enough to be
        print(rdf['dspeed'])
        max_crs = rdf.groupby('name')[['usize', 'csize']].min()
        print(max_crs)
        print(max_crs.sum())
        max_crs.to_csv('/home/santi/maxcrs.html')
    columnar_best(results_df)

    def shuffle_needed(rdf):
        # select only resluts with the whole dataset
        rdf = rdf.query('name == "whole_df"')
        # select only non-diff results
        rdf = rdf[~rdf.compressor.str.contains('DiffCompressor')]
        # select only compressing results
        rdf = rdf.query('cr > 1.2')

        print(rdf.groupby('shuffle')['cr'].max())
        # print(rdf.groupby('shuffle')['ctime'].min())
        # print(rdf.groupby('shuffle')['ctime'].max())
        # print(rdf.groupby('shuffle')['dtime'].max())

        print(rdf.groupby('shuffle')['dtime'].min())
        rdf.sort('cr', ascending=False).to_html('/home/santi/mola.html')
        print(rdf.shape)
    shuffle_needed(results_df)

    means = results_df.groupby(['name',
                                'compressor_nick',
                                'indiv',
                                'chunk_size',
                                'host',
                                'dataset'])[['cr',
                                             'rctime',
                                             'rdtime',
                                             'rrtime']].mean()


#
# TODO: group by most similar trials, compress these together
# TODO: sort whole array, unsort
# TODO: Use highest compression per column
#

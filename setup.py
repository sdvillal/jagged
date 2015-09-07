#!/usr/bin/env python
# coding=utf-8

# Authors: Santi Villalba <sdvillal@gmail.com>
# Licence: BSD 3 clause

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import jagged

setup(
    name='jagged',
    license='BSD 3 clause',
    description='Simple tricks for efficient loading or merging collections of unevenly sized elements',
    long_description=open('README.rst').read().replace('|Build Status| |Coverage Status| |Scrutinizer Status|', ''),
    version=jagged.__version__,
    url='https://github.com/sdvillal/jagged',
    author='Santi Villalba',
    author_email='sdvillal@gmail.com',
    packages=['jagged',
              'jagged.compression',
              'jagged.benchmarks',
              'jagged.tests'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: Unix',
    ],
    install_requires=['future',
                      'numpy',
                      'whatami>=4.0.0',
                      'toolz'],
    tests_require=['pytest'],
    extras_require={
        'blosc': ['blosc'],
        'bloscpack': ['bloscpack'],
        'bcolz': ['bcolz'],
        'h5py': ['h5py'],
        'joblib': ['joblib'],
        'lz4': ['lz4'],
        'benchmarks': ['psutil']
    }
)

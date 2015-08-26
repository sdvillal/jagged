# coding=utf-8
"""Benchmarking utilities.
Some of these are inspired by bloscpack / bloscpack-benchmarks.
  https://github.com/Blosc/bloscpack-benchmarking
"""
import os.path as op
import socket
import datetime
import subprocess
import json
import os
from jagged.misc import ensure_dir
import psutil

#
# Timing is hard and we should at least use timeit
# (something with support for calibration and repetition).
# A great resource is also pytest benchmark
#   https://pypi.python.org/pypi/pytest-benchmark/2.5.0
#   https://bitbucket.org/haypo/misc/src/tip/python/benchmark.py
# There are a bunch of benchmarker / timer etc. libraries in pypi
# Do not forget about /usr/bin/time -v
#


def timestr():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def hostname():
    return socket.gethostname()


def collect_sysinfo(dest=None):
    """
    Collects basic information from the machine using several tools.
    This needs to run as root.
    Note that speeds are theoretical, not measured
    (specially peak network and network drives speeds should be measured).

    Prerequisites
    -------------
    If in ubuntu:
      sudo apt-get install smartmontools inxi dmidecode
    If in arch:
     sudo pacman -S smartmontools inxi dmidecode

    What is run
    -----------
    # Basic information about mount points
    mount > mount.info
    # Inxi reports
    inxi > inxi.info
    # Full dmidecode
    dmidecode > dmidecode.info
    # Network speed information
    dmesg | grep -i duplex > network-speed.info
    # SMART information
    sudo smartctl -a /dev/sda > smartctl-sda.info

    References
    ----------
    http://www.binarytides.com/linux-commands-hardware-info/
    http://www.cyberciti.biz/faq/linux-command-to-find-sata-harddisk-link-speed/
    http://www.cyberciti.biz/faq/howto-setup-linux-lan-card-find-out-full-duplex-half-speed-or-mode/
    http://www.cyberciti.biz/tips/linux-find-out-wireless-network-speed-signal-strength.html
    """

    #
    # Any way of getting actual memory latencies, CAS...?
    # Also we could look at pure python libraries like dmidecode
    #

    if dest is None:
        dest = op.join(op.dirname(__file__), 'sysinfo')
    dest = op.join(ensure_dir(op.join(dest, hostname())), timestr() + '.json')

    info = {
        'mount': subprocess.check_output('mount'),
        'dmesg-eth': '\n'.join(line for line in subprocess.check_output('dmesg').splitlines() if 'duplex' in line),
        'iwconfig': subprocess.check_output('iwconfig'),
        'inxiF': subprocess.check_output(['inxi', '-c 0', '-F']),
        # add some more inxi stuff
    }

    with open(dest, 'w') as writer:
        json.dump(info, writer, indent=2, sort_keys=True)


def du(path):
    """Returns the size of the tree under path in bytes."""
    return int(subprocess.check_output(['du', '-s', '-L', '-B1', path]).split()[0].decode('utf-8'))


def drop_caches(path, drop_level=3, verbose=False):
    #
    # Some light reading
    #   http://www.linuxatemyram.com/play.html
    # vmtouch
    #   https://aur.archlinux.org/packages/vmtouch/
    #   http://serverfault.com/questions/278454/is-it-possible-to-list-the-files-that-are-cached
    #   http://serverfault.com/questions/43383/caching-preloading-files-on-linux-into-ram
    # fincore
    #   yaourt -S --noconfirm perl-file-sharedir-install
    #   yaourt -S --noconfirm fincore
    # To drop system caches, one needs root; an option, add the program
    # to sudoers so no pass is required.
    #
    if 0 != os.system('vmtouch -e -f -q "%s"' % path):
        if os.geteuid() == 0:
            os.system('echo %d > /proc/sys/vm/drop_caches' % drop_level)
            if verbose:
                print('Full system cache dropped because of %s' % path)
        else:
            raise RuntimeError('Need vmtouch or root permission to drop caches')
    else:
        if verbose:
            print('All pages under %s evicted' % path)


def sync():
    """Flushes buffers to disk."""
    os.system('sync')


def available_ram():
    return psutil.virtual_memory().available

#
# We need to make sure that:
#  - we go beyond microbenchmarks and look at relevant tasks
#    e.g. realtime visualisation or data exploration as opposed to batch
#

#
# Measure dataset complexity (e.g. lempel ziv via compression) and report it
#

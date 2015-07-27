# coding=utf-8
"""A jumble of seemingly useful stuff."""
from __future__ import unicode_literals
import os
import os.path as op


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

"""
Utilities to access data cache directories.

There is one cache rootdir for writing and reading:
    PYTHOR3_DATA (default: $HOME/.pythor3/data)

There is zero or more colon-delimited root dirs for only reading:
    PYTHOR3_DATA_ALT_READ

The cache_reopen function will prefer to load from PYTHOR3_DATA, but fall back
to the directories listed in PYTHOR3_DATA_ALT_READ if a file is not found
there.

"""

import os
import sys

from . import PYTHOR3_DATA
from . import PYTHOR3_DATA_ALT_READ


def cache_open(relpath, mode='r', consult_alt_read=True):
    """
    Open a file in the pythor cache.

    For read-only modes, the PYTHOR3_DATA_ALT_READ directories are consulted if
    PYTHOR3_DATA doesn't have the file.  For write modes, PYTHOR3_DATA_ALT_READ
    directories will not be consulted.

    If consult_alt_read==False, even read-only mode opens will ignore those
    directories.
    """
    if relpath.startswith(str(os.path)):
        raise ValueError('not a relpath', relpath)
    try:
        return open(os.path.join(PYTHOR3_DATA, relpath), mode)
    except IOError:
        if not (mode in ('r', 'rb') and consult_alt_read):
            raise

    assert mode in ('r', 'rb')
    assert consult_alt_read
    for i, root in enumerate(PYTHOR3_DATA_ALT_READ):
        fullpath = os.path.join(root, relpath)
        try:
            return open(fullpath, mode)
        except IOError:
            if i == len(PYTHOR3_DATA_ALT_READ) - 1:
                raise
    assert 0  # execution never gets here


def cache_rm(relpath):
    """
    Delete a file out of PYTHOR3_DATA
    """


def cache_mkdir(subdirname):
    """
    Create a subdirectory of PYTHOR3_DATA

    """
    raise NotImplementedError('')


def cache_rmdir(subdirname):
    """
    Remove a subdirectory of PYTHOR3_DATA

    """
    raise NotImplementedError('')


def cache_join(*names):
    """
    Return a path formed by $PYTHOR3_DATA/names[0]/names[1]/...
    """
    return os.path.join(PYTHOR3_DATA, *names)

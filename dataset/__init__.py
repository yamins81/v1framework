from os import path, environ, getenv

PYTHOR3_HOME = path.join(environ.get('HOME'), '.pythor3')

def get_home():
    pythor3_home = path.expanduser(environ.get('PYTHOR3_HOME', PYTHOR3_HOME))
    return pythor3_home

# -- root directory for Dataset and DatasetBlob objects to store data.
PYTHOR3_DATA = getenv("PYTHOR3_DATA",path.join(get_home(), 'data'))

# -- zero or more paths to supplement the PYTHOR3_DATA root when looking for
# existing files to re-open.  See cache_open() for documentation and usage.
PYTHOR3_DATA_ALT_READ = getenv("PYTHOR3_DATA_ALT_READ", "").split(':')

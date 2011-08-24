"""
Dataset interface and utilities.

A Dataset is essentially a set of named Columns (see column.py).

"""

import logging
_logger = logging.getLogger("pythor.dataset")

class VerificationError(Exception):
    """
    A dataset has failed a verification test.
    """


class DatasetBlob(object):
    """
    API for managing a dataset's footprint in the pythor cache.

    See data_cache.py for useful helper routines.

    This class is purely for documentation - it doesn't provide any
    functionality of its own.

    """
    def fetch(self):
        """
        Cache official data locally (i.e., download them).

        Not all datasets have a sensible interpretation for this, but enough do
        that this is in the base class API.
        """
        pass

    def load(self):
        """
        Parse official local files into derived local files, and load them into
        memory.

        This function may call fetch() if necessary.

        Implementers should feel free to save the result of parsing to a pickle
        file or something in the pythor data cache, and attempt to reload that
        file in future load() calls.

        The way to unload the result of this function from memory should be to
        delete this object and trigger the Python GC.  Of course, we can come
        back to this design point if necessary.
        """
        pass

    def usage_fetch(self):
        """ Size that would be freed by erase_fetch.
        """

    def usage_load(self):
        """ Size that would be freed by erase_load.
        """

    def usage(self):
        """ Size that would be freed by erase.
        """

    def erase_load(self):
        """
        Erase files created by load.  Future calls to load() should have to
        re-parse the data.
        """
        pass

    def erase_fetch(self):
        """
        Potentially erase [some] files created by fetch.

        This function should generally not also delete files created by load.
        This way, the sequence: "fetch(); load(); erase_fetch()" results in a
        minimal amount of data on disk that supports future load() calls.

        This function could also be called "compact" or something - it erases
        files not strictly necessary for the ongoing use of the data.

        """
        pass

    def erase(self):
        """Erase all files associated with this dataset. Attempt to return the
        file system to the state before the first fetch() was called.  This
        might involve calls to erase_fetch and erase_load, but it might also
        just rmdir a cache folder.
        """


class Dataset(object):
    """
    Interface-defining class that provides parsed data.

    A Dataset behaves approximately like a dictionary of Columns, which are the
    elements of the dataset.  Neither Datasets nor Columns are not meant to be
    mutable. This interface should not be used to modify this list or its
    contents.
    >>> dataset = Dataset(meta, {'meta':meta, 'data': data_column}, stats={})

    A Dataset can be indexed by an integer key. This will return a dictionary
    mapping column names to column elements at the position indicated.
    >>> dataset[5] # returns {'meta':meta[5], 'data':data[5]}

    A Dataset can also be indexed by a slice.
    >>> dataset[1:5] # return [{'meta':meta[1], 'data':data[1]},
    >>>              #         {'meta':meta[2], 'data':data[2]},
    >>>              #         {'meta':meta[3], 'data':data[3]},
    >>>              #         {'meta':meta[4], 'data':data[4]}]

    A Dataset can also be index using numpy-style advanced indexing, in which
    the idx is a list/tuple/array of integers.
    >>> dataset[(1,5)] # return [{'meta':meta[1], 'data':data[1]},
    >>>                #         {'meta':meta[5], 'data':data[5]}]

    A dataset is also meant to provide direct access to columns, and metadata.

    >>> dataset.meta[:] # retrieve all metadata dictionaries as numpy ndarray
    >>> dataset.meta[i] # return i'th object's metadata
    >>> dataset.meta[1:7] # return numpy array of objects' metadata
    >>> dataset.meta[[3,14,5]] # return numpy array of objects' metadata

    >>> dataset.columns['data'][:] # retrieve all objects as numpy ndarray
    >>> dataset.columns['data'][i] # return i'th object
    >>> dataset.columns['data'][1:7] # return numpy array of objects
    >>> dataset.columns['data'][[3,14,5]] # return numpy array of objects


    The dataset.meta property is meant to behave like a read-only numpy array
    whose elements are dictionaries.  Those dictionaries define
    dataset-dependent attributes of the objects in the corresponding position
    of the dataset.data container.  The meta data in each example is a
    dictionary, and the keys must be valid identifier strings (strings that
    could pass as python variable names) but the items can be anything.  (Note
    to implementers of Datasets - simple meta data item types are preferred
    because they are easier to persist via formats such as cPickle, sqlite, and
    mongodb.)

    Each dictionary in  dataset.meta is largely unstructured, except for any
    key that matches a key in dataset.columns.  The item associated with such a
    key is meta data for the value in the corresponding column.  So in our
    example, if meta[i] had a key 'data', then meta[i]['data'] would describe
    data[i].  The only key that is common to all meta data descriptor
    dictionaries is 'type', which is generally a string and indicates the
    structure of the meta data descriptor dictionary.  For example, if data[i]
    were an int, then meta[i]['data']['type'] would be 'int'. If it were a
    numpy ndarray, then the type would be 'ndarray', and there would be
    additional meta data keys 'dtype', and 'shape'.  The meta data
    corresponding to a column element should generally be computable from the
    column element itself, without other sources of information.  Several
    functions are provided (via infer_meta()) for inferring meta data from
    python objects, and new dataset developers are encouraged to provide
    others.

    Meta-data must be JSON-encodable. Serializing meta data, which may include
    hashes of data, is a way to create relatively compact long-term records of
    datasets.

    Datasets can be queried by meta data using MongoDB query syntax.  (See
    http://www.mongodb.org/display/DOCS/Advanced+Queries.) The direct result of
    querying is a list/array of matching integer positions, which can be used
    to index into either dataset, dataset.data or dataset.meta.

    """

    def __init__(self, columns, meta=None, stats=None):
        if meta is None:
            meta = columns['meta']
        self.meta = meta
        self.columns = columns
        self.stats = stats

    def __getitem__(self, idx):
        try:
            lookup = [(cname, col[idx]) for cname, col in self.columns.items()]
        except:
            _logger.error('Problem with column %s index %s' %(cname, idx))
            raise
        if isinstance(idx, int):  # TODO: does this catch e.g. numpy.uint8?
            # one dict, containing one example
            return dict(lookup)
        else:
            # list of dicts, each of which contains one example
            rval = [dict([(cname, col_idx[i]) for cname, col_idx in lookup])
                    for i in xrange(len(lookup[0][1]))]
            return rval

    def query(self, spec):
        """
        Return (data, meta) pairs for which `meta` matches specification
        `spec`.

        See Mongodb docs for documentation of query specification.
        http://www.mongodb.org/display/DOCS/Advanced+Queries
        """
        idx = self.query_idx(spec)
        return self[idx]

    def query_idx(self, spec):
        """Return an idx array or iterator for where spec matches meta.

        See Mongodb docs for documentation of query specification.
        http://www.mongodb.org/display/DOCS/Advanced+Queries
        """
        raise NotImplementedError('Dan - want to implement this?')

    def rebuild_meta(self, colname=None):
        """
        Recompute self.meta[colname] and self.stats[colname]

        If colname is None, it recomputes meta-data for all columns.
        """

        if colname is None:
            # recurse on each column
            for colname in self.columns:
                self.rebuild_meta(colname)
        else:
            col = self.columns[colname]
            if self.meta is None:
                self.meta = [{} for i in xrange(len(col))]
            if self.stats is None:
                self.stats = {}

            meta, stats = col.infer_stats()
            self.stats[colname] = stats
            assert len(meta) == len(self.meta)
            for self_meta_i, col_meta_i in zip(self.meta, meta):
                self_meta_i[colname] = col_meta_i

    def verify_meta(self, colname=None, typename=None, max_errs=0):
        """
        Raise exception if self.meta does not correspond to current meta data.

        This is useful if meta-data has been loaded rather than computed by
        self.rebuild_data().

        """

        if colname is None:
            # recurse on each column
            for colname in self.columns:
                self.verify_meta(colname)
        else:
            stats = self.stats[colname]
            meta = [m[colname] for m in self.meta]
            self.columns[colname].verify_stats(meta, stats)

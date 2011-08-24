"""
Provides Column objects that support the Dataset interface.

A column corresponds to an attribute of all elements of a dataset.

Numpy ndarrays are valid Columns, and they provide fast element access.  For
columns that are infinite and dynamically generated, other implementation
strategies can provide the Column API.  The Column API is intended to provide
an indexing mechanism similar to that of a read-only numpy ndarray.

ProxyColumns implement the Column API by transforming the elements of other
Column (or Columns). The transformation can be simple (e.g. reading a filename
to string) or complex (e.g. filtering an example through a classification
algorithm).  Columns provide a layer of abstraction similar to variables in the
S/R programming language.

Creating a column is intended to be a very "cheap" thing to do - no expensive
computation should take place.  Accessing elements in a column should (whenever
possible) imply no more computation than is proportional to the number of
elements accessed.

:TODO: organize classes by backend
    - mongo-backed
    - ndarray-backed
    - computation-on-anything
    - computation-on-ndarray
    - computation-on-mongo

:TODO: organize mixins by element type

:TODO: Consider adding a class (VirtualDatabase?) that is a container for
Columns and datasets. The purpose would be to forward-propagate changes, so that
Columns that act as caches can be brought up to date.  Alternatively we can
insist that Columns Don't Change.... but of course, they do.  If this class
comes into existance, we might rename Dataset -> Table to make the relation
between DB, Table/Dataset, and Column even more clear.  Tables can't usually
have overlapping sets of Columns, but whatever.

"""
import logging
import time
import itertools
import numpy as np
import scipy.misc
import Image
import ndarray_hash   # could be removed if necessary

_logger = logging.getLogger("pythor.column")


class Column(object):
    """
    Convenient proxy for lazy-evaluation of read-only Dataset fields.
    """

    def __init__(self, length):
        self._length = int(length)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if isinstance(idx, int):  # does this catch e.g. numpy.uint8?
            return self.get_int(idx)
        elif isinstance(idx, slice):
            return self.get_slice(idx)
        else:
            return self.get_array(idx)

    def get_int(self, int_idx):
        raise NotImplementedError('override-me')

    def get_slice(self, slice_idx):
        start, stop, step = slice_idx.indices(self._length)
        rval = [self.get_int(i) for i in xrange(start, stop, step)]
        return rval

    def get_array(self, array_idx):
        rval = [self.get_int(i) for i in array_idx]
        return rval

    def filter_idx(self, elem_accept, multi_accept=None):
        return [i for i,s_i in enumerate(self) if elem_accept(s_i)]

    def __array__(self):
        return np.asarray(self[:])


class ProxyColumn(Column):
    def __init__(self, col):
        self._col = col
        Column.__init__(self, len(col))

    def __len__(self):
        return len(self._col)

class NdarrayBacked(ProxyColumn):

    def __getitem__(self, idx):
        return self._col[idx]

    def __array__(self):
        return self._col

    def filter_idx(self, elem_accept, multi_accept=None):
        if multi_accept:
            return self._col.argwhere(multi_accept)
        return ProxyColumn.filter_idx(self, elem_accpet, multi_accept)


class PermutedColumn(Column):

    def __init__(self, col, permutation):
        Column.__init__(self, len(permutation))
        self._col = col
        self._permutation = permutation
        if isinstance(permutation, np.ndarray):
            if permutation.ndim != 1:
                raise TypeError('perm must be 1d array', permutation.shape)

    def __getitem__(self, idx):
        return self._col[self._permutation[idx]]

class ListColumn(ProxyColumn):

    def get_int(self, i):
        return self._col[i]

    def get_slice(self, slice_idx):
        return self._col.__getitem__(slice_idx)

    def __array__(self):
        return np.asarray(self._col)

class MapColumn(ProxyColumn):
    def __init__(self, col, fn=None):
        ProxyColumn.__init__(self, col)
        if fn:
            self._fn = fn

    def get_int(self, i):
        return self._fn(self._col[i])


class ColumnTypeMixin(object):
    typename = None  # subclasses should override

    def infer_meta(self, x):
        return {'type': self.typename}

    def infer_stats(self):

        # N.B. doesn't work when mc_i has different structure on each iter
        # N.B. doesn't work for nested constants
        constants = {}
        meta = []

        for val_i in self:
            meta_i = self.infer_meta(val_i)
            if not constants:
                constants.update(meta_i)
            elif len(constants) == 1:
                pass  # only 'type' is left
            else:
                for k, v in constants.items():
                    if meta_i[k] != v:
                        del constants[k]
            meta.append(meta_i)
        stats = dict(meta_constants=constants)
        return meta, stats

    def verify_meta(self, value, meta, stats):
        if meta != self.infer_meta(value):
            raise VerificationError()

    def verify_stats(self, meta, stats):
        constants = stats['meta_constants']
        if len(self) != len(meta):
            raise VerificationError('length mismatch')
        for v, m in zip(self, meta):
            self.verify_meta(v, m, stats)
            for k in constants:
                if constants[k] != m[k]:
                    raise VerificationError('meta violates constants',
                            (meta, constants))


class StrColumnMixin(ColumnTypeMixin):
    typename = 'str'

    def infer_meta(self, x):
        if type(x) != str: raise TypeError(x)
        return {'type': self.typename, 'value': x}

    # methods from base class should work.
    def verify_meta(self, value, meta, stats):
        ColumnTypeMixin.verify_meta(self, value, meta, stats)
        assert type(value) == str


class ScalarColumnMixin(ColumnTypeMixin):

    def __init__(self, typename):
        self.typename = typename
        self._npy_dtype = getattr(np, typename)

    def infer_meta(self, x):
        assert x == self._npy_dtype(x)
        return {'type': self.typename, 'value': x}

    def infer_stats(self):
        meta, stats = ColumnTypeMixin.infer_stats(self)
        stats['min'] = np.min(self)
        stats['max'] = np.max(self)
        return meta, stats

    def verify_meta(self, value, meta, stats):
        ColumnTypeMixin.verify_meta(self, value, meta, stats)
        assert type(value) == int
        if 'min' in stats:
            assert value >= stats['min']
        if 'max' in stats:
            assert value <= stats['max']

class ScalarColumn(ProxyColumn, ScalarColumnMixin):

    def __init__(self, col, typename):
        ProxyColumn.__init__(self, col)
        ScalarColumnMixin.__init__(self, typename)

    def __getitem__(self, item):
        return self._col[item]


class NdarrayColumnMixin(ColumnTypeMixin):
    typename = 'ndarray'

    def __init__(self, algo=ndarray_hash.default_algo):
        self._algo = algo

    def infer_meta(self, x):
        return dict(
                dtype=str(x.dtype),
                shape=x.shape,
                vhash=ndarray_hash.ndarray_hash(x, self._algo),
                )

    def verify_meta(self, value, meta, stats):
        assert isinstance(value, np.ndarray)
        if meta['dtype'] != str(value.dtype):
            raise VerificationError('dtype mismatch')
        if meta['shape'] != value.shape:
            raise VerificationError('shape mismatch')
        if 'vhash' in meta:
            key, algo = meta['vhash']
            if (key, algo) != ndarray_hash.ndarray_hash(x, algo):
                raise VerificationError('hash mismatch')


class PIL_ProxyColumn(MapColumn):
    _fn = Image.open


class AsarrayProxyColumn(MapColumn, NdarrayColumnMixin):
    _fn = np.asarray


class NdarrayFromImagepath(MapColumn, NdarrayColumnMixin):

    @staticmethod
    def _fn(x):
        img = Image.open(x)
        # workaround object array bug
        #rval = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
        #rval = scipy.misc.fromimage(img)
        rval = np.asarray(img)
        #img.close()
        #print rval.dtype, rval.shape
        return rval


class NdarrayColumn(ProxyColumn, NdarrayColumnMixin):
    """Column that is backed by an ndarray
    """
    def __init__(self, col):
        ProxyColumn.__init__(self, col)
        NdarrayColumnMixin.__init__(self)

    def __getitem__(self, key):
        return self._col[key]

    def __array__(self):
        return self._col


if 0:
    #
    # This function seems like it can't be made fast enough to be worthwhile,
    # and raw numpy can do the same thing.
    #
    def inner_join(primary, switch, n_cases, verify_no_missing=True):
        """
        Return a matrix (locs) of size M x n_cases.

        M is the number of unique elements in primary, which should be a column
        containing integer values 0 .. M-1, in which each integer appears at least
        n_cases times.

        locs[i, case] is the position in switch where the primary took value `i`
        and the switch took value `case`.  If switch gives value None then the
        example is skipped.

        """

        t0 = time.time()

        if len(primary) != len(switch):
            raise ValueError()

        M = len(set(primary))

        if M == np.int32(M):
            locs = np.zeros((M, n_cases), dtype='int32') - 1
            keys = np.zeros(M, dtype='int32')-1
        else:
            locs = np.zeros((M, n_cases), dtype='int64') - 1
            keys = np.zeros(M, dtype='int64')-1

        primary = np.asarray(primary)
        switch = np.asarray(switch)
        for i, (p_i, s_i) in enumerate(itertools.izip(primary, switch)):
            #print i, p_i, s_i
            if s_i is None:
                continue
            else:
                if locs[p_i, s_i] != -1:
                    raise ValueError('duplicate entry', (p_i, s_i, i))
                locs[p_i, s_i] = i
        if verify_no_missing:
            if locs.min() < 0:
                raise ValueError('missing entries')
        _logger.info('inner_join took %f seconds' % (time.time() - t0))
        return locs

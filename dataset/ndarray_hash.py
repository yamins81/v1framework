"""
Utility functions for verifying that meta-data and ndarray data are uncorrupted.
"""
import hashlib
import numpy


def ndarray_hash_given_arch_a1(arch, x):
    """
    Return the hash of numpy array using hashlib.sha1.

    The `arch` parameter indicates which architecture to assume this is running
    on.  Currently only little-endian (arch='LE') is implemented.
    """

    x = numpy.asarray(x)

    if arch != 'LE':
        # Figure out a bit-mangling transformation that
        # works on your architecture,
        # and use that instead of this method
        raise NotImplementedError()

    if 'int' in str(x.dtype):
        m = hashlib.sha1()
        to_hash = numpy.asarray(x, order='C')
        m.update(to_hash.data)
        m.update(str(x.dtype))
        m.update(str(x.shape))
        return (m.hexdigest(), 'a1')

    if 'float' in str(x.dtype):
        # algorithm sketch:
        # 1. allocate quasi-random numbers that are reproducible anywhere
        #    such as Q = numpy.arange(BIG) * prime1 % prime2
        # 2. flatten x, and compute the inner product with Q[:len(x)]
        # 3. for more certainty, allocate a few different vectors Q and 
        #    save all of the inner products as the hash/signature of the data
        raise NotImplementedError()
    if 'complex' in str(x.dtype):
        # re-use float algorithm.
        raise NotImplementedError()

    raise TypeError('non-integer dtype not supported', x.dtype)

this_arch = {
        ('e274a7126d5cbb2e2bed2f0dab8bae4e9f6f848e', 'a1'): 'LE',
        }[ndarray_hash_given_arch_a1('LE', [555])]
"""
this_arch is the name of the architecture running the process.

If your arch is not listed here, run ndarray_hash_given_arch_a1 in ipython and
paste the result in here.

If your arch is listed, but you get a different hash, something is wrong and
maybe the hashing algorithm needs fixing.
"""

# in future code releases, we have to always be able to
# provide all of the old hashing algorithms. The point of this function is
# to verify old results after all.
# Every time this code is modified in a way that changes a hash value that
# could have been computed, the new version must be only accessible by a new
# `algo` descriptor.

default_algo = 'a1'


def ndarray_hash(x, algo=default_algo):
    """
    Return the `(f(x), algo)` pair.

    ndarray_hash(x, a) will return the same thing on all platforms and in all
    future versions of this library.

    """
    assert algo=='a1'
    return ndarray_hash_given_arch_a1(this_arch, x)


def test_ndarray_hash():
    # this works
    assert ndarray_hash([2,3,4]) == ndarray_hash(numpy.asarray([2,3,4]))

    # hash is sensitive to ndim
    assert ndarray_hash([2,3,4]) != ndarray_hash(numpy.asarray([[2,3,4]]))

    # hash is sensitive to dtype
    assert ndarray_hash([2,3,4]) != ndarray_hash(numpy.asarray([2,3,4],dtype='uint8'))




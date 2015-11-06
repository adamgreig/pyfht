import ctypes
import random
import importlib
import numpy as np


# Set up ctypes function bindings
libpath = importlib.find_loader("libfht").path
libfht = ctypes.CDLL(libpath)
_fht = libfht.fht
_shuffle_bigger_lfsr = libfht.shuffle_bigger_lfsr
_shuffle_smaller_lfsr = libfht.shuffle_smaller_lfsr
_shuffle_bigger_o = libfht.shuffle_bigger_o
_shuffle_smaller_o = libfht.shuffle_smaller_o
u32 = ctypes.c_uint32
u64 = ctypes.c_uint64
ptype = np.ctypeslib.ndpointer(dtype=np.double, ndim=1)
u8array = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1)
u32array = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1)
_fht.argtypes = [u32, ptype]
_shuffle_bigger_lfsr.argtypes = [u32, ptype, u32, ptype, u32]
_shuffle_smaller_lfsr.argtypes = [u32, ptype, u32, ptype, u32]
_shuffle_bigger_o.argtypes = [u32, ptype, u32, ptype, u32array]
_shuffle_smaller_o.argtypes = [u32, ptype, u32, ptype, u32array]


def FHTlfsr(N, n):
    """
    Legacy LFSR based subsampled FHT
    """
    assert (N & (N-1)) == 0

    def Az(z, seed=1):
        """Computes A'.z, returns an Nx1 vector."""
        zc = np.zeros(N)
        _shuffle_bigger_lfsr(n, z.reshape(n), N, zc, seed)
        _fht(N, zc)
        return zc

    def Ab(beta, seed=1):
        """Computes A.b, returns an nx1 vector."""
        bc = beta.copy().reshape(N)
        _fht(N, bc)
        out = np.empty(n)
        _shuffle_smaller_lfsr(N, bc, n, out, seed)
        return out

    return Az, Ab


def FHTo(N, n):
    """
    Legacy ordering-specified subsampled FHT
    """
    assert (N & (N-1)) == 0

    def Az(z, seed=1):
        rng = random.Random(seed)
        order = np.array(rng.sample(range(1, N), n), dtype=np.uint32)
        zc = np.zeros(N)
        _shuffle_bigger_o(n, z.reshape(n), N, zc, order)
        _fht(N, zc)
        return zc

    def Ab(beta, seed=1):
        rng = random.Random(seed)
        order = np.array(rng.sample(range(1, N), n), dtype=np.uint32)
        bc = beta.copy().reshape(N)
        _fht(N, bc)
        out = np.empty(n)
        _shuffle_smaller_o(N, bc, n, out, order)
        return out

    return Az, Ab

FHT = FHTo




def fht(x):
    """
    Compute the Walsh-Hadamard transform of x,
    which must be a 1d array whose length is a power of 2.
    """

    assert len(x.shape) == 1, "x must be 1-dimensional"
    assert x.size != 0, "x must not be empty"
    assert x.size & (x.size - 1) == 0, "len(x) must be a power of 2"
    
    out = x.copy()
    _fht(out.size, out)
    return out


def sub_fht(n, m, seed=0, ordering=None):
    """
    Returns functions to compute the sub-sampled Walsh-Hadamard transform,
    i.e., operating with a wide rectangular matrix of random +/-1 entries.

    n: number of rows (smaller dimension), must be smaller than m
    m: number of columns (larger dimension), must be a power of two

    seed: determines choice of random matrix
    order: optional n-long array of row indices to implement subsampling;
           generated by seed if not specified, but may be given to speed up
           subsequent runs on the same matrix.

    Returns (Ax, Ay, ordering):
        Ax(x): computes A.x (of length n), with x having length m
        Ay(y): computes A'.y (of length m), with y having length n
        ordering: the ordering in use, which may have been generated from seed
    """
    assert n > 0
    assert m > 0
    assert n < m
    assert m & (m-1) == 0

    if ordering is not None:
        assert ordering.shape == (n,)
    else:
        rng = random.Random(seed)
        ordering = np.array(rng.sample(range(1, m), n), dtype=np.uint32)

    def Ax(x):
        assert x.size == m
        x = x.copy().reshape(m)
        _fht(m, x)
        return x[ordering]

    def Ay(y):
        assert y.size == n
        out = np.zeros(m)
        out[ordering] = y
        _fht(m, out)
        return out

    return Ax, Ay, ordering

def block_sub_fht(n, m, l, seed=0, ordering=None):
    """
    As `sub_fht`, but computes in `l` blocks of size `n` by `m`, potentially
    offering substantial speed improvements.

    n: number of rows, must be smaller than m
    m: number of columns per block, must be a power of two
    l: number of blocks

    seed: determines choice of random matrix
    ordering: optional (l, n) shaped array of row indices to implement
              subsampling; generated by seed if not specified, but may be
              given to speed up subsequent runs on the same matrix.

    Returns (Ax, Ay, ordering):
        Ax(x): computes A.x (of length n), with x having length l*m
        Ay(y): computes A'.y (of length l*m), with y having length n
        ordering: the ordering in use, which may have been generated from seed
    """
    assert n > 0
    assert m > 0
    assert l > 0
    assert n < m
    assert m & (m-1) == 0

    if ordering is not None:
        assert ordering.shape == (l, n)
    else:
        rng = random.Random(seed)
        ordering = np.empty((l, n), dtype=np.uint32)
        for ll in range(l):
            ordering[ll] = np.array(rng.sample(range(1, m), n))

    def Ax(x):
        assert x.size == l*m
        out = np.zeros(n)
        for ll in range(l):
            ax, ay, _ = sub_fht(n, m, ordering=ordering[ll])
            out += ax(x[ll*m:(ll+1)*m])
        return out

    def Ay(y):
        assert y.size == n
        out = np.empty(l*m)
        for ll in range(l):
            ax, ay, _ = sub_fht(n, m, ordering=ordering[ll])
            out[ll*m:(ll+1)*m] = ay(y)
        return out

    return Ax, Ay, ordering

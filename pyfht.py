import ctypes
import random
import importlib
import numpy as np

libpath = importlib.find_loader("libfht").path
libfht = ctypes.CDLL(libpath)
fht = libfht.fht
shuffle_bigger = libfht.shuffle_bigger
shuffle_smaller = libfht.shuffle_smaller
shuffle_bigger_xs = libfht.shuffle_bigger_xs
shuffle_smaller_xs = libfht.shuffle_smaller_xs
u32 = ctypes.c_uint32
u64 = ctypes.c_uint64
ptype = np.ctypeslib.ndpointer(dtype=np.double, ndim=1)
u8array = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1)
fht.argtypes = [u32, ptype]
shuffle_bigger.argtypes = [u32, ptype, u32, ptype, u32]
shuffle_smaller.argtypes = [u32, ptype, u32, ptype, u32]
shuffle_bigger_xs.argtypes = [u32, ptype, u32, ptype, u8array, u64, u64]
shuffle_smaller_xs.argtypes = [u32, ptype, u32, ptype, u8array, u64, u64]


def FHT(N, n):
    assert (N & (N-1)) == 0

    def Az(z, seed=1):
        """Computes A'.z, returns an Nx1 vector."""
        zc = np.zeros(N)
        shuffle_bigger(n, z.reshape(n), N, zc, seed)
        fht(N, zc)
        return zc

    def Ab(beta, seed=1):
        """Computes A.b, returns an nx1 vector."""
        bc = beta.copy().reshape(N)
        fht(N, bc)
        out = np.empty(n)
        shuffle_smaller(N, bc, n, out, seed)
        return out

    return Az, Ab


def FHTxs(N, n):
    assert (N & (N-1)) == 0

    def Az(z, seed=1):
        """Computes A'.z, returns an Nx1 vector."""
        rng = random.Random(seed)
        s0 = rng.getrandbits(64)
        s1 = rng.getrandbits(64)
        zc = np.zeros(N)
        used = np.zeros(n, dtype=np.uint8)
        shuffle_bigger_xs(n, z.reshape(n), N, zc, used, s0, s1)
        fht(N, zc)
        return zc

    def Ab(beta, seed=1):
        """Computes A.b, returns an nx1 vector."""
        rng = random.Random(seed)
        s0 = rng.getrandbits(64)
        s1 = rng.getrandbits(64)
        bc = beta.copy().reshape(N)
        used = np.zeros(n, dtype=np.uint8)
        fht(N, bc)
        out = np.empty(n)
        shuffle_smaller_xs(N, bc, n, out, used, s0, s1)
        return out

    return Az, Ab

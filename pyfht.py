import ctypes
import random
import importlib
import numpy as np

libpath = importlib.find_loader("libfht").path
libfht = ctypes.CDLL(libpath)
fht = libfht.fht
shuffle_bigger_lfsr = libfht.shuffle_bigger_lfsr
shuffle_smaller_lfsr = libfht.shuffle_smaller_lfsr
shuffle_bigger_o = libfht.shuffle_bigger_o
shuffle_smaller_o = libfht.shuffle_smaller_o
u32 = ctypes.c_uint32
u64 = ctypes.c_uint64
ptype = np.ctypeslib.ndpointer(dtype=np.double, ndim=1)
u8array = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1)
u32array = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=1)
fht.argtypes = [u32, ptype]
shuffle_bigger_lfsr.argtypes = [u32, ptype, u32, ptype, u32]
shuffle_smaller_lfsr.argtypes = [u32, ptype, u32, ptype, u32]
shuffle_bigger_o.argtypes = [u32, ptype, u32, ptype, u32array]
shuffle_smaller_o.argtypes = [u32, ptype, u32, ptype, u32array]


def FHTlfsr(N, n):
    assert (N & (N-1)) == 0

    def Az(z, seed=1):
        """Computes A'.z, returns an Nx1 vector."""
        zc = np.zeros(N)
        shuffle_bigger_lfsr(n, z.reshape(n), N, zc, seed)
        fht(N, zc)
        return zc

    def Ab(beta, seed=1):
        """Computes A.b, returns an nx1 vector."""
        bc = beta.copy().reshape(N)
        fht(N, bc)
        out = np.empty(n)
        shuffle_smaller_lfsr(N, bc, n, out, seed)
        return out

    return Az, Ab


def FHTo(N, n):
    assert (N & (N-1)) == 0

    def Az(z, seed=1):
        rng = random.Random(seed)
        order = np.array(rng.sample(range(1, N), n), dtype=np.uint32)
        zc = np.zeros(N)
        shuffle_bigger_o(n, z.reshape(n), N, zc, order)
        fht(N, zc)
        return zc

    def Ab(beta, seed=1):
        rng = random.Random(seed)
        order = np.array(rng.sample(range(1, N), n), dtype=np.uint32)
        bc = beta.copy().reshape(N)
        fht(N, bc)
        out = np.empty(n)
        shuffle_smaller_o(N, bc, n, out, order)
        return out

    return Az, Ab

FHT = FHTo

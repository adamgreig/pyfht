import ctypes
import importlib
import numpy as np


def FHT(N, n):
    assert (N & (N-1)) == 0
    libpath = importlib.find_loader("libfht").path
    libfht = ctypes.CDLL(libpath)
    ptype = np.ctypeslib.ndpointer(dtype=np.double, ndim=1)
    fht = libfht.fht
    shuffle_bigger = libfht.shuffle_bigger
    shuffle_smaller = libfht.shuffle_smaller
    u32 = ctypes.c_uint32
    fht.argtypes = [u32, ptype]
    shuffle_bigger.argtypes = [u32, ptype, u32, ptype, u32]
    shuffle_smaller.argtypes = [u32, ptype, u32, ptype, u32]

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

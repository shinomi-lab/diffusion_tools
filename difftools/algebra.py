import numpy as np
import numpy.random as nrd

from numba import njit
from numba.pycc import CC

cc = CC("algebra")


@cc.export("_make_simplex", "f8[:](i8, f8[:])")
@njit
def _make_simplex(n, s):
    m = n - 1
    r = np.zeros(n)
    r[0] = s[0]
    r[n - 1] = 1 - s[-1]
    for i in range(m - 1):
        r[i + 1] = s[i + 1] - s[i]
    return r


@cc.export("random_simplex", "f8[:](optional(i8), i8)")
@njit
def random_simplex(seed, n):
    if not seed is None:
        nrd.seed(seed)
    return _make_simplex(n, np.sort(nrd.random(n - 1)))


def random_simplex_gen(n, f):
    return _make_simplex(n, np.sort(f(n - 1)))

import difftools.diffusion as dd

import numpy as np
import numpy.random as nrd

import numba
from numba import njit
from numba.pycc import CC

cc = CC("maximization")


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


@cc.export("ic_dist_x", "i8[:](optional(i8), i8, i8[:,:], i8[:], f8[:,:])")
@njit
def ic_dist_x(seed, n, adj, S, prob_mat):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : a seed set as an index vector (1d {0, 1} array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)

    Returns
    -------
    An activated node vector
    """
    return dd.ic_mat(n, adj, S, prob_mat, seed)[0]


@cc.export(
    "icu_dist_x",
    "f8[:](optional(i8), i8, i8[:,:], i8[:], f8[:,:], f8[:])",
)
@njit
def icu_dist_x(seed, n, adj, S, prob_mat, util_dist):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : a seed set as an index vector (1d {0, 1} array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    util_dist : an utility distribusion (1d float64 array)

    Returns
    -------
    A gained utility distribution, a vector of which i-th component is i's utility if i is activated and 0 otherwise
    """
    d = dd.ic_mat(n, adj, S, prob_mat, seed)[0]
    return d * util_dist


@cc.export("ic_dist", "(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:])")
@njit
def ic_dist(seed, m, n, adj, S, prob_mat):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    m : an iteration count of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : a seed set as an index vector (1d {0, 1} array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)

    Returns
    -------
    An active rate vector (a node distribution of the rate of actived times in m iterations)
    """
    d = np.zeros(n, np.float64)
    if not seed is None:
        nrd.seed(seed)
    for i in numba.prange(m):
        d += ic_dist_x(None, n, adj, S, prob_mat)
    dist = d / m
    return dist


@cc.export(
    "icu_dist",
    "(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:], f8[:])",
)
@njit
def icu_dist(seed, m, n, adj, S, prob_mat, util_dist):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    m : an iteration count of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : a seed set as an index vector (1d {0, 1} array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    util_dist : an utility distribusion (1d float64 array)

    Returns
    -------
    An average of m gained utility distributions (= Hadamard product of an utilty distribution and an active rate vector)
    """
    return ic_dist(seed, m, n, adj, S, prob_mat) * util_dist


@cc.export(
    "ic_sigma",
    "f8(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:])",
)
@njit
def ic_sigma(seed, m, n, adj, S, prob_mat):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    m : an iteration count of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : a seed set as an index vector (1d {0, 1} array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)

    Returns
    -------
    An average of the number of activated nodes
    """
    return ic_dist(seed, m, n, adj, S, prob_mat).sum()


@cc.export(
    "icu_sigma",
    "f8(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:], f8[:])",
)
@njit
def icu_sigma(seed, m, n, adj, S, prob_mat, util_dist):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    m : an iteration count of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : a seed set as an index vector (1d {0, 1} array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    util_dist : an utility distribusion (1d float64 array)

    Returns
    -------
    An average of the sum of gained utilities
    """
    return icu_dist(seed, m, n, adj, S, prob_mat, util_dist).sum()


@cc.export("im_greedy", "(optional(i8), i8, i8, i8, i8[:,:], f8[:,:])")
@njit
def im_greedy(seed, k, m, n, adj, prob_mat):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    k : the maximum size of seed node set
    m : sampling size of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)

    Returns
    -------
    The tuple of an optimal seed set and the history of influence distribusions
    """

    V = np.repeat(1, n)
    S = np.zeros(n, dtype=np.int64)

    hist = np.zeros((k, n), dtype=np.float64)

    for j in range(k):
        s_dist = np.zeros(n)
        W = V - S
        for i in range(n):
            if W[i] != 0:
                Su = S.copy()
                Su[i] = 1
                s_dist[i] = ic_sigma(seed, m, n, adj, Su, prob_mat)

        S[s_dist.argmax()] = 1
        hist[j] = s_dist

    return (S, hist)


@cc.export(
    "um_greedy",
    "(optional(i8), i8, i8, i8, i8[:,:], f8[:,:], f8[:])",
)
@njit
def um_greedy(seed, k, m, n, adj, prob_mat, util_dist):
    """
    Parameters
    ----------
    seed : a random seed set initially unless it is None
    k : the maximum size of seed node set
    m : sampling size of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    util_dist : an utility distribusion (1d float64 array)

    Returns
    -------
    The tuple of an optimal seed set and the history of obtained utility distribusions
    """

    V = np.repeat(1, n)
    S = np.zeros(n, dtype=np.int64)

    hist = np.zeros((k, n), dtype=np.float64)

    for j in range(k):
        s_dist = np.zeros(n)
        W = V - S
        for i in range(n):
            if W[i] != 0:
                Su = S.copy()
                Su[i] = 1
                s_dist[i] = icu_sigma(seed, m, n, adj, Su, prob_mat, util_dist)

        S[s_dist.argmax()] = 1
        hist[j] = s_dist

    return (S, hist)

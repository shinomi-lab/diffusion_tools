import difftools.diffusion as diff

import networkx as nx
import numpy as np
import numpy.random as nrd
import matplotlib.pyplot as plt

import sys
import numba

from joblib import Parallel, delayed, parallel_backend


@numba.njit("float64[:](int64, float64[:])")
def _make_simplex(n, s):
    m = n - 1
    r = np.zeros(n)
    r[0] = s[0]
    r[n - 1] = 1 - s[-1]
    for i in range(m - 1):
        r[i + 1] = s[i + 1] - s[i]
    return r


@numba.njit("float64[:](optional(int64), int64)")
def random_simplex(seed, n):
    if not seed is None:
        nrd.seed(seed)
    return _make_simplex(n, np.sort(nrd.random(n - 1)))


def random_simplex_gen(n, f):
    return _make_simplex(n, np.sort(f(n - 1)))


@numba.njit("(optional(int64), int64, int64[:,:], int64[:], float64[:,:])")
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
    return diff.ic_mat(n, adj, S, prob_mat, seed)[0]


@numba.njit("(optional(int64), int64, int64[:,:], int64[:], float64[:,:], float64[:])")
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
    d = diff.ic_mat(n, adj, S, prob_mat, seed)[0]
    return d * util_dist


@numba.njit(
    "(optional(int64), int64, int64, int64[:,:], int64[:], float64[:,:])", parallel=True
)
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


@numba.njit(
    "(optional(int64), int64, int64, int64[:,:], int64[:], float64[:,:], float64[:])"
)
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


@numba.njit(
    "float64(optional(int64), int64, int64, int64[:,:], int64[:], float64[:,:])"
)
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


@numba.njit(
    "float64(optional(int64), int64, int64, int64[:,:], int64[:], float64[:,:], float64[:])"
)
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


@numba.njit("(optional(int64), int64, int64, int64, int64[:,:], float64[:,:])")
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


@numba.njit(
    "(optional(int64), int64, int64, int64, int64[:,:], float64[:,:], float64[:])"
)
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


@numba.njit(
    "(int64, int64, int64, int64[:,:], float64[:,:], float64[:,:])", parallel=False
)
def trial_with_sample_jit(k, m, n, adj, prob_mat, util_dists):
    """
    Parameters
    ----------
    k : the maximum size of seed node set
    m : sampling size of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    util_dists : utility sample list (2d float64 array)

    Returns
    -------
    The tuple of:
        a list of the total utility by IC with IM opt seed
        a list of the optimal max utility with each utility sample
        an opt-seed by influence maximization
        a history of influence maximization
        a list of utility vectors,
        an opt-seed list by utility maximization
        a list of a history of utility maximization
    """
    S, im_hist = im_greedy(None, k, m, n, adj, prob_mat)
    l = len(util_dists)
    Ts = np.zeros((l, n), dtype=np.int64)
    um_hists = np.zeros((l, k, n))
    total_utils = np.zeros(l)
    max_utils = np.zeros(l)

    for i in numba.prange(l):
        util_dist = util_dists[i]

        T, um_hist = um_greedy(None, k, m, n, adj, prob_mat, util_dist)
        Ts[i] = T
        um_hists[i] = um_hist

        total_utils[i] = icu_sigma(None, m, n, adj, S, prob_mat, util_dist)
        max_utils[i] = icu_sigma(None, m, n, adj, T, prob_mat, util_dist)

    return (
        total_utils,
        max_utils,
        S,
        im_hist,
        util_dists,
        Ts,
        um_hists,
    )


@numba.njit("(int64, int64, int64, int64, int64[:,:], float64[:,:])", parallel=False)
def trial_jit(l, k, m, n, adj, prob_mat):
    """
    Parameters
    ----------
    l : sampling size of utilities
    k : the maximum size of seed node set
    m : sampling size of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)

    Returns
    -------
    The tuple of:
        a list of the total utility by IC with IM opt seed
        a list of the optimal max utility with each utility sample
        an opt-seed by influence maximization
        a history of influence maximization
        a list of utility vectors,
        an opt-seed list by utility maximization
        a list of a history of utility maximization
    """
    util_dists = np.zeros((l, n), dtype=np.float64)
    for i in range(l):
        util_dists[i] = random_simplex(None, n)

    return trial_with_sample_jit(k, m, n, adj, prob_mat, util_dists)


def trial(l, k, m, n, adj, prob_mat):
    """
    Parameters
    ----------
    l : sampling size of utilities
    k : the maximum size of seed node set
    m : sampling size of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)

    Returns
    -------
    The dictionary as:
        `total-utils`: a list of the total utility by IC with IM opt seed
        `max-utils`: a list of the optimal max utility with each utility sample
        `im-seed`: an opt-seed by influence maximization
        `im-hist`: a history of influence maximization
        `utils`: a list of utility vectors,
        `um-seeds`: an opt-seed list by utility maximization
        `um-hists`: a list of a history of utility maximization
    """

    ret = trial_jit(l, k, m, n, adj, prob_mat)

    return {
        "total-utils": ret[0],
        "max-utils": ret[1],
        "im-seed": ret[2],
        "im-hist": ret[3],
        "utils": ret[4],
        "um-seeds": ret[5],
        "um-hists": ret[6],
    }


def trial_with_sample(k, m, n, adj, prob_mat, util_dists):
    """
    Parameters
    ----------
    k : the maximum size of seed node set
    m : sampling size of influence functions
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    util_dists : utility sample list (2d float64 array)

    Returns
    -------
    The dictionary as:
        `total-utils`: a list of the total utility by IC with IM opt seed
        `max-utils`: a list of the optimal max utility with each utility sample
        `im-seed`: an opt-seed by influence maximization
        `im-hist`: a history of influence maximization
        `utils`: a list of utility vectors,
        `um-seeds`: an opt-seed list by utility maximization
        `um-hists`: a list of a history of utility maximization
    """

    ret = trial_with_sample_jit(k, m, n, adj, prob_mat, util_dists)

    return {
        "total-utils": ret[0],
        "max-utils": ret[1],
        "im-seed": ret[2],
        "im-hist": ret[3],
        "utils": ret[4],
        "um-seeds": ret[5],
        "um-hists": ret[6],
    }


def plot_samples(total_utils, max_utils):
    fig = plt.figure(dpi=100)
    fig.patch.set_facecolor("white")

    x_max = max_utils.max() + 0.02
    x_min = max_utils.min() - 0.02

    ax = fig.add_subplot(111)
    ax.scatter(total_utils, max_utils, s=10, label="utility sample")

    x = np.linspace(x_min, x_max, 100)
    ax.plot(x, x, c="r", linewidth=0.5, label="$y=x$")

    ax.set_xlabel("Total Utility of Optimal Max Influence Range")
    ax.set_ylabel("Optimal Max Utility")
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    plt.show()

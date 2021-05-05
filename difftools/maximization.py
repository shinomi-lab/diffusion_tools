from typing import Tuple, Any, Set, List, Optional
import difftools.diffusion as dd
import difftools.algebra as da

import numpy as np
import numpy.random as nrd
import numpy.typing as npt

import numba
from numba import njit
from numba.pycc import CC

cc = CC("maximization")


@cc.export("ic_infl_prop", "i8[:](optional(i8), i8, i8[:,:], i8[:], f8[:,:])")
@njit
def ic_infl_prop(
    seed: Optional[int], n: int, adj: np.ndarray, S: np.ndarray, prob_mat: np.ndarray
) -> np.ndarray:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        S : the indicator of a seed set ($\\{0,1\\}^n$) as 1d int64 array.
            Let $V=\\{1, ..., n\\}$ and a seed set $S \\subseteq V$,
            the indicator of $S$ is a $n$-dimentional binary vector where $i$-th component equals 1 if $i \\in V$ otherwise 0.
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray

    Returns:
        the indicator of an activated node set
    """
    return dd.ic_adjmat(n, adj, S, prob_mat, seed)[0]


@cc.export(
    "ic_util_prop",
    "f8[:](optional(i8), i8, i8[:,:], i8[:], f8[:,:], f8[:])",
)
@njit
def ic_util_prop(
    seed: Optional[int],
    n: int,
    adj: np.ndarray,
    S: np.ndarray,
    prob_mat: np.ndarray,
    util_dist: np.ndarray,
) -> np.ndarray:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        S : the indicator of a seed set ($\\{0,1\\}^n$) as 1d int64 array.
            Let $V=\\{1, ..., n\\}$ and a seed set $S \\subseteq V$,
            the indicator of $S$ is a $n$-dimentional binary vector where the $i$-th component equals 1 if $i \\in V$ otherwise 0.
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray
        util_dist : an utility distribusion on the indicator of $V$ as 1d float64 array

    Returns:
        the utility distribution on the indicator of an activated node set
        where $i$-th component equals the utility of $i$ if $i$ is activated otherwise 0
    """
    d = dd.ic_adjmat(n, adj, S, prob_mat, seed)[0]
    return d * util_dist


@cc.export("ic_infl_prop_exp", "(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:])")
@njit
def ic_infl_prop_exp(
    seed: Optional[int],
    m: int,
    n: int,
    adj: np.ndarray,
    S: np.ndarray,
    prob_mat: np.ndarray,
) -> np.ndarray:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        S : the indicator of a seed set ($\\{0,1\\}^n$) as 1d int64 array.
            Let $V=\\{1, ..., n\\}$ and a seed set $S \\subseteq V$,
            the indicator of $S$ is a $n$-dimentional binary vector where $i$-th component equals 1 if $i \\in V$ otherwise 0.
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray

    Returns:
        An activation ratio distribution on $V$
        where $i$-th component is the average times that $i$ is activated in m iterations
    """
    d = np.zeros(n, np.float64)
    if not seed is None:
        nrd.seed(seed)
    for i in numba.prange(m):
        d += ic_infl_prop(None, n, adj, S, prob_mat)
    dist = d / m
    return dist


@cc.export(
    "ic_util_prop_exp",
    "f8[:](optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:], f8[:])",
)
@njit
def ic_util_prop_exp(
    seed: Optional[int],
    m: int,
    n: int,
    adj: np.ndarray,
    S: np.ndarray,
    prob_mat: np.ndarray,
    util_dist: np.ndarray,
) -> np.ndarray:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        S : the indicator of a seed set ($\\{0,1\\}^n$) as 1d int64 array.
            Let $V=\\{1, ..., n\\}$ and a seed set $S \\subseteq V$,
            the indicator of $S$ is a $n$-dimentional binary vector where $i$-th component equals 1 if $i \\in V$ otherwise 0.
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray
        util_dist : an utility distribusion on the indicator of $V$ as 1d float64 array

    Returns:
        An average utility distribution on $V$
        where $i$-th component is the utility of $i$ times the activation ratio
        (which is equal to the Hadamard product of the utilty distribution and an activation ratio ditribution)
    """
    return ic_infl_prop_exp(seed, m, n, adj, S, prob_mat) * util_dist


@cc.export(
    "ic_infl_sprd",
    "f8(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:])",
)
@njit
def ic_infl_sprd_exp(
    seed: Optional[int],
    m: int,
    n: int,
    adj: np.ndarray,
    S: np.ndarray,
    prob_mat: np.ndarray,
) -> np.number:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        S : the indicator of a seed set ($\\{0,1\\}^n$) as 1d int64 array.
            Let $V=\\{1, ..., n\\}$ and a seed set $S \\subseteq V$,
            the indicator of $S$ is a $n$-dimentional binary vector where the $i$-th component equals 1 if $i \\in V$ otherwise 0.
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray

    Returns:
        The average number of activated nodes
    """
    return ic_infl_prop_exp(seed, m, n, adj, S, prob_mat).sum()


@cc.export(
    "ic_sw_sprd_exp",
    "f8(optional(i8), i8, i8, i8[:,:], i8[:], f8[:,:], f8[:])",
)
@njit
def ic_sw_sprd_exp(
    seed: Optional[int],
    m: int,
    n: int,
    adj: np.ndarray,
    S: np.ndarray,
    prob_mat: np.ndarray,
    util_dist: np.ndarray,
) -> np.number:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        S : the indicator of a seed set ($\\{0,1\\}^n$) as 1d int64 array.
            Let $V=\\{1, ..., n\\}$ and a seed set $S \\subseteq V$,
            the indicator of $S$ is a $n$-dimentional binary vector where the $i$-th component equals 1 if $i \\in V$ otherwise 0.
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray
        util_dist : an utility distribusion on the indicator of $V$ as 1d float64 array

    Returns:
        The average of social welfare ($\\mathbb{E}\\left[\\sum_{i \\in V}u_i\\right] = \\sum_{i \\in V}\\mathbb{E}\\left[u_i\\right]$)
    """
    return ic_util_prop_exp(seed, m, n, adj, S, prob_mat, util_dist).sum()


@cc.export("im_greedy", "(optional(i8), i8, i8, i8, i8[:,:], f8[:,:])")
@njit
def im_greedy(
    seed: Optional[int], k: int, m: int, n: int, adj: np.ndarray, prob_mat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        k : the maximum size of seed node sets
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray

    Returns:
        The tuple of:

        - an IM near optimal seed set
        - the $k$-length list of 1d $n$-length vectors
            where the $i$-th component of the $j$-th vector element is the influence average of a seed set $S_j \\cup \\{i\\}$
            where $S_j$ is a near optimal $j$-size set
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
                s_dist[i] = ic_infl_sprd_exp(seed, m, n, adj, Su, prob_mat)

        S[s_dist.argmax()] = 1
        hist[j] = s_dist

    return S, hist


@cc.export(
    "swm_greedy",
    "(optional(i8), i8, i8, i8, i8[:,:], f8[:,:], f8[:])",
)
@njit
def swm_greedy(
    seed: Optional[int],
    k: int,
    m: int,
    n: int,
    adj: np.ndarray,
    prob_mat: np.ndarray,
    util_dist: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        seed : a seed value to initialize RNG unless None given
        k : the maximum size of seed node sets
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray
        util_dist : an utility distribusion on the indicator of $V$ as 1d float64 array

    Returns:
        The tuple of:

        - the indicator of an SWM near optimal seed set
        - the $k$-length list of 1d $n$-length vectors
            where the $i$-th component of the $j$-th vector element is the social welfare average of a seed set $S_j \\cup \\{i\\}$
            where $S_j$ is a near optimal $j$-size set
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
                s_dist[i] = ic_sw_sprd_exp(seed, m, n, adj, Su, prob_mat, util_dist)

        S[s_dist.argmax()] = 1
        hist[j] = s_dist

    return S, hist

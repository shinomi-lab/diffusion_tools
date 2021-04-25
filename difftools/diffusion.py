from typing import Tuple, Any, Set, List, Optional
import numpy as np
import math
import networkx as nx
import numpy.random as nrd

from numba.pycc import CC
from numba import njit

cc = CC("diffusion")

NDArray = Any


def get_gen(seed):
    sq = nrd.SeedSequence(seed)
    return nrd.Generator(nrd.PCG64(sq))


def information_vitality(p, i) -> float:
    return np.inner(p, i)


def information_entropy(p) -> float:
    return np.sum(np.ma.log(p) * p)


def normal_information_potential(p0, p1, i) -> float:
    dt = np.inner(p1 - p0, i)

    if dt == 0:
        return 0

    def f(p: float) -> float:
        if p == 0 or p == 1:
            return 0
        else:
            return p * p * math.log(p)

    def v(p: Tuple[Any, Any]) -> float:
        p0 = p[0]
        p1 = p[1]

        if p0 == p1:
            if p0 == 0:
                return 0
            elif p0 == 1:
                return 1
            else:
                return p0 * (1 + 2 * math.log(p0))
        else:
            return (f(p1) - f(p0)) / (p1 - p0)

    return dt * (1.0 - sum(map(v, zip(p0, p1))))


def potentials(p0, p1, i):
    return np.fromiter(
        map(lambda j: normal_information_potential(p0[j], p1[j], i[j]), range(4)),
        "float",
    )


def mapping(score, patterns):
    rank = np.argsort(np.argsort(score)) / score.shape[0]
    n = patterns.shape[0]
    p = {}
    for i in range(n):
        p[i] = rank >= (n - 1 - i) / n

    ret = p[0] * patterns[0]
    for i in range(1, n):
        ret += (~p[i - 1] & p[i]) * patterns[i]

    return ret


def single_source(n, i):
    v = np.zeros(n, bool)
    v[i] = True
    return v


def diffuse(adj, sender_vec, s0) -> List[Tuple[NDArray, NDArray]]:
    adjd = adj * sender_vec

    rs = [(s0, s0)]
    cs = s0
    ct = s0

    while not all(cs == 0):
        cs = (np.matmul(adjd, cs) > 0) & (~ct)
        ct = cs + ct
        rs.append((cs, ct))

    return rs


def diffuse_with_dq(
    n, adj, dq_vec, source, gen
) -> Tuple[List[Tuple[NDArray, NDArray]], NDArray, NDArray]:
    s0 = single_source(n, source)
    theta_vec = np.asarray(gen.random(n))  # thresholds(size, gen)
    sender_vec = dq_vec > theta_vec
    sender_vec[source] = True

    return diffuse(adj, sender_vec, s0), sender_vec, theta_vec


def independent_cascade(g, I0, ep_map, gen) -> List[Tuple[Set[int], Set[int]]]:
    """
    Parameters
    ----------
    g : networkx graph
    I0 : seed node set
    ep_map : propagation probability map
    gen : numpy random generator

    Returns
    -------
    The history of tuple of active node set and activated node set
    """
    G = g if g.is_directed() else g.to_directed()

    # gen = nrd.Generator(nrd.PCG64(nrd.SeedSequence(seed)))

    I = I0
    S = I.copy()

    ss = [(I, S)]
    while len(I) > 0:
        J = set()
        for i in I:
            for j in G.successors(i):
                if j in S:
                    continue
                if ep_map[(i, j)] > gen.random():
                    J.add(j)
        I = J
        S = S | I
        ss.append((I, S))

    return ss


@cc.export(
    "ic_adjmat",
    "(i8, i8[:,:], i8[:], f8[:,:], optional(i8))",
)
@njit
def ic_adjmat(
    n: int, adj: np.ndarray, S: np.ndarray, prob_mat: np.ndarray, seed: Optional[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    n : the number of nodes
    adj : the adjacency matrix of a graph as 2d int64 ndarray
    S : the indicator of a seed set (${0,1}^n$) as 1d int64 array
       Let $V={1, ..., n}$ and a seed set $S \\subseteq V$,
       the indicator of $S$ is a $n$-dimentional binary vector where the $i$-th component equals 1 if $i \\in V$ otherwise 0.
    prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray
    seed : a seed value to initialize RNG unless None given

    Returns
    -------
    The tuple of:
        - the indicator of an activated node set
        - the history of active node indicators
        - the history of activated node indicators
    """

    I = S.astype(np.int64)  # active node group (currently activated)
    T = S.astype(np.int64)  # total active node group (activated)

    # history
    hist = [(I, T)]

    if not seed is None:
        nrd.seed(seed)

    while np.count_nonzero(I) > 0:
        # make a new active group with a current one
        J = np.zeros(n, np.int64)  # init a new active group
        for i in range(n):
            if I[i] == 0:
                continue
            # i is in a current active group
            for j in range(n):
                if T[j] == 1:
                    continue
                if J[j] == 1:
                    continue
                # j is not active yet
                if adj[i, j] == 0:
                    continue
                # j is a successor of i
                # note: a_ij = 0 if and only if an edge (i, j) does not exist in a graph
                if (
                    prob_mat[i, j] > nrd.random()
                ):  # j should not be activated if p = 0, and j should be acitvated if p = 1
                    # activate j with a probability as prob_mat[i, j]
                    J[j] = 1

        # replace an active group to new one
        I = J.astype(np.int64)
        # add new active nodes to the total group
        # note: for all j, J_j = 1 & T_j = 0, so every component of T + I (= T + J) is at most 1
        T += I
        hist.append((I, T))

    l = len(hist)
    Is = np.zeros((l, n), np.int64)
    Ts = np.zeros((l, n), np.int64)
    for i, h in enumerate(hist):
        Is[i] = h[0]
        Ts[i] = h[1]

    return T, Is, Ts

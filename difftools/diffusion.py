from typing import Tuple, Any, Set, List
import numpy as np
import math
import networkx as nx
import numpy.random as nrd

import numba

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
        map(lambda j: normal_information_potential(p0[j], p1[j], i[j]),
            range(4)), 'float')


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

    while (not all(cs == 0)):
        cs = (np.matmul(adjd, cs) > 0) & (~ct)
        ct = cs + ct
        rs.append((cs, ct))

    return rs


def diffuse_with_dq(n, adj, dq_vec, source, gen) -> Tuple[List[Tuple[NDArray, NDArray]], NDArray, NDArray]:
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

@numba.njit('(int64, int64[:,:], int64[:], float64[:,:], optional(int64))')
def ic_mat(n, adj, S, prob_mat, seed):
    """
    Parameters
    ----------
    n : the number of nodes
    adj : adjacency matrix of a grpah as numpy matrix (2d int64 array)
    S : the numpy array form of a seed set
    prob_mat : propagation probabilities as adjacency matrix form (2d float64 array)
    seed : a random seed set initially unless it is None

    Returns
    -------
    The history of tuple of active node vector and activated node vector
    """

    I = S.astype(np.int64)
    S = S.astype(np.int64)

    ss = [(I, S)]

    if not seed is None: nrd.seed(seed)

    while np.count_nonzero(I) > 0:
        J = np.zeros(n, np.int64)
        
        for i in range(n):
            if I[i] == 0: continue
            for j in range(n):
                if adj[i, j] == 0: continue
                if S[j] == 1: continue
                if prob_mat[i, j] > nrd.random(): # gen.random():
                    J[j] = 1
        I = J.astype(np.int64)
        S += I
        ss.append((I, S))

    return S, ss

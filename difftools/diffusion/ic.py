from typing import Tuple, Any, Set, List, Optional
import numpy as np
import math
import networkx as nx
import numpy.random as nrd
from enum import Enum

from numba.pycc import CC
from numba import njit
import numba

cc = CC("ic")

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
        hist.append((I, T.copy()))

    l = len(hist)
    Is = np.zeros((l, n), np.int64)
    Ts = np.zeros((l, n), np.int64)
    for i, h in enumerate(hist):
        Is[i] = h[0]
        Ts[i] = h[1]

    return T, Is, Ts

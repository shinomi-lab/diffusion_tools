from typing import Tuple, Any, Set, List, Dict, Optional
import numpy as np
import numpy.typing as npt
import math
import networkx as nx
import numpy.random as nrd

from numba.pycc import CC
from numba import njit
import numba
import numba.types as nt
import numba.typed as ntd

from ._types import *

cc = CC("_jit")


@cc.export(
    "run_markov",
    (numba.i8[:,:,:], numba.f8[:], numba.f8[:,:], numba.i8, numba.i8),
)
@njit
def run_markov(
    stt: npt.NDArray[np.int8], ma: npt.NDArray[np.float64], mp: npt.NDArray[np.float64], s: np.int64, b: np.int64
) -> Tuple[np.int64, np.float64]:
    et = stt[s][b]
    e = et[0]
    t = et[1]
    if e < 0:
        return t, 0
    if ma[e] == 0:
        c = InfoType_F
    elif ma[e] == 1 or ma[e] > nrd.random():
        c = InfoType_T
    else:
        c = InfoType_F
    return t, mp[e][c]


@cc.export(
    "count_nonzero",
    (numba.i8[:,:], numba.i8[:]),
)
@njit
def count_nonzero(I_multi: npt.NDArray[np.int64], InfoTypes: npt.NDArray[np.int64]) -> np.int64:
    c = 0
    for it in InfoTypes:
        c += np.count_nonzero(I_multi[it])
    return c
    
@cc.export(
    "multi_ic_adjmat",
    (numba.i8, numba.i8[:,:], numba.i8[:,:], numba.i8[:,:,:], numba.f8[:,:], numba.f8[:,:,:], nt.optional(numba.i8)),
)
@njit
def multi_ic_adjmat(
    n: int, adj: npt.NDArray[np.int64], S_multi: npt.NDArray[np.int64],
    stt: npt.NDArray[np.int64], asm_maps: npt.NDArray[np.float64], prp_maps: npt.NDArray[np.float64],
    seed: Optional[int]
) -> Tuple[Dict[str, npt.NDArray[np.int64]], Dict[str, npt.NDArray[np.int64]], Dict[str, npt.NDArray[np.int64]]]:
    """
    Parameters
    ----------
    n : the number of nodes
    adj : the adjacency matrix of a graph as 2d int64 ndarray
    S_multi : the list of seed set indicators of each type of information as 1d int64 array
    stt: state transition table
    asm_maps: 
    prp_maps: 
    seed : a seed value to initialize RNG unless None given
    Note
    ----
    Let $V={1, ..., n}$.
    For any seed set $S \\subseteq V$, the seed set indicator of $S$ is
    a $n$-dimentional binary vector where the $i$-th component equals 1 if $i \\in V$ otherwise 0.
    Returns
    -------
    The tuple of:
        - the indicator of an activated node set
        - the history of active node indicators
        - the history of activated node indicators
    """
    InfoTypes = make_InfoTypes()

    if not seed is None:
        nrd.seed(seed)

    # The pair of active node groups (currently activated)
    I = S_multi.copy()

    # The pair of total active node groups (activated)
    T = I.copy()

    # history
    hist = [(I.copy(), T.copy())]

    # state list for markov models   
    s = np.zeros(n, np.int8)
    
    while count_nonzero(I, InfoTypes) > 0:
        # init new active groups
        J = np.zeros((InfoTypes_n, n), np.int64)
        
        # iterate user i from a current active group
        for i in range(n):
            # make the sequence of received information
            rs = np.zeros(0, np.int64)
            for it in InfoTypes:
                if I[it][i] == 1:
                    rs = np.append(rs, it)

            nrs = len(rs)

            if nrs == 0:
                continue
            # if user i received multi types information simultaneously 
            elif nrs > 1:
                # shuffle the order of the types of information at random
                nrd.shuffle(rs)

            for b in rs:
                # Run i's markov model
                t, p = run_markov(stt, asm_maps[i], prp_maps[i], s[i], b)
                s[i] = t
    
                if p == 0:
                    continue
    
                for j in range(n):
                    # j should be a successor of i and not active
                    # note: a_ij = 0 if and only if an edge (i, j) does not exist in a graph
                    if adj[i, j] == 0 or T[b][j] == 1 or J[b][j] == 1:
                        continue

                    # j should not be activated if p = 0, and j should be acitvated if p = 1
                    if p == 1 or p > nrd.random():
                        # activate j with a probability
                        J[b][j] = 1

        # replace old active groups to new ones
        I = J.copy() #.astype(np.int64)
        # add new active nodes to the total group
        # note: for all j, the proposition J_j = 1 & T_j = 0 holds, so every component of T + J is at most 1
        T += J

        hist.append((I.copy(), T.copy()))

    l = len(hist)
    Is = np.zeros((InfoTypes_n, l, n), dtype=np.int64)
    Ts = np.zeros((InfoTypes_n, l, n), dtype=np.int64)

    for it in InfoTypes:
        for i, h in enumerate(hist):
            Is[it][i] = h[0][it]
            Ts[it][i] = h[1][it]

    return T, Is, Ts
    
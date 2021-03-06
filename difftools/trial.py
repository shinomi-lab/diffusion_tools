from typing import Tuple, Dict
import difftools.maximization as dm
import difftools.algebra as da

import numpy as np

from joblib import Parallel, delayed

from numba import njit
from numba.pycc import CC

cc = CC("trial")


@cc.export("trial_with_sample_jit", "(i8, i8, i8, i8[:,:], f8[:,:], f8[:,:])")
@njit
def trial_with_sample_jit(
    k: int,
    m: int,
    n: int,
    adj: np.ndarray,
    prob_mat: np.ndarray,
    util_dists: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    S, _ = dm.im_greedy(None, k, m, n, adj, prob_mat)
    l = len(util_dists)
    Ts = np.zeros((l, n), np.float64)
    sw_ims = np.zeros(l, np.float64)
    sw_swms = np.zeros(l, np.float64)

    for i in range(l):
        util_dist = util_dists[i]
        T, _ = dm.swm_greedy(None, k, m, n, adj, prob_mat, util_dist)
        sw_im = dm.ic_sw_sprd_exp(None, m, n, adj, S, prob_mat, util_dist)
        sw_swm = dm.ic_sw_sprd_exp(None, m, n, adj, T, prob_mat, util_dist)
        Ts[i] = T
        sw_ims[i] = sw_im
        sw_swms[i] = sw_swm

    return sw_ims, sw_swms, S, Ts


def __f(
    util_dist, k, m, n, adj, prob_mat, S
) -> Tuple[np.ndarray, np.ndarray, np.number, np.number]:
    T, swm_hist = dm.swm_greedy(None, k, m, n, adj, prob_mat, util_dist)
    sw_im = dm.ic_sw_sprd_exp(None, m, n, adj, S, prob_mat, util_dist)
    sw_swm = dm.ic_sw_sprd_exp(None, m, n, adj, T, prob_mat, util_dist)

    return T, swm_hist, sw_im, sw_swm


def trial_with_sample(
    k: int,
    m: int,
    n: int,
    adj: np.ndarray,
    prob_mat: np.ndarray,
    util_dists: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Parameters:
        k : the maximum size of seed node sets
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray
        util_dists : utility distribusion samples on the indicator of $V$ as 1d float64 array

    Returns:
        The dictionary as:

        - `sw-ims`: a list of the social welfare by an IM opt seed set under the IC model
        - `sw-swms`: a list of the near maximums of social welfare for each utility distribution samples
        - `im-seed`: an opt-seed by influence maximization
        - `im-hist`: a history of influence maximization
        - `swm-seeds`: an opt-seed list by utility maximization
        - `swm-hists`: a list of a history of utility maximization
    """

    S, im_hist = dm.im_greedy(None, k, m, n, adj, prob_mat)
    ret = Parallel(n_jobs=-1)(
        delayed(__f)(util_dist, k, m, n, adj, prob_mat, S) for util_dist in util_dists
    )
    ret = list(zip(*ret))

    return {
        "sw-ims": np.stack(ret[2]),
        "sw-swms": np.stack(ret[3]),
        "im-seed": S,
        "im-hist": im_hist,
        "swm-seeds": np.stack(ret[0]),
        "swm-hists": np.stack(ret[1]),
    }


def trial(
    l: int,
    k: int,
    m: int,
    n: int,
    adj: np.ndarray,
    prob_mat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Parameters:
        l : the number of utility distribution samples
        k : the maximum size of seed node sets
        m : an iteration number of the IC model
        n : the number of nodes
        adj : the adjacency matrix of a graph as 2d int64 ndarray
        prob_mat : propagation probabilities on the adjacency matrix as 2d float64 ndarray

    Returns:
        The dictionary as:

        - `sw-ims`: a list of the social welfare by an IM near opt seed set under the IC model
        - `sw-swms`: a list of the opt-maximal social welfare for each utility distribution samples
        - `im-seed`: an opt-seed by influence maximization
        - `im-hist`: a history of influence maximization
        - `swm-seeds`: an indicator list of SWM near optimal seed sets
        - `swm-hists`: a list of a history of SWM
        - `utils` : $l$-size uniform samples of utility distribusions on the indicator of $V$ as 1d float64 array
    """
    util_dists = np.zeros((l, n), dtype=np.float64)
    for i in range(l):
        util_dists[i] = da.random_simplex(None, n)

    ret = trial_with_sample(k, m, n, adj, prob_mat, util_dists)
    ret["utils"] = util_dists

    return ret
import difftools.maximization as dm

import numpy as np
import matplotlib.pyplot as plt

# import numba

from joblib import Parallel, delayed, parallel_backend


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
    S, im_hist = dm.im_greedy(None, k, m, n, adj, prob_mat)
    l = len(util_dists)
    # Ts = np.zeros((l, n), dtype=np.int64)
    # um_hists = np.zeros((l, k, n))
    # total_utils = np.zeros(l)
    # max_utils = np.zeros(l)

    def f(util_dist):
        # util_dist = util_dists[i]

        # T, um_hist = dm.um_greedy(None, k, m, n, adj, prob_mat, util_dist)
        # Ts[i] = T
        # um_hists[i] = um_hist

        # total_utils[i] = dm.icu_sigma(None, m, n, adj, S, prob_mat, util_dist)
        # max_utils[i] = dm.icu_sigma(None, m, n, adj, T, prob_mat, util_dist)
        T, um_hist = dm.um_greedy(None, k, m, n, adj, prob_mat, util_dist)
        total_util = dm.icu_sigma(None, m, n, adj, S, prob_mat, util_dist)
        max_util = dm.icu_sigma(None, m, n, adj, T, prob_mat, util_dist)

        return (T, um_hist, total_util, max_util)

    r = Parallel(n_jobs=-1)(delayed(f)(util_dist) for util_dist in util_dists)

    return r
    # _Ts, _um_hists, _total_utils, _max_utils = zip(*r)
    # Ts = np.stack(_Ts)
    # um_hists = np.stack(_um_hists)
    # total_utils = np.stack(_total_utils)
    # max_utils = np.stack(_max_utils)

    # return (
    #     total_utils,
    #     max_utils,
    #     S,
    #     im_hist,
    #     util_dists,
    #     Ts,
    #     um_hists,
    # )


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
        util_dists[i] = dm.random_simplex(None, n)

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
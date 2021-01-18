import difftools.diffusion as diff

import networkx as nx
import numpy as np
import numpy.random as nrd
import matplotlib.pyplot as plt

from joblib import Parallel, delayed, parallel_backend


def random_simplex(n):
    m = n - 1
    s = np.sort(nrd.random(m))
    r = np.zeros(n)
    r[0] = s[0]
    r[n-1] = 1 - s[-1]
    for i in range(m-1):
        r[i + 1] = s[i+1] - s[i]
    return r

def random_simplex_gen(n, f):
    m = n - 1
    s = np.sort(f(m))
    r = np.zeros(n)
    r[0] = s[0]
    r[n-1] = 1 - s[-1]
    for i in range(m-1):
        r[i + 1] = s[i+1] - s[i]
    return r

## monte carlo for simplex
def greedy(n, k, sigma):
    """
    Parameters
    ----------
    n : the number of nodes
    k : the maximum size of seed node set
    sigma : (stocastic) influence function (`Idx(n) x {0,1}^n -> R`)

    Returns
    -------
    The tuple of opt-seed set and the list of influence vectors
    """

    V = np.repeat(1, n)
    S = np.zeros(n, dtype=int)
    idx = np.arange(n)

    def sus(i, e):
        if e == 0:
            return 0
            
        Su = S.copy()
        Su[i] = 1
        return sigma(idx, Su)

    hist = []
    for i in range(1, k+1):
        dds = np.asarray(Parallel(n_jobs=-1)(delayed(sus)(i, e) for i, e in enumerate(V - S)))
        # print(i, dds)
        hist.append(dds)
        v = dds.argmax()
        S[v] = 1

    return (S, hist)

def ic_sigma_x(gen, I, n, G, probs):
    d = np.zeros(n, int)
    dset = diff.independent_cascade(G, I, probs, gen)[-1][1]
    for i in dset:
        d[i] = 1
    return d

def icu_sigma_x(gen, I, n, G, probs, utils):
    d = ic_sigma_x(gen, I, n, G, probs)
    u = np.zeros(n)
    for i, v in utils.items():
        u[i] = d[i] * v
    return u

def simulate(idx, S, m, sigma_x, *args):
    """
    Parameters
    ----------
    idx : a numpy array as [0, ... , n-1]
    S : the numpy array form of a seed set
    m : sampling size
    sigma_x: (deterministic) influence function (Gen -> set[int] -> [arg] -> NDArray[number])
    args: arguments for sigma_x

    Returns
    -------
    the dictionary as:
        `dist`: a distribution (as numpy array) of influence value samples
        `mean`: the mean of the distribution
    """

    gen = nrd.default_rng()
    I = set(idx[S==1])
    ds = np.stack(Parallel(n_jobs=-1)(delayed(sigma_x)(gen, I, *args) for i in range(m)))
    dist = ds.mean(axis=0)
    return { 'mean': dist.sum(), 'dist': dist }

def gen_sigma(m, sigma_x, *args):
    return lambda idx, S: simulate(idx, S, m, sigma_x, *args)['mean']

def array_to_dict(arr):
    map = {}
    for i, v in enumerate(arr):
        map[i] = v
    return map

def trial(n, k, m, G, l, probs):
    """
    Parameters
    ----------
    n : the number of nodes
    k : the maximum size of seed node set
    m : sampling size of influence functions
    G : a networkx graph
    l : sampling size of utilities
    probs: propagation probabilities (as edge dictionary)

    Returns
    -------
    The dictionary as:
        `im-seed`: an opt-seed by influence maximization
        `im-hist`: a history of influence maximization
        `utils`: a list of utility vectors, 
        `um-seeds`: an opt-seed list by utility maximization
        `um-hists`: a list of a history of utility maximization 
        `util-samples`: a pair list of the total utility by IC with IM opt seed named as `inf`
            and optimal total utility with each UM opt seed named as `opt`
    """
    
    idx = np.arange(n)

    ic_sigma = gen_sigma(m, ic_sigma_x, n, G, probs)
    S, im_hist = greedy(n, k, ic_sigma)

    utils_dists = [random_simplex(n) for _ in range(l)]
    Ts = []
    um_hists = []
    util_samples = []
    
    for i, utils_dist in enumerate(utils_dists):
        print('.', end=('\n' if (i+1) % 64 == 0 else ''))
        icu_sigma = sigma(m, icu_sigma_x, n, G, probs, array_to_dict(utils_dist))
        T, um_hist = greedy(n, k, icu_sigma)
        Ts.append(T)
        um_hists.append(um_hist)

        util_inf_opt = np.inner(utils_dist, simulate(idx, S, m, ic_sigma_x, n, G, probs)['dist'])
        util_opt = np.inner(utils_dist, simulate(idx, T, m, ic_sigma_x, n, G, probs)['dist'])
    
        util_samples.append(np.asarray([(util_opt, util_inf_opt)], dtype=[("opt", float), ("inf", float)]))
 
    print('')
    
    Ts = np.stack(Ts)
    hist_utils = np.stack(hist_utils)
    plots = np.stack(plots)

    return {
        'im-seed': S, 
        'im-hist': im_hist,
        'utils': utils_dists, 
        'um-seeds': Ts,
        'um-hists': um_hists, 
        'util-samples': util_samples
    }


def plot_samples(samples):
    fig = plt.figure(dpi=100)
    fig.patch.set_facecolor('white')
    
    x_max = max(samples['opt'].max(), samples['opt'].max()) + 0.02
    x_min = min(samples['opt'].min(), samples['opt'].min()) - 0.02

    ax = fig.add_subplot(111)
    ax.scatter(samples['inf'], samples['opt'], s=10, label='utility sample')

    x = np.linspace(x_min, x_max, 100)
    ax.plot(x, x, c='r', linewidth=0.5, label='$y=x$')
    
    ax.set_xlabel('Total Utility of Optimal Max Influence Range')
    ax.set_ylabel('Optimal Max Utility')
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    plt.show()

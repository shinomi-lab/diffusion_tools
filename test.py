import networkx as nx
import difftools.maximization as dm
import difftools.diffusion as dd
import numpy as np
import numpy.random as nrd
import numba
from timeit import timeit

def test():
    n = 50
    g = nx.scale_free_graph(n, seed=0)
    adj = nx.to_numpy_matrix(g, dtype=int)
    
    nrd.seed(0)
    probs = {}
    for e in g.edges(): probs[e] = np.random.random()

    prob_mat = np.zeros((n,n))
    for e, p in probs.items(): prob_mat[e[0], e[1]] = p

    S = np.zeros(n, np.int64)
    S[0] = 1
    S[1] = 1
    S[4] = 1

    util_dist = dm.random_simplex(0, n)
    # print(util_dist)
    # I = set(np.arange(n)[S == 1])

    # print(dm.ic_sigma(0, 100, n, adj, S, prob_mat))
    # print(dm.ic_dist_x(0, n, adj, S, prob_mat))
    # print(dm.icu_dist_x(0, n, adj, S, prob_mat, util_dist))
    # print(dm._ic_sigma_jit(100, S, n, adj, prob_mat, 0))

    # print(timeit(lambda: dm.ic_dist(0, 100, n, adj, S, prob_mat), number=1))
    # print(timeit(lambda: dm.icu_dist(0, 100, n, adj, S, prob_mat, util_dist), number=1))
    # print(timeit(lambda: dm.im_greedy_jit(0, 2, 100, n, adj, prob_mat), number=1))
    # print(timeit(lambda: dm.um_greedy_jit(0, 2, 100, n, adj, prob_mat, util_dist), number=1))
    # print(timeit(lambda: dm.trial_jit(100, 2, 100, n, adj, prob_mat), number=1))
    uds = np.stack([dm.random_simplex(None, n) for _ in range(100)])
    # print(timeit(lambda: dm.trial_jit(100, 2, 100, n, adj, prob_mat), number=1))
    print(timeit(lambda: dm.trial_with_sample_jit(2, 100, n, adj, prob_mat, uds), number=1))
    print(timeit(lambda: dm.trial_with_sample(2, 100, n, adj, prob_mat, uds), number=1))
    
if __name__ == "__main__":
    test()
    # print(hoge(1))
    # print(hoge(None))
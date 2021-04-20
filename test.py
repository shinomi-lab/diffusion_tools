import networkx as nx
import difftools.maximization as dm
import difftools.trial as dt
import numpy as np
import numpy.random as nrd
import numba
from timeit import timeit


def test():
    g = nx.read_weighted_edgelist("./test/graph.txt", nodetype=int)
    n = g.number_of_nodes()
    # adj = nx.to_numpy_matrix(g, nodelist=range(n), dtype=int)
    adj = nx.to_numpy_matrix(g, nodelist=g.nodes(), weight=None, dtype=int)
    ns = g.nodes()
    idxs = np.zeros(n, int)
    for x, i in enumerate(ns):
        idxs[x] = i
    print(adj)

    # nrd.seed(0)
    # prob_mat = np.multiply(nrd.rand(n * n).reshape((n, n)), adj)

    prob_mat = np.zeros((n, n))
    for ix, jx in g.edges():
        p = g.edges[ix, jx]["weight"]
        i = idxs[ix]
        j = idxs[jx]
        prob_mat[i, j] = p
        if not nx.is_directed(g):
            prob_mat[j, i] = p

    S = np.zeros(n, int)
    S[0] = 1
    # print(timeit(lambda: dd.ic_mat(n, adj, S, prob_mat, 0), number=1))
    # print(timeit(lambda: dm.ic_dist(0, 100, n, adj, S, prob_mat), number=1))
    # print(timeit(lambda: dm.icu_dist(0, 100, n, adj, S, prob_mat, util_dist), number=1))
    # print(timeit(lambda: dm.im_greedy_jit(0, 2, 100, n, adj, prob_mat), number=1))
    # print(dm.um_greedy(0, 2, 100, n, adj, prob_mat, util_dist))
    # print(timeit(lambda: dm.trial_jit(100, 2, 100, n, adj, prob_mat), number=1))
    # print(timeit(lambda: print(dm.trial_jit(100, 2, 100, n, adj, prob_mat)), number=1))
    uds = np.stack([dm.random_simplex(None, n) for _ in range(100)])
    # print(timeit(lambda: print(dm.trial_with_sample(2, 100, n, adj, prob_mat, uds)), number=1))
    print(
        timeit(
            lambda: dt.trial_with_sample_jit(2, 1000, n, adj, prob_mat, uds), number=1
        )
    )
    # print(timeit(lambda: dm.trial_with_sample(2, 100, n, adj, prob_mat, uds), number=1))


if __name__ == "__main__":
    test()
    # print(hoge(1))
    # print(hoge(None))
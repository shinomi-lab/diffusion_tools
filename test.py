import networkx as nx
import difftools.algebra as da
import difftools.maximization as dm
import difftools.trial as dt
import numpy as np
import numpy.random as nrd
from timeit import timeit


def get_adj(g):
    adj = nx.to_numpy_matrix(g, nodelist=g.nodes(), weight=None, dtype=int)

    return adj


def load_test_graph():
    g = nx.read_weighted_edgelist("./test/graph.txt", nodetype=int)
    n = g.number_of_nodes()

    ns = g.nodes()
    idxs = np.zeros(n, int)
    for x, i in enumerate(ns):
        idxs[x] = i

    prob_mat = np.zeros((n, n))
    for ix, jx in g.edges():
        p = g.edges[ix, jx]["weight"]
        i = idxs[ix]
        j = idxs[jx]
        prob_mat[i, j] = p
        if not nx.is_directed(g):
            prob_mat[j, i] = p

    return g, n, get_adj(g), prob_mat


def gen_ba(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed)
    adj = get_adj(g)

    nrd.seed(0)
    prob_mat = np.multiply(nrd.rand(n * n).reshape((n, n)), adj)

    return g, n, adj, prob_mat


def test():
    _, n, adj, prob_mat = gen_ba(10, 2, 0)

    print(adj)

    S = np.zeros(n, int)
    S[0] = 1

    # print(timeit(lambda: dd.ic_mat(n, adj, S, prob_mat, 0), number=1))
    # print(timeit(lambda: dm.ic_dist(0, 100, n, adj, S, prob_mat), number=1))
    # print(timeit(lambda: dm.icu_dist(0, 100, n, adj, S, prob_mat, util_dist), number=1))
    # print(timeit(lambda: dm.im_greedy_jit(0, 2, 100, n, adj, prob_mat), number=1))
    # print(dm.um_greedy(0, 2, 100, n, adj, prob_mat, util_dist))
    print(
        timeit(
            lambda: (lambda r: print(r["sw-ims"], "\n", r["sw-opts"]))(
                dt.trial(10, 2, 1000, n, adj, prob_mat)
            ),
            number=1,
        )
    )


if __name__ == "__main__":
    test()
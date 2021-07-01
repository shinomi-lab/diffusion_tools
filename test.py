import networkx as nx
import difftools.algebra as da
import difftools.maximization as dm
import difftools.trial as dt
import difftools.diffusion.multi as ddm
import difftools.diffusion.ic as ddi
import numba

import numpy as np
import numpy.random as nrd
from timeit import timeit
import fire


def get_adj(g):
    adj = nx.to_numpy_array(g, nodelist=g.nodes(), weight=None, dtype=np.int64)

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

    return g, n, adj

def gen_ba_with_prob(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed)
    adj = get_adj(g)

    nrd.seed(0)
    prob_mat = np.multiply(nrd.rand(n * n).reshape((n, n)), adj)

    return g, n, adj, prob_mat


def test_trial():
    _, n, adj, prob_mat = gen_ba_with_prob(10, 2, 0)

    print(adj)

    S = np.zeros(n, int)
    S[0] = 1

    # print(timeit(lambda: dd.ic_mat(n, adj, S, prob_mat, 0), number=1))
    # print(timeit(lambda: dm.ic_dist(0, 100, n, adj, S, prob_mat), number=1))
    # print(timeit(lambda: dm.icu_dist(0, 100, n, adj, S, prob_mat, util_dist), number=1))
    # print(timeit(lambda: dm.im_greedy_jit(0, 2, 100, n, adj, prob_mat), number=1))
    # print(dm.um_greedy(0, 2, 100, n, adj, prob_mat, util_dist))
    l = 100
    util_dists = np.zeros((l, n), dtype=np.float64)
    for i in range(l):
        util_dists[i] = da.random_simplex(None, n)
    print(
        timeit(
            lambda: print(
                dt.trial_with_sample_jit(1, 1000, n, adj, prob_mat, util_dists)
            ),
            number=1,
        )
    )

def test_multi():
    _, n, adj = gen_ba(10, 2, 0)
    
    p = 0.7
    prob_mat = adj * p

    S_multi = np.zeros((2, n), np.int64)
    S_multi[ddm.InfoType_T][1] = 1
    S_multi[ddm.InfoType_F][1] = 1

    stt = ddm.make_simple_stt()
    mas = np.stack([ddm.make_simple_assumption_map(1, 0, 0, 1) for _ in range(n)])
    mps = np.stack([ddm.make_simple_propagation_map(p, p, p, p, p, p, p, p) for _ in range(n)])

    m_is_t = np.zeros(n)
    m_is_f = np.zeros(n)
    s_is = np.zeros(n)

    s = 0
    it = ddm.InfoType_T
    # print(S_multi[it][1])
    for _ in range(5):
        s, p = ddm.run_markov(stt, mas[1], mps[1], s, it)
        it = ddm.InfoType_F
        print(s, p)
    
    # print(ddm.multi_ic_adjmat(n, adj, S_multi, stt, mas, mps, 0))
    # print(ddi.ic_adjmat(n, adj, S_multi[ddm.InfoType_T], prob_mat, 0)[0])
    N = 10000
    for _ in range(N):
        r = ddm.multi_ic_adjmat(n, adj, S_multi, stt, mas, mps, None)[0]
        m_is_t += r[ddm.InfoType_T] / N
        m_is_f += r[ddm.InfoType_F] / N
        s_is += ddi.ic_adjmat(n, adj, S_multi[ddm.InfoType_T], prob_mat, None)[0] / N

    print(m_is_t)
    print(m_is_f)
    print(s_is)

def main():
    fire.Fire({'trial': test_trial, 'multi': test_multi})

if __name__ == "__main__":
    main()
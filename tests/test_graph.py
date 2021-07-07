from difftools.graph import randomize_direction, randomize_direction_adj
import numpy.random as nrd
import numpy as np
import networkx as nx


def test_randomize_direction():
    rng = nrd.default_rng(0)
    f = nx.barabasi_albert_graph(10, 2, 0)
    g = randomize_direction(f, rng)

    for (i, j) in f.edges:
        assert g.has_edge(i, j) ^ g.has_edge(j, i)


def test_randomize_direction_adj():
    rng = nrd.default_rng(0)
    g = nx.barabasi_albert_graph(10, 2, 0)
    adj = nx.to_numpy_array(g, weight=None)
    new_adj = randomize_direction_adj(adj, rng)

    assert np.array_equal(adj, new_adj + new_adj.T)

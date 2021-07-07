import numpy as np
import numpy.random as nrd
import numpy.typing as npt
import networkx as nx


def randomize_direction(f: nx.Graph, rng: nrd.Generator) -> nx.DiGraph:
    g = f.to_directed()
    for (i, j) in f.edges:
        if rng.random() > 0.5:
            g.remove_edge(i, j)
        else:
            g.remove_edge(j, i)

    return g


def randomize_direction_adj(adj: npt.NDArray[np.int64], rng: nrd.Generator) -> npt.NDArray[np.int64]:
    new_adj = np.zeros(adj.shape, np.int64)
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            assert adj[i][j] == adj[j][i], "the given adj is not symmetric"
            if adj[i][j] == 0:
                continue
            if rng.random() >= 0.5:
                new_adj[i][j] = 1
            else:
                new_adj[j][i] = 1

    return new_adj

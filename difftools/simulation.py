import numpy as np
import scipy as sp
import difftools.diffusion as diff
import difftools.combination as comb
from joblib import Parallel, delayed
from joblib import parallel_backend


def final_states(rs):
    return np.stack(list(map(lambda r: r[-1], rs)))


def traverse_all_pattern(n, adj, k, i):
    rs = []
    s0 = diff.single_source(n, i)
    for j, s in enumerate(comb.comb_binary_with_index(n, k, i)):
        if (j + 1) % 10 == 0: print(".", end="")
        if (j + 1) % 1000 == 0: print()
        rs.append(diff.diffuse(adj, s, s0))

    return rs


def __rbswi_bstn(n, k, i, gen):
    s = comb.random_bit_seq_with_index(n, k, i, gen)
    num = comb.bit_seq_to_num(s)
    return num, s


def random_search(n, adj, k, i, min_rep, rate, seed):
    ts = []

    gen = diff.get_gen(seed)

    comb = sp.special.comb(n - 1, k - 1, True)
    repeat = int(comb * rate)

    s0 = diff.single_source(n, i)

    if comb <= min_rep:
        with parallel_backend('multiprocessing'):
            ts = Parallel(n_jobs=-1)(
                delayed(diff.diffuse)(adj, s, s0)
                for (num, s) in comb.comb_binary_with_index(n, k, i))
    else:
        with parallel_backend('multiprocessing'):
            ns = Parallel(n_jobs=-1)(delayed(__rbswi_bstn)(n, k, i, gen)
                                     for j in range(repeat))

        mem = {}
        for num, s in ns:
            mem[num] = s

        with parallel_backend('multiprocessing'):
            ts = Parallel(n_jobs=-1)(delayed(diff.diffuse)(adj, s, s0)
                                     for s in mem.values())

        print(len(ts) / comb)
    return ts


def scoring(n, ss, data, i):
    cs = []
    for k in range(n + 1):
        s = ss[data == k]
        m = s.shape[0]
        if m:
            c = s.sum(axis=0) / m
            c[i] = 0
            cs.append((k, c))
    return cs

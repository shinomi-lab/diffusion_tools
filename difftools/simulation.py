from typing import Tuple, Any, Set, List
import numpy as np
import scipy as sp
import difftools.diffusion as diff
import difftools.combination as comb
from joblib import Parallel, delayed
from joblib import parallel_backend


def iterate(f, t, *args):
    ts = []
    for i in range(t):
        ts.append(f(*args))
        if (i + 1) % 10 == 0: print(".", end="")
        if (i + 1) % 1000 == 0: print("")

    return ts


def iterate_parallel(f, t, *args):
    with parallel_backend('multiprocessing'):
        ts = Parallel(n_jobs=-1)(delayed(f)(*args) for j in range(t))

    return ts


def fold_all_pattern(n, adj, k, i) -> List[Tuple[Any, Any]]:
    s0 = diff.single_source(n, i)

    with parallel_backend('multiprocessing'):
        rs: List[Tuple[Any, Any]] = Parallel(n_jobs=-1)(
            delayed(diff.diffuse)(adj, s, s0)
            for (num, s) in comb.comb_binary_with_index(n, k, i))

    return rs


def fold_random_pattern(n, adj, k, i, repeat, gen) -> List[Tuple[Any, Any]]:
    s0 = diff.single_source(n, i)

    with parallel_backend('multiprocessing'):
        ss = Parallel(n_jobs=-1)(
            delayed(comb.random_bit_seq_with_index)(n, k, i, gen)
            for _ in range(repeat))

    with parallel_backend('multiprocessing'):
        ns = Parallel(n_jobs=-1)(delayed(comb.bit_seq_to_num)(s) for s in ss)

    mem = {}
    for num, s in zip(ns, ss):
        mem[num] = s

    with parallel_backend('multiprocessing'):
        rs = Parallel(n_jobs=-1)(delayed(diff.diffuse)(adj, s, s0)
                                 for s in mem.values())

    return rs


def search_with_approx(n, adj, k, i, min_rep, rate, seed) -> List[Tuple[Any, Any]]:
    comb_count = sp.special.comb(n - 1, k - 1, True)

    if comb_count <= min_rep:
        return fold_all_pattern(n, adj, k, i)
    else:
        repeat = int(comb_count * rate)
        gen = diff.get_gen(seed)
        return fold_random_pattern(n, adj, k, i, repeat, gen)


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

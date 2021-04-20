import numpy as np
from joblib import Parallel, delayed
from joblib import parallel_backend


def comb_binary(n, r):
    stack = {(n, r, 0)}
    ks = []

    while len(stack):
        m, s, k = stack.pop()
        sd = s - 1
        for i in range(sd, m):
            t = (i, sd, k + (2 ** i))
            if sd == 0:
                ks.append(t[2])
            else:
                stack.add(t)

    return ks


def to_bit_seq(n, l):
    b = []

    k = n
    i = 0
    while i < l:
        if k == 0:
            b.append(False)
        else:
            b.append(k % 2 == 1)
            k = k >> 1
        i += 1

    return np.asarray(b, dtype=bool)


def bit_seq_to_num(seq):
    n = len(seq)
    return ((1 << np.arange(n)) * seq).sum().item()


def insert_bit(k, n, i):
    s = 1 << i
    r = k % s
    m = ((k - r) << 1) + s + r
    return (m, np.asarray(to_bit_seq(m, n)))


def comb_binary_with_index(n, r, i):
    if r == 1:
        m = 1 << i
        return [(m, np.asarray(to_bit_seq(m, n)))]

    ts = []
    ks = comb_binary(n - 1, r - 1)

    with parallel_backend("multiprocessing"):
        ts = Parallel(n_jobs=-1)(delayed(insert_bit)(k, n, i) for k in ks)

    return ts


def random_bit_seq_with_index(n, r, i, gen):
    return np.insert(gen.permutation(np.arange(n - 1) < (r - 1)), i, True)

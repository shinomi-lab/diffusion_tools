from difftools.diffusion.multi import run_markov, make_simple_stt, make_simple_assumption_map, make_simple_propagation_map, InfoType_T, InfoType_F, multi_ic_adjmat
from difftools.diffusion.ic import ic_adjmat
import numpy as np


def test_run_markov():
    stt = make_simple_stt()
    ma1 = make_simple_assumption_map(1, 0, 0, 1)
    ma2 = make_simple_assumption_map(0, 1, 1, 0)
    ma3 = make_simple_assumption_map(0.8, 0.2, 0.2, 0.8)

    ppt_t0 = 0.0
    ppf_t0 = 0.1
    ppt_f0 = 0.2
    ppf_f0 = 0.3
    ppt_f1 = 0.4
    ppf_f1 = 0.5
    ppt_t1 = 0.6
    ppf_t1 = 0.7

    mp = make_simple_propagation_map(
        ppt_t0, ppf_t0, ppt_f0, ppf_f0,
        ppt_f1, ppf_f1, ppt_t1, ppf_t1)

    assert run_markov(stt, ma1, mp, 0, InfoType_T) == (1, ppt_t0)
    assert run_markov(stt, ma1, mp, 1, InfoType_T) == (1, 0)
    assert run_markov(stt, ma1, mp, 1, InfoType_F) == (3, ppf_f1)
    assert run_markov(stt, ma1, mp, 0, InfoType_F) == (2, ppf_f0)
    assert run_markov(stt, ma1, mp, 2, InfoType_F) == (2, 0)
    assert run_markov(stt, ma1, mp, 2, InfoType_T) == (3, ppt_t1)
    assert run_markov(stt, ma1, mp, 3, InfoType_T) == (3, 0)
    assert run_markov(stt, ma1, mp, 3, InfoType_F) == (3, 0)

    assert run_markov(stt, ma2, mp, 0, InfoType_T) == (1, ppf_t0)
    assert run_markov(stt, ma2, mp, 1, InfoType_T) == (1, 0)
    assert run_markov(stt, ma2, mp, 1, InfoType_F) == (3, ppt_f1)
    assert run_markov(stt, ma2, mp, 0, InfoType_F) == (2, ppt_f0)
    assert run_markov(stt, ma2, mp, 2, InfoType_F) == (2, 0)
    assert run_markov(stt, ma2, mp, 2, InfoType_T) == (3, ppf_t1)
    assert run_markov(stt, ma2, mp, 3, InfoType_T) == (3, 0)
    assert run_markov(stt, ma2, mp, 3, InfoType_F) == (3, 0)

    N = 1000
    ppt_t0_n = 0
    ppf_t0_n = 0
    ppt_f0_n = 0
    ppf_f0_n = 0
    ppt_f1_n = 0
    ppf_f1_n = 0
    ppt_t1_n = 0
    ppf_t1_n = 0

    for _ in range(N):
        if run_markov(stt, ma3, mp, 0, InfoType_T)[1] == ppt_t0:
            ppt_t0_n += 1
        else:
            ppf_t0_n += 1
        if run_markov(stt, ma3, mp, 0, InfoType_F)[1] == ppf_f0:
            ppf_f0_n += 1
        else:
            ppt_f0_n += 1
        if run_markov(stt, ma3, mp, 1, InfoType_F)[1] == ppf_f1:
            ppf_f1_n += 1
        else:
            ppt_f1_n += 1
        if run_markov(stt, ma3, mp, 2, InfoType_T)[1] == ppt_t1:
            ppt_t1_n += 1
        else:
            ppf_t1_n += 1

    assert ppt_t0_n + ppf_t0_n == N and ppt_t0_n > ppf_t0_n
    assert ppf_f0_n + ppt_f0_n == N and ppf_f0_n > ppt_f0_n
    assert ppf_f1_n + ppt_f1_n == N and ppf_f1_n > ppt_f1_n
    assert ppt_t1_n + ppf_t1_n == N and ppt_t1_n > ppf_t1_n


def test_multi_ic_adjmat():
    n = 4
    adj = np.zeros((n, n), np.int64)
    adj[0][1] = 1
    adj[0][2] = 1
    adj[2][3] = 1

    S = np.zeros(n, np.int64)
    S[0] = 1

    S_multi1 = np.zeros((2, n), np.int64)
    S_multi1[0][0] = 1

    S_multi2 = np.zeros((2, n), np.int64)
    S_multi2[1][0] = 1

    p = 0.8
    prob_mat = adj.astype(np.float64) * p

    stt = make_simple_stt()
    ma = make_simple_assumption_map(1, 0, 0, 1)
    mp = make_simple_propagation_map(p, p, p, p, p, p, p, p)
    asm = np.stack([ma for _ in range(n)])
    prp = np.stack([mp for _ in range(n)])

    ans = ic_adjmat(n, adj, S, prob_mat, 0)[0]
    ans_multi1 = multi_ic_adjmat(n, adj, S_multi1, stt, asm, prp, 0)[0]
    ans_multi2 = multi_ic_adjmat(n, adj, S_multi2, stt, asm, prp, 0)[0]

    assert np.array_equal(ans, ans_multi1[0])
    assert np.array_equal(ans, ans_multi2[1])

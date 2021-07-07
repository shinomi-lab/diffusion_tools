import numpy as np

NDArray = Any

def information_vitality(p, i) -> float:
    return np.inner(p, i)


def information_entropy(p) -> float:
    return np.sum(np.ma.log(p) * p)


def normal_information_potential(p0, p1, i) -> float:
    dt = np.inner(p1 - p0, i)

    if dt == 0:
        return 0

    def f(p: float) -> float:
        if p == 0 or p == 1:
            return 0
        else:
            return p * p * math.log(p)

    def v(p: Tuple[Any, Any]) -> float:
        p0 = p[0]
        p1 = p[1]

        if p0 == p1:
            if p0 == 0:
                return 0
            elif p0 == 1:
                return 1
            else:
                return p0 * (1 + 2 * math.log(p0))
        else:
            return (f(p1) - f(p0)) / (p1 - p0)

    return dt * (1.0 - sum(map(v, zip(p0, p1))))


def potentials(p0, p1, i):
    return np.fromiter(
        map(lambda j: normal_information_potential(p0[j], p1[j], i[j]), range(4)),
        "float",
    )


def mapping(score, patterns):
    rank = np.argsort(np.argsort(score)) / score.shape[0]
    n = patterns.shape[0]
    p = {}
    for i in range(n):
        p[i] = rank >= (n - 1 - i) / n

    ret = p[0] * patterns[0]
    for i in range(1, n):
        ret += (~p[i - 1] & p[i]) * patterns[i]

    return ret


def single_source(n, i):
    v = np.zeros(n, bool)
    v[i] = True
    return v


def diffuse(adj, sender_vec, s0) -> List[Tuple[NDArray, NDArray]]:
    adjd = adj * sender_vec

    rs = [(s0, s0)]
    cs = s0
    ct = s0

    while not all(cs == 0):
        cs = (np.matmul(adjd, cs) > 0) & (~ct)
        ct = cs + ct
        rs.append((cs, ct))

    return rs


def diffuse_with_dq(
    n, adj, dq_vec, source, gen
) -> Tuple[List[Tuple[NDArray, NDArray]], NDArray, NDArray]:
    s0 = single_source(n, source)
    theta_vec = np.asarray(gen.random(n))  # thresholds(size, gen)
    sender_vec = dq_vec > theta_vec
    sender_vec[source] = True

    return diffuse(adj, sender_vec, s0), sender_vec, theta_vec

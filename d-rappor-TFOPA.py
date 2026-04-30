import numpy as np
import random
import math


# -------------------------
# d-RAPPOR primitives (from your TFOPA code)
# -------------------------
def encode(domain, data_user):
    l = []
    d = len(domain)
    for data in data_user:
        temp = [0] * d
        for x in data:
            inx = domain.index(x)
            temp[inx] = 1
        l.append(temp)
    return l


def perturb(data_encode, domain, epsilon, c):
    d = len(domain)
    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)

    for data in data_encode:
        for i in range(d):
            if np.random.uniform() > p:
                data[i] = 1 - data[i]
    return data_encode


def estimate(data_perturb, domain, epsilon, c):
    array = np.sum(data_perturb, axis=0)
    count_dict = dict(zip(domain, array))

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    n = len(data_perturb)
    es_data = []
    Ct = []
    for x in domain:
        x_count = count_dict.get(x, 0)
        rs = (x_count - n * q) / (n * (p - q))
        es_data.append(rs)
        Ct.append(x_count)
    return Ct, es_data


def data_fake_estimate(data_fake_encode, domain, epsilon, c, ct_dict, n_benign):
    """
    TFOPA aggregation: combine benign perturbed counts (ct_dict) with forged perturbed counts (data_fake_encode),
    then decode frequency.
    """
    array = np.sum(data_fake_encode, axis=0)
    fake_count_dict = dict(zip(domain, array))

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    N = len(data_fake_encode) + n_benign
    es_data = []
    for x in domain:
        x_count_fake = fake_count_dict.get(x, 0)
        x_count_all = x_count_fake + ct_dict.get(x, 0)
        rs = (x_count_all - N * q) / (N * (p - q))
        es_data.append(rs)
    return es_data


# -------------------------
# Data + attack helpers (from your code)
# -------------------------
def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


def calculate_column_averages(matrix):
    if len(matrix) == 0:
        return []
    arr = np.asarray(matrix, dtype=float)
    return arr.mean(axis=0).tolist()


def round_list_values(lst, ndigits=6):
    return [round(float(v), ndigits) for v in lst]


def calculate_mse(list1, list2):
    a = np.asarray(list1, dtype=float)
    b = np.asarray(list2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Length mismatch")
    return float(np.mean((a - b) ** 2))


def output_attack(epsilon: float, c: int, r_item: list, r_fre: list, n: int, S_dict):
    """
    Solve minimal u and forged perturbed counts per target (integer rounding).
    Returns: [u, S_u(v1), S_u(v2), ...]
    """
    result = []
    rlen = len(r_item)

    e_ep = np.exp(epsilon / (2 * c))
    p = e_ep / (1 + e_ep)
    q = 1 - p

    r_solve1, r_solve2, r_solve3 = [], [], []
    for i in range(rlen):
        t1 = r_fre[i] * (p - q) + q
        t2 = (r_fre[i] * (p - q) + q) * n
        t3 = S_dict[r_item[i]]
        r_solve1.append(t1)
        r_solve2.append(t2)
        r_solve3.append(t3)

    u = 0
    while True:
        u += 1
        ok = True
        for i in range(rlen):
            tt = r_solve1[i] * u + r_solve2[i] - r_solve3[i]
            if tt < 0 or tt > u:
                ok = False
                break
        if ok:
            result.append(u)
            for i in range(rlen):
                tt = round(r_solve1[i] * u + r_solve2[i] - r_solve3[i])
                result.append(tt)
            return result


def generate_fake_data(att_result: list, r_item: list, remain_list: list, c: int):
    """
    Construct fake users' *raw sets* of length c, then encode them as bit vectors.
    This matches your original construction.
    """
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    # capacity: each fake user contributes at most c target placements
    if sum(r_count) > (u * c):
        u = math.ceil(sum(r_count) / c)

    fake_data = [[] for _ in range(u)]
    index = 0
    for (k_item, v_cnt) in r_dict.items():
        for i in range(v_cnt):
            fake_data[(index + i) % u].append(k_item)
        index += v_cnt

    for idx, x in enumerate(fake_data):
        if len(x) < c:
            x.extend(random.sample(remain_list, c - len(x)))
        elif len(x) > c:
            fake_data[idx] = random.sample(x, c)

    return fake_data


# -------------------------
# Wasserstein on unordered discrete domain (unit ground distance)
# W1 = 0.5 * L1(p, q)
# -------------------------
def wasserstein_unit_discrete(f_clean, f_atk, eps: float = 1e-12, clip01: bool = True):
    p = np.asarray(f_clean, dtype=float).copy()
    q = np.asarray(f_atk, dtype=float).copy()
    if p.shape != q.shape:
        raise ValueError("f_clean and f_atk must have same length")

    p[~np.isfinite(p)] = 0.0
    q[~np.isfinite(q)] = 0.0

    if clip01:
        p = np.clip(p, 0.0, 1.0)
        q = np.clip(q, 0.0, 1.0)

    sp = float(p.sum())
    sq = float(q.sum())
    if sp <= eps:
        p[:] = 1.0 / len(p)
    else:
        p /= sp
    if sq <= eps:
        q[:] = 1.0 / len(q)
    else:
        q /= sq

    return 0.5 * float(np.sum(np.abs(p - q)))




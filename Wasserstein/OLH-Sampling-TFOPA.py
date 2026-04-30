import numpy as np
import math
import xxhash
import random


# -------------------------
# Wasserstein (unordered discrete domain, unit ground distance)
# W1 = 0.5 * L1(p, q) after normalization
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


# -------------------------
# OLH-Sampling primitives (your code)
# -------------------------
def set_to_single(data_user, c):
    n = len(data_user)
    data_sample = [0 for _ in range(n)]
    for i in range(n):
        data_sample[i] = data_user[i][np.random.randint(c)]
    return data_sample


def OLH_Perturb_sample(data_sample, epsilon):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    n = len(data_sample)
    data_perturb = [0 for _ in range(n)]
    for i in range(n):
        data_perturb[i] = xxhash.xxh32(str(data_sample[i]), seed=i).intdigest() % g
        t = np.random.random()
        if t > p:
            temp = np.random.randint(g)
            while temp == data_perturb[i]:
                temp = np.random.randint(g)
            data_perturb[i] = temp
    return data_perturb


def OLH_Aggregate_sample(data_perturb, c, epsilon, domain):
    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g
    d = len(domain)
    n = len(data_perturb)
    count = [0 for _ in range(d)]
    Z = [0.0 for _ in range(d)]

    for i in range(d):
        t_count = 0
        for j in range(n):
            temp = xxhash.xxh32(str(domain[i]), seed=j).intdigest() % g
            if temp == data_perturb[j]:
                t_count += 1
        count[i] = t_count
        Z[i] = c * (t_count / n - q) / (p - q)
    return count, Z


# -------------------------
# Data + metrics helpers
# -------------------------
def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


def calculate_mse(list1, list2):
    a = np.asarray(list1, dtype=float)
    b = np.asarray(list2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Length mismatch")
    return float(np.mean((a - b) ** 2))


def calculate_column_averages(matrix):
    if not matrix:
        return []
    arr = np.asarray(matrix, dtype=float)
    return arr.mean(axis=0).tolist()


def round_list_values(lst, ndigits=6):
    return [round(float(v), ndigits) for v in lst]


# -------------------------
# TFOPA helpers (your code)
# -------------------------
def output_attack(epsilon: float, r_item: list, r_fre: list, n: int, S_dict, c: int):
    result = []
    rlen = len(r_item)

    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g

    r_solve1, r_solve2, r_solve3 = [], [], []
    for i in range(rlen):
        t1 = (r_fre[i] / c) * (p - q) + q
        t2 = ((r_fre[i] / c) * (p - q) + q) * n
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


def generate_fake_data_olh_sample(att_result: list, r_item: list, remain_list: list):
    """
    Your original construction: each fake user holds a (possibly small) set of candidate items.
    Later we pick one item per fake user via random.choice(x) to form fake sampled items.
    """
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    # In OLH-Sampling, each user contributes only one sampled item.
    if sum(r_count) > u:
        u = sum(r_count)

    fake_data = [[] for _ in range(u)]
    index = 0
    for (k, v) in r_dict.items():
        for i in range(v):
            fake_data[(index + i) % u].append(k)
        index += v

    # Ensure each fake user has at least one item to sample from
    for x in fake_data:
        if len(x) == 0:
            x.append(random.choice(remain_list))

    return fake_data


def data_estimate(n_benign, epsilon, domain, Ct, fake_sample_items, c):
    """
    Your TFOPA decoding shortcut:
    - Ct: benign-side OLH match counts for each domain item (from OLH_Aggregate_sample)
    - fake_sample_items: one sampled item per fake user (not hashed/matched), counted directly
    """
    d = len(domain)
    Estimate_Dist = [0.0 for _ in range(d)]

    g = math.ceil(math.exp(epsilon) + 1)
    p = 1 / 2
    q = 1 / g

    u = len(fake_sample_items)
    all_n = n_benign + u

    temp = [0 for _ in range(d)]
    for i in range(d):
        item = domain[i]
        temp[i] = sum(1 for x in fake_sample_items if x == item)

    for i in range(d):
        Estimate_Dist[i] = c * ((Ct[i] + temp[i]) / all_n - q) / (p - q)
    return Estimate_Dist




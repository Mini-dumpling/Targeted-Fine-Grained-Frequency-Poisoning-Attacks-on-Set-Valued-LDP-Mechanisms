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
# OLH-Sampling (your code)
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
    Z = [0.0 for _ in range(d)]
    for i in range(d):
        t_count = 0
        for j in range(n):
            temp = xxhash.xxh32(str(domain[i]), seed=j).intdigest() % g
            if temp == data_perturb[j]:
                t_count += 1
        Z[i] = c * (t_count / n - q) / (p - q)
    return Z


# -------------------------
# Data + attack helpers (your code)
# -------------------------
def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


def calculate_mse(list1, list2):
    a = np.asarray(list1, dtype=float)
    b = np.asarray(list2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Length mismatch")
    return float(np.mean((a - b) ** 2))


def count_occurrences(A, B):
    result_dict = {}
    for element in A:
        count = 0
        for row in B:
            count += row.count(element)
        result_dict[element] = count
    return result_dict


def input_attack(r_item: list, r_fre: list, n: int, count_dict: dict):
    result = []
    r = len(r_item)
    r_solve1, r_solve2, r_solve3 = [], [], []
    for i in range(r):
        r_solve1.append(r_fre[i])
        r_solve2.append(r_fre[i] * n)
        r_solve3.append(count_dict[r_item[i]])

    u = 0
    while True:
        u += 1
        ok = True
        for i in range(r):
            tt = r_solve1[i] * u + r_solve2[i] - r_solve3[i]
            if tt < 0 or tt > u:
                ok = False
                break
        if ok:
            result.append(u)
            for i in range(r):
                result.append(int(round(r_solve1[i] * u + r_solve2[i] - r_solve3[i])))
            return result


def generate_fake_data(att_result: list, r_item: list, remain_list: list, c: int):
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    # NOTE: your original code uses capacity sum(r_count) > u (one sampled item per fake user)
    if sum(r_count) > u:
        u = sum(r_count)

    fake_data = [[] for _ in range(u)]
    index = 0
    for (k, v) in r_dict.items():
        for i in range(v):
            fake_data[(index + i) % u].append(k)
        index += v

    for idx, x in enumerate(fake_data):
        if len(x) < c:
            x.extend(random.sample(remain_list, c - len(x)))
        if len(x) > c:
            fake_data[idx] = random.sample(x, c)

    return fake_data


def calculate_column_averages(matrix):
    if not matrix:
        return []
    arr = np.asarray(matrix, dtype=float)
    return arr.mean(axis=0).tolist()


def round_list_values(lst, ndigits=6):
    return [round(float(v), ndigits) for v in lst]




import numpy as np
import numpy.random as r
from scipy.special import comb
import random
import math


# -------------------------
# Utilities
# -------------------------
def count_occurrences(A, B):
    result_dict = {}
    for element in A:
        count = 0
        for row in B:
            count += row.count(element)
        result_dict[element] = count
    return result_dict


def calculate_column_averages(matrix):
    if not matrix:
        return []
    col_count = len(matrix[0])
    column_sums = [0.0] * col_count
    for row in matrix:
        for i, value in enumerate(row):
            column_sums[i] += value
    return [s / len(matrix) for s in column_sums]


def round_list_values(lst, ndigits=6):
    return [round(float(value), ndigits) for value in lst]


def calculate_mse(list1, list2):
    a = np.asarray(list1, dtype=float)
    b = np.asarray(list2, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Length mismatch")
    return float(np.mean((a - b) ** 2))


# -------------------------
# Wasserstein distance on unordered discrete domain (unit ground distance)
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


# -------------------------
# PrivSet Mechanism
# -------------------------
class PrivSet:
    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k
        self.__setparams()

    def __setparams(self):
        if self.k is None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        self.normalizer = nonInterCount + interCount * np.exp(self.ep)

    @staticmethod
    def bestSubsetSize(d, m, ep):
        errorbounds = np.full(d + m, 0.0)
        infos = [None] * (d + m)
        for k in range(1, d):
            interCount = comb(d + m, k) - comb(d, k)
            nonInterCount = comb(d, k)
            normalizer = nonInterCount + interCount * np.exp(ep)
            trate = comb(d + m - 1, k - 1) * np.exp(ep) / normalizer
            frate = (comb(d - 1, k - 1) +
                     (interCount * k - comb(d + m - 1, k - 1) * m) * np.exp(ep) / d) / normalizer
            errorbounds[k] = (trate * (1.0 - trate) + (d + m - 1) * frate * (1.0 - frate)) / (
                (trate - frate) * (trate - frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d]) + 1
        return [bestk] + infos[bestk]

    def randomizer(self, secrets, domain):
        pub = np.zeros(self.d + self.m, dtype=int)
        probs = np.full(self.k + 1, 0.0)

        for inter in range(0, self.k + 1):
            probs[inter] = comb(self.m, inter) * comb(self.d, self.k - inter) / self.normalizer

        probs = probs * np.exp(self.ep)
        probs[0] = probs[0] / np.exp(self.ep)

        for inter in range(1, self.k + 1):
            probs[inter] += probs[inter - 1]

        p = r.random(1)[0]
        sinter = 0
        while probs[sinter] <= p:
            sinter += 1

        domain_pad = domain + []
        for i in range(self.m):
            domain_pad.append(self.d + i)

        remain = list(set(domain_pad) - set(secrets))
        pubset = random.sample(secrets, sinter) + random.sample(remain, self.k - sinter)

        for i in range(0, self.d + self.m):
            pub[i] = 1 if i in pubset else 0
        return pub


class PrivSet_SERVER:
    def __init__(self, d, m, ep, k=None):
        self.ep = ep
        self.d = d
        self.m = m
        self.k = k
        self.__setparams()

    def __setparams(self):
        if self.k is None:
            self.k = self.bestSubsetSize(self.d, self.m, self.ep)[0]
        interCount = comb(self.d + self.m, self.k) - comb(self.d, self.k)
        nonInterCount = comb(self.d, self.k)
        normalizer = nonInterCount + interCount * np.exp(self.ep)
        self.trate = comb(self.d + self.m - 1, self.k - 1) * np.exp(self.ep) / normalizer
        self.frate = (comb(self.d - 1, self.k - 1) +
                      (interCount * self.k - comb(self.d + self.m - 1, self.k - 1) * self.m) * np.exp(self.ep) / self.d) / normalizer

    @staticmethod
    def bestSubsetSize(d, m, ep):
        return PrivSet.bestSubsetSize(d, m, ep)

    def decoder(self, domain, hits):
        domain_pad = domain + [self.d + i for i in range(self.m)]
        array = np.sum(hits, axis=0)
        count_dict = dict(zip(domain_pad, array))
        num = len(hits)

        es_data = []
        for x in range(0, self.d + self.m):
            x_count = count_dict.get(x, 0)
            fs = (x_count - num * self.frate) / (num * (self.trate - self.frate))
            es_data.append(fs)
        return es_data


# -------------------------
# Data generation + protocol run
# -------------------------
def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


def run_randomizer(data: list, domain: list, m: int, ep, k):
    per_data = []
    d = len(domain)
    privset = PrivSet(d, m, ep, k)
    for x in data:
        per_data.append(privset.randomizer(x, domain).tolist())
    return per_data


def estimate_freq(per_data: list, domain: list, d: int, m: int, ep, k):
    server = PrivSet_SERVER(d, m, ep, k)
    return server.decoder(domain, per_data)


# -------------------------
# TFIPA-style attack helper (your original logic)
# -------------------------
def input_attack(r_item: list, r_fre: list, n: int, count_dict: dict):
    rlen = len(r_item)
    a = [r_fre[i] for i in range(rlen)]
    b = [r_fre[i] * n for i in range(rlen)]
    c = [count_dict[r_item[i]] for i in range(rlen)]

    u = 0
    while True:
        u += 1
        ok = True
        for i in range(rlen):
            tt = a[i] * u + b[i] - c[i]
            if tt < 0 or tt > u:
                ok = False
                break
        if ok:
            result = [u]
            for i in range(rlen):
                result.append(int(round(a[i] * u + b[i] - c[i])))
            return result


def generate_fake_data(att_result: list, r_item: list, remain_list: list, c: int):
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    # capacity constraint: each user has c items
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


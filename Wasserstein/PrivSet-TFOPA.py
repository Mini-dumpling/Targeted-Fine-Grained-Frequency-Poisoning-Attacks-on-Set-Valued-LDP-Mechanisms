import numpy as np
import numpy.random as r
from scipy.special import comb
import random
import math


# -------------------------
# Wasserstein distance (unordered discrete domain, unit ground distance)
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
# PrivSet mechanism
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
            frate = (
                comb(d - 1, k - 1)
                + (interCount * k - comb(d + m - 1, k - 1) * m) * np.exp(ep) / d
            ) / normalizer
            errorbounds[k] = (
                trate * (1.0 - trate) + (d + m - 1) * frate * (1.0 - frate)
            ) / ((trate - frate) * (trate - frate))
            infos[k] = [trate, frate, errorbounds[k]]
        bestk = np.argmin(errorbounds[1:d]) + 1
        return [bestk] + infos[bestk]

    def randomizer(self, secrets, domain):
        pub = np.zeros(self.d, dtype=int)
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

        remain = list(set(domain) - set(secrets))
        pubset = random.sample(secrets, sinter) + random.sample(remain, self.k - sinter)
        for i in range(0, self.d):
            if i in pubset:
                pub[i] = 1
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
        self.frate = (
            comb(self.d - 1, self.k - 1)
            + (interCount * self.k - comb(self.d + self.m - 1, self.k - 1) * self.m)
            * np.exp(self.ep)
            / self.d
        ) / normalizer

    @staticmethod
    def bestSubsetSize(d, m, ep):
        return PrivSet.bestSubsetSize(d, m, ep)

    def estimate_counts(self, domain, hits):
        array = np.sum(hits, axis=0)
        count_dict = dict(zip(domain, array))
        return [count_dict.get(x, 0) for x in domain]

    def decoder(self, domain, hits):
        # debias hits (may be outside [0,1], not projected to simplex)
        array = np.sum(hits, axis=0)
        count_dict = dict(zip(domain, array))
        num = len(hits)
        es_data = []
        for x in domain:
            x_count = count_dict.get(x, 0)
            fs = (x_count / num - self.frate) / (self.trate - self.frate)
            es_data.append(fs)
        return es_data


# -------------------------
# Helpers
# -------------------------
def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


def run(data: list, domain: list, m: int, ep, k):
    per_data = []
    d = len(domain)
    privset = PrivSet(d, m, ep, k)
    for x in data:
        per_data.append(privset.randomizer(x, domain).tolist())
    return per_data


def frequency_es(per_data: list, domain: list, d: int, m: int, ep, k):
    server = PrivSet_SERVER(d, m, ep, k)
    ct = server.estimate_counts(domain, per_data)
    fs = server.decoder(domain, per_data)
    return ct, fs


def output_attack(ep: float, c: int, r_item: list, r_fre: list, n: int, C_dict, d, k):
    """
    Solve minimal u and forged perturbed counts per target (rounded).
    Returns: [u, S_u(v1), S_u(v2), ...]
    """
    privset_server = PrivSet_SERVER(d, c, ep, k)
    p = privset_server.trate
    q = privset_server.frate

    rlen = len(r_item)
    a, b, c0 = [], [], []
    for i in range(rlen):
        t1 = r_fre[i] * (p - q) + q
        t2 = t1 * n
        t3 = C_dict[r_item[i]]
        a.append(t1)
        b.append(t2)
        c0.append(t3)

    u = 0
    while True:
        u += 1
        ok = True
        for i in range(rlen):
            tt = a[i] * u + b[i] - c0[i]
            if tt < 0 or tt > u:
                ok = False
                break
        if ok:
            res = [u]
            for i in range(rlen):
                res.append(int(round(a[i] * u + b[i] - c0[i])))
            return res


def generate_fake_data(att_result: list, r_item: list, remain_list: list, d, c, ep, k):
    """
    Construct forged *perturbed reports* as sets, then later convert to binary.
    Here each forged report is a set of size m = PrivSet_SERVER.k.
    """
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    privset_server = PrivSet_SERVER(d, c, ep, k)
    m = privset_server.k  # report size in PrivSet

    # If need more users to place all target occurrences (one report can carry multiple targets),
    # keep your original logic but ensure we do not drop already-placed targets.
    if sum(r_count) > u:
        u = sum(r_count)

    fake_data = [[] for _ in range(u)]
    index = 0
    for (item, cnt) in r_dict.items():
        for i in range(cnt):
            fake_data[(index + i) % u].append(item)
        index += cnt

    result = []
    for x in fake_data:
        if len(x) > m:
            result.append(random.sample(x, m))
        elif len(x) < m:
            # IMPORTANT: extend (do not overwrite), to keep placed targets
            x = x + random.sample(remain_list, m - len(x))
            result.append(x)
        else:
            result.append(x)
    return result


def convert_binary(domain: list, data_list: list):
    out = []
    for x in data_list:
        temp = [0] * len(domain)
        for i in x:
            temp[domain.index(i)] = 1
        out.append(temp)
    return out


def column_avg(matrix):
    arr = np.asarray(matrix, dtype=float)
    return arr.mean(axis=0)


def round_list(lst, nd=6):
    return [round(float(v), nd) for v in lst]


def mse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean((a - b) ** 2))




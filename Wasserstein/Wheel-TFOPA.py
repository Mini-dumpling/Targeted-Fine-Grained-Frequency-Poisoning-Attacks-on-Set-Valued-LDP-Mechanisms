import numpy as np
import random
import math
import xxhash


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
# Wheel mechanism (your code)
# -------------------------
def user(epsilon, domain, data_lists):
    data = []
    for data_list in data_lists:
        wu = Wheel_USER(epsilon, domain, data_list)
        wu.run()
        data.append(wu.get_per_data())
    return data


def server(epsilon, domain, data, c):
    ws = Wheel_SERVER(epsilon, domain, data, c)
    ct = ws.estimate()
    return ct, ws.get_es_data()


def data_estimate(N, c, epsilon, D, Ct, fake_data):
    k = len(D)
    Estimate_Dist = [0.0 for _ in range(k)]
    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p

    all_n = N + len(fake_data)

    data_list = [random.choice(x) for x in fake_data]  # one chosen item per forged report
    all_data_count = []
    for i in range(k):
        all_data_count.append(Ct[i] + data_list.count(D[i]))

    for i in range(k):
        Estimate_Dist[i] = (all_data_count[i] - all_n * pf) / (all_n * (pt - pf))

    return Estimate_Dist


class Wheel_USER(object):
    def __init__(self, epsilon: float, domain: list, data_list: list):
        self.epsilon = epsilon
        self.domain = domain
        self.data_list = data_list
        self.c = len(data_list)
        self.per_data = 0

    def run(self):
        epsilon = self.epsilon
        c = self.c
        X = [self.data_list]
        N = 1
        seed = random.randint(0, 100000)

        max_int_32 = (1 << 32) - 1
        Y = [0 for _ in range(N)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        omega = c * temp_p * s + (1 - c * temp_p)

        for i in range(N):
            V = [0 for _ in range(c)]
            for j in range(c):
                V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=seed) / max_int_32

            bSize = math.ceil(1 / temp_p)
            lef = [0 for _ in range(bSize)]
            rig = [0 for _ in range(bSize)]
            for b in range(bSize):
                lef[b] = min((b + 1) * temp_p, 1.0)
                rig[b] = b * temp_p

            for v in V:
                temp_b = math.ceil(v / temp_p) - 1
                lef[temp_b] = min(v, lef[temp_b])
                if temp_b < math.ceil(1 / temp_p) - 1:
                    rig[temp_b + 1] = max(v + temp_p, rig[temp_b + 1])
                else:
                    rig[0] = max(v + temp_p - 1, rig[0])

            temp_rig0 = rig[0]
            for b in range(bSize - 1):
                lef[b] = max(lef[b], rig[b])
                rig[b] = rig[b + 1]
            lef[bSize - 1] = max(lef[bSize - 1], rig[bSize - 1])
            rig[bSize - 1] = temp_rig0 + 1.0

            ll = 0.0
            for b in range(bSize):
                ll += rig[b] - lef[b]

            rnum = np.random.random_sample()
            a = 0.0
            for b in range(bSize):
                a = a + s * (rig[b] - lef[b]) / omega
                if a > rnum:
                    z = rig[b] - (a - rnum) * omega / s
                    break
                a = a + (omega - ll * s) * (
                    lef[(b + 1) % round(bSize)] + math.floor((b + 1) * temp_p) - rig[b]
                ) / ((1 - ll) * omega)
                if a > rnum:
                    z = lef[(b + 1) % bSize] - (a - rnum) * (1 - ll) * omega / (omega - ll * s)
                    break

            z = z % 1.0
            Y[i] = z

        self.per_data = [seed, Y[0]]

    def get_per_data(self):
        return self.per_data


class Wheel_SERVER(object):
    def __init__(self, epsilon: float, domain: list, per_datalist: list, c: int):
        self.epsilon = epsilon
        self.domain = domain
        self.n = len(per_datalist)
        self.c = c
        self.count = []
        self.es_data = []
        self.per_datalist = []
        self.seed = []
        for x in per_datalist:
            self.seed.append(x[0])
            self.per_datalist.append(x[1])

    def estimate(self):
        Y = self.per_datalist
        N = self.n
        c = self.c
        epsilon = self.epsilon
        D = self.domain

        max_int_32 = (1 << 32) - 1
        k = len(D)
        Ct = [0 for _ in range(k)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)

        for i in range(N):
            z = Y[i]
            for j in range(k):
                x = D[j]
                v = xxhash.xxh32_intdigest(str(x), seed=self.seed[i]) / max_int_32
                if (z - temp_p < v <= z) or (z - temp_p + 1 < v < 1):
                    Ct[j] += 1

        self.count = Ct
        return self.count

    def get_es_data(self):
        N = self.n
        c = self.c
        epsilon = self.epsilon
        D = self.domain
        Ct = self.count

        k = len(D)
        Estimate_Dist = [0.0 for _ in range(k)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
        pf = temp_p

        for i in range(k):
            Estimate_Dist[i] = (Ct[i] - N * pf) / (N * (pt - pf))

        self.es_data = Estimate_Dist
        return self.es_data


# -------------------------
# Attack helpers (TFOPA)
# -------------------------
def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


def output_attack(epsilon: float, c: int, r_item: list, r_fre: list, n: int, S_dict):
    r = len(r_item)
    r_solve1, r_solve2, r_solve3 = [], [], []

    s = math.exp(epsilon)
    temp_p = 1 / (2 * c - 1 + c * s)
    pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
    pf = temp_p

    for i in range(r):
        t1 = r_fre[i] * (pt - pf) + pf
        t2 = t1 * n
        t3 = S_dict[r_item[i]]
        r_solve1.append(t1)
        r_solve2.append(t2)
        r_solve3.append(t3)

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
            res = [u]
            for i in range(r):
                res.append(int(round(r_solve1[i] * u + r_solve2[i] - r_solve3[i])))
            return res


def generate_fake_data_wheel(att_result: list, r_item: list, remain_list: list):
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    # each forged report contributes one chosen item in data_estimate()
    if sum(r_count) > u:
        u = sum(r_count)

    fake_data = [[] for _ in range(u)]
    index = 0
    for (k, v) in r_dict.items():
        for i in range(v):
            fake_data[(index + i) % u].append(k)
        index += v

    # ensure each forged report is non-empty
    for x in fake_data:
        if len(x) == 0:
            x.append(random.choice(remain_list))

    return fake_data


# -------------------------
# Metrics helpers (same output style)
# -------------------------
def calculate_column_averages(matrix):
    if not matrix:
        return []
    arr = np.asarray(matrix, dtype=float)
    return arr.mean(axis=0).tolist()


def round_list_values(lst, ndigits=6):
    return [round(float(v), ndigits) for v in lst]


def calculate_mse(list1, list2):
    a = np.asarray(list1, dtype=float)
    b = np.asarray(list2, dtype=float)
    return float(np.mean((a - b) ** 2))


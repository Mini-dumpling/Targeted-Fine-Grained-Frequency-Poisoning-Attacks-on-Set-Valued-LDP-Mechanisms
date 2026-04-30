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
# Helpers
# -------------------------
def count_occurrences(A, B):
    result_dict = {}
    for element in A:
        count = 0
        for row in B:
            count += row.count(element)
        result_dict[element] = count
    return result_dict


def generate_data(domain: list, n: int, c: int):
    return [random.sample(domain, c) for _ in range(n)]


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
    if a.shape != b.shape:
        raise ValueError("Length mismatch")
    return float(np.mean((a - b) ** 2))


# -------------------------
# Wheel mechanism (from your code, refactored as functions)
# -------------------------
class Wheel_USER:
    def __init__(self, epsilon: float, domain: list, data_list: list):
        self.epsilon = epsilon
        self.domain = domain
        self.data_list = data_list
        self.c = len(data_list)
        self.per_data = None

    def run(self):
        epsilon = self.epsilon
        c = self.c
        X = [self.data_list]  # keep same structure as your original code
        N = 1

        # choose hash seed per user
        seed = random.randint(0, 100000)

        max_int_32 = (1 << 32) - 1
        Y = [0.0 for _ in range(N)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)
        omega = c * temp_p * s + (1 - c * temp_p)

        for i in range(N):
            V = [0.0 for _ in range(c)]
            for j in range(c):
                V[j] = xxhash.xxh32_intdigest(str(X[i][j]), seed=seed) / max_int_32

            bSize = math.ceil(1 / temp_p)
            lef = [0.0 for _ in range(bSize)]
            rig = [0.0 for _ in range(bSize)]
            for b in range(bSize):
                lef[b] = min((b + 1) * temp_p, 1.0)
                rig[b] = b * temp_p

            # interval merge
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
                a = a + (omega - ll * s) * (lef[(b + 1) % round(bSize)] +
                                            math.floor((b + 1) * temp_p) - rig[b]) / ((1 - ll) * omega)
                if a > rnum:
                    z = lef[(b + 1) % bSize] - (a - rnum) * (1 - ll) * omega / (omega - ll * s)
                    break

            z = z % 1.0
            Y[i] = z

        self.per_data = [seed, Y[0]]

    def get_per_data(self):
        return self.per_data


class Wheel_SERVER:
    def __init__(self, epsilon: float, domain: list, per_datalist: list, c: int):
        self.epsilon = epsilon
        self.domain = domain
        self.n = len(per_datalist)
        self.c = c
        self.es_data = []

        self.seed = [x[0] for x in per_datalist]
        self.per_y = [x[1] for x in per_datalist]

    def estimate(self):
        Y = self.per_y
        N = self.n
        c = self.c
        epsilon = self.epsilon
        D = self.domain

        max_int_32 = (1 << 32) - 1
        k = len(D)
        Estimate_Dist = [0 for _ in range(k)]
        s = math.exp(epsilon)
        temp_p = 1 / (2 * c - 1 + c * s)

        for i in range(N):
            z = Y[i]
            for j in range(k):
                x = D[j]
                v = xxhash.xxh32_intdigest(str(x), seed=self.seed[i]) / max_int_32
                if (z - temp_p < v <= z) or (z - temp_p + 1 < v < 1):
                    Estimate_Dist[j] += 1

        pt = temp_p * s / (c * temp_p * s + (1 - c * temp_p))
        pf = temp_p
        for i in range(k):
            Estimate_Dist[i] = (1 / N) * (Estimate_Dist[i] - N * pf) / (pt - pf)

        self.es_data = Estimate_Dist

    def get_es_data(self):
        return self.es_data


def wheel_user_batch(epsilon, domain, data_lists):
    out = []
    for data_list in data_lists:
        wu = Wheel_USER(epsilon, domain, data_list)
        wu.run()
        out.append(wu.get_per_data())
    return out


def wheel_server_estimate(epsilon, domain, per_data, c):
    ws = Wheel_SERVER(epsilon, domain, per_data, c)
    ws.estimate()
    return ws.get_es_data()


# -------------------------
# TFIPA solver + fake data generation (from your code logic)
# -------------------------
def input_attack(r_item: list, r_fre: list, n: int, count_dict: dict):
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
            result = [u]
            for i in range(r):
                result.append(int(round(r_solve1[i] * u + r_solve2[i] - r_solve3[i])))
            return result


def generate_fake_data(att_result: list, r_item: list, remain_list: list, c: int):
    u = int(att_result[0])
    r_count = list(map(int, att_result[1:]))
    r_dict = dict(zip(r_item, r_count))

    if sum(r_count) > (u * c):
        u = math.ceil(sum(r_count) / c)

    fake_data = [[] for _ in range(u)]
    index = 0
    for (k, v) in r_dict.items():
        for i in range(v):
            fake_data[(index + i) % u].append(k)
        index += v

    for idx, x in enumerate(fake_data):
        if len(x) < c:
            x.extend(random.sample(remain_list, c - len(x)))
        elif len(x) > c:
            fake_data[idx] = random.sample(x, c)

    return fake_data


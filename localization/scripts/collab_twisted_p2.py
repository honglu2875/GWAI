from fractions import Fraction
from functools import lru_cache
from itertools import product, permutations
from math import factorial
from collections import Counter
import argparse


MARK_DESCENDANT = 4
MARK_H_POWER = 1


def double_factorial_odd(n):
    if n == -1 or n == 0:
        return 1
    res = 1
    for k in range(n, 0, -2):
        res *= k
    return res


def stable(g, n):
    return 2 * g - 2 + n > 0


@lru_cache(None)
def psi_int(g, ds):
    ds = tuple(sorted(ds))
    n = len(ds)
    if any(d < 0 for d in ds):
        return Fraction(0)
    if not stable(g, n):
        if g == 0 and n == 3 and all(d == 0 for d in ds):
            return Fraction(1)
        return Fraction(0)
    if sum(ds) != 3 * g - 3 + n:
        return Fraction(0)
    if g == 0:
        num = factorial(n - 3)
        den = 1
        for d in ds:
            den *= factorial(d)
        return Fraction(num, den)
    if 0 in ds:
        arr = list(ds)
        idx = arr.index(0)
        arr.pop(idx)
        total = Fraction(0)
        for j, d in enumerate(arr):
            if d > 0:
                new = arr.copy()
                new[j] = d - 1
                total += psi_int(g, tuple(new))
        return total
    if g == 1 and n == 1 and ds[0] == 1:
        return Fraction(1, 24)
    arr = list(ds)
    d1 = arr[-1]
    rest = arr[:-1]
    denom = double_factorial_odd(2 * d1 + 1)
    total = Fraction(0)
    for j, dj in enumerate(rest):
        coeff = Fraction(
            double_factorial_odd(2 * d1 + 2 * dj - 1),
            double_factorial_odd(2 * dj - 1),
        )
        new = rest.copy()
        new.pop(j)
        new.append(d1 + dj - 1)
        total += coeff * psi_int(g, tuple(new))
    if d1 >= 2:
        others = rest
        m = len(others)
        for a in range(d1 - 1):
            b = d1 - 2 - a
            coeff = Fraction(
                double_factorial_odd(2 * a + 1) * double_factorial_odd(2 * b + 1),
                2,
            )
            total += coeff * psi_int(g - 1, tuple(list(others) + [a, b]))
            for g1 in range(g + 1):
                g2 = g - g1
                for mask in range(1 << m):
                    s_vals = [others[i] for i in range(m) if (mask >> i) & 1]
                    t_vals = [others[i] for i in range(m) if not ((mask >> i) & 1)]
                    val1 = psi_int(g1, tuple(s_vals + [a]))
                    if val1 == 0:
                        continue
                    val2 = psi_int(g2, tuple(t_vals + [b]))
                    if val2 == 0:
                        continue
                    total += coeff * val1 * val2
    return total / denom


def lambda_g_const(g):
    if g == 0:
        return Fraction(1)
    if g == 1:
        return Fraction(1, 24)
    if g == 2:
        return Fraction(7, 5760)
    raise NotImplementedError


@lru_cache(None)
def lambda_g_int(g, ds):
    ds = tuple(ds)
    n = len(ds)
    if not stable(g, n):
        return Fraction(0)
    if sum(ds) != 2 * g - 3 + n:
        return Fraction(0)
    if g == 0:
        return psi_int(0, ds)
    b = lambda_g_const(g)
    num = factorial(2 * g + n - 3)
    den = 1
    for d in ds:
        den *= factorial(d)
    return Fraction(num, den) * b


@lru_cache(None)
def lambda1_int(g, ds):
    ds = tuple(ds)
    n = len(ds)
    if not stable(g, n):
        return Fraction(0)
    if g == 0:
        return Fraction(0)
    if sum(ds) != 3 * g - 4 + n:
        return Fraction(0)
    if g == 1:
        return lambda_g_int(1, ds)
    if g != 2:
        raise NotImplementedError
    kappa = psi_int(g, tuple(list(ds) + [2]))
    psisum = Fraction(0)
    for i, d in enumerate(ds):
        new = list(ds)
        new[i] = d + 1
        psisum += psi_int(g, tuple(new))
    boundary = Fraction(0)
    boundary += Fraction(1, 2) * psi_int(g - 1, tuple(list(ds) + [0, 0]))
    indices = list(range(n))
    for h in range(g + 1):
        g2 = g - h
        for mask in range(1 << n):
            s_idx = [i for i in indices if (mask >> i) & 1]
            t_idx = [i for i in indices if not ((mask >> i) & 1)]
            if not stable(h, len(s_idx) + 1):
                continue
            if not stable(g2, len(t_idx) + 1):
                continue
            ds1 = [ds[i] for i in s_idx] + [0]
            ds2 = [ds[i] for i in t_idx] + [0]
            boundary += Fraction(1, 2) * psi_int(h, tuple(ds1)) * psi_int(g2, tuple(ds2))
    return Fraction(1, 12) * (kappa + boundary - psisum)


C_l1l2_g2 = Fraction(1, 2880)


@lru_cache(None)
def lambda1lambda2_int_g2(ds):
    ds = tuple(ds)
    g = 2
    n = len(ds)
    if not stable(g, n):
        return Fraction(0)
    if sum(ds) != n:
        return Fraction(0)
    for idx, d in enumerate(ds):
        if d == 0:
            arr = list(ds)
            arr.pop(idx)
            total = Fraction(0)
            for j, dj in enumerate(arr):
                if dj > 0:
                    new = arr.copy()
                    new[j] = dj - 1
                    total += lambda1lambda2_int_g2(tuple(new))
            return total
    den = 1
    for k in ds:
        den *= double_factorial_odd(2 * k - 1)
    factor = Fraction(
        factorial(2 * g + n - 3) * double_factorial_odd(2 * g - 1),
        factorial(2 * g - 1) * den,
    )
    return factor * C_l1l2_g2


@lru_cache(None)
def hodge_int(g, ds, l1pow, l2pow):
    ds = tuple(ds)
    if g == 0:
        if l1pow == 0 and l2pow == 0:
            return psi_int(0, ds)
        return Fraction(0)
    if g == 1:
        if l2pow != 0:
            return Fraction(0)
        if l1pow == 0:
            return psi_int(1, ds)
        if l1pow == 1:
            return lambda1_int(1, ds)
        return Fraction(0)
    if g == 2:
        terms = {(l1pow, l2pow): Fraction(1)}
        changed = True
        while changed:
            changed = False
            newterms = {}
            for (a, b), c in terms.items():
                if c == 0:
                    continue
                if b >= 2:
                    changed = True
                    continue
                if a >= 4:
                    changed = True
                    continue
                if a >= 2:
                    newterms[(a - 2, b + 1)] = newterms.get((a - 2, b + 1), Fraction(0)) + c * 2
                    changed = True
                else:
                    newterms[(a, b)] = newterms.get((a, b), Fraction(0)) + c
            terms = newterms
        total = Fraction(0)
        for (a, b), c in terms.items():
            if b >= 2:
                continue
            if a == 0 and b == 0:
                val = psi_int(2, ds)
            elif a == 1 and b == 0:
                val = lambda1_int(2, ds)
            elif a == 0 and b == 1:
                val = lambda_g_int(2, ds)
            elif a == 1 and b == 1:
                val = lambda1lambda2_int_g2(ds)
            else:
                val = Fraction(0)
            total += c * val
        return total
    raise NotImplementedError


def reduce_hodge_poly(poly, g):
    res = {}
    for (a, b), coeff in poly.items():
        if coeff == 0:
            continue
        if g == 0:
            if a == 0 and b == 0:
                res[(0, 0)] = res.get((0, 0), Fraction(0)) + coeff
        elif g == 1:
            if b == 0 and a == 0:
                res[(0, 0)] = res.get((0, 0), Fraction(0)) + coeff
            elif b == 0 and a == 1:
                res[(1, 0)] = res.get((1, 0), Fraction(0)) + coeff
        elif g == 2:
            terms = {(a, b): coeff}
            changed = True
            while changed:
                changed = False
                new = {}
                for (aa, bb), cc in terms.items():
                    if cc == 0:
                        continue
                    if bb >= 2 or aa >= 4:
                        changed = True
                        continue
                    if aa >= 2:
                        new[(aa - 2, bb + 1)] = new.get((aa - 2, bb + 1), Fraction(0)) + 2 * cc
                        changed = True
                    else:
                        new[(aa, bb)] = new.get((aa, bb), Fraction(0)) + cc
                terms = new
            for k, c in terms.items():
                if k[1] < 2:
                    res[k] = res.get(k, Fraction(0)) + c
        else:
            raise NotImplementedError
    return {k: v for k, v in res.items() if v}


def poly_mul(p, q, g):
    temp = {}
    for (a, b), c in p.items():
        for (a2, b2), d in q.items():
            temp[(a + a2, b + b2)] = temp.get((a + a2, b + b2), Fraction(0)) + c * d
    return reduce_hodge_poly(temp, g)


def hodge_factor_poly_for_weight(g, weight):
    if g == 0:
        return {(0, 0): Fraction(1)}
    if g == 1:
        return reduce_hodge_poly({(0, 0): weight, (1, 0): Fraction(-1)}, g)
    if g == 2:
        return reduce_hodge_poly({(0, 0): weight * weight, (1, 0): -weight, (0, 1): Fraction(1)}, g)
    raise NotImplementedError


class Graph:
    def __init__(self, colors, genera, mark_vertex, edges):
        self.colors = tuple(colors)
        self.genera = tuple(genera)
        self.mark = mark_vertex
        self.edges = tuple(tuple(e) for e in edges)
        self.nv = len(colors)

    def canonical_key(self):
        n = self.nv
        best = None
        for p in permutations(range(n)):
            inv = [None] * n
            for new, old in enumerate(p):
                inv[old] = new
            verts = tuple((self.colors[old], self.genera[old], 1 if old == self.mark else 0) for old in p)
            ed = []
            for u, v, d in self.edges:
                uu = inv[u]
                vv = inv[v]
                if uu > vv:
                    uu, vv = vv, uu
                ed.append((uu, vv, d))
            key = (verts, tuple(sorted(ed)))
            if best is None or key < best:
                best = key
        return best

    def vertex_perm_aut_count(self):
        n = self.nv
        cnt = 0
        for p in permutations(range(n)):
            ok = True
            for i in range(n):
                j = p[i]
                if (
                    self.colors[i] != self.colors[j]
                    or self.genera[i] != self.genera[j]
                    or ((i == self.mark) != (j == self.mark))
                ):
                    ok = False
                    break
            if not ok:
                continue
            old_edges = sorted((min(u, v), max(u, v), d) for u, v, d in self.edges)
            new_edges = []
            for u, v, d in self.edges:
                uu = p[u]
                vv = p[v]
                if uu > vv:
                    uu, vv = vv, uu
                new_edges.append((uu, vv, d))
            if sorted(new_edges) == old_edges:
                cnt += 1
        return cnt

    def edge_mult_factor(self):
        counts = {}
        for u, v, d in self.edges:
            if u > v:
                u, v = v, u
            counts[(u, v, d)] = counts.get((u, v, d), 0) + 1
        res = 1
        for m in counts.values():
            res *= factorial(m)
        return res

    def aut_graph(self):
        return self.vertex_perm_aut_count() * self.edge_mult_factor()

    def deck_factor(self):
        res = 1
        for _, _, d in self.edges:
            res *= d
        return res

    def incidence(self):
        inc = [[] for _ in range(self.nv)]
        for ei, (u, v, d) in enumerate(self.edges):
            inc[u].append((ei, v, d))
            inc[v].append((ei, u, d))
        return inc

    def __repr__(self):
        return f"Graph(colors={self.colors}, genera={self.genera}, mark={self.mark}, edges={self.edges})"


def allowed_vertex(g, edge_val, has_mark):
    val = edge_val + (1 if has_mark else 0)
    if stable(g, val):
        return True
    if g != 0:
        return False
    if (not has_mark) and edge_val in (1, 2):
        return True
    if has_mark and edge_val == 1:
        return True
    return False


def gen_graphs():
    graphs = []
    seen = set()
    edge_configs = [
        (2, [(0, 1, 2)]),
        (2, [(0, 1, 1), (0, 1, 1)]),
        (3, [(0, 1, 1), (1, 2, 1)]),
    ]
    for nv, edges in edge_configs:
        for colors in product(range(3), repeat=nv):
            ok = True
            for u, v, d in edges:
                if colors[u] == colors[v]:
                    ok = False
                    break
            if not ok:
                continue
            e_count = len(edges)
            h1 = e_count - nv + 1
            total_g = 2 - h1
            if total_g < 0:
                continue
            for genera in product(range(total_g + 1), repeat=nv):
                if sum(genera) != total_g:
                    continue
                for mark in range(nv):
                    inc_counts = [0] * nv
                    for u, v, d in edges:
                        inc_counts[u] += 1
                        inc_counts[v] += 1
                    if all(allowed_vertex(genera[v], inc_counts[v], v == mark) for v in range(nv)):
                        graph = Graph(colors, genera, mark, edges)
                        key = graph.canonical_key()
                        if key not in seen:
                            seen.add(key)
                            graphs.append(graph)
    return graphs


def eT(color, lambdas):
    res = Fraction(1)
    for j in range(3):
        if j != color:
            res *= lambdas[color] - lambdas[j]
    return res


def edge_normal_factor(ci, cj, d, lambdas):
    diff = lambdas[ci] - lambdas[cj]
    res = eT(ci, lambdas) * eT(cj, lambdas)
    res *= Fraction(((-1) ** d) * (d ** (2 * d)), (factorial(d) ** 2))
    res /= diff ** (2 * d)
    for k in range(3):
        if k == ci or k == cj:
            continue
        prodw = Fraction(1)
        for a in range(d + 1):
            prodw *= Fraction((d - a) * (lambdas[ci] - lambdas[k]) + a * (lambdas[cj] - lambdas[k]), d)
        res /= prodw
    return res


def edge_twist_factor(ci, cj, d, lambdas):
    wi = -lambdas[ci]
    wj = -lambdas[cj]
    res = Fraction(1)
    for a in range(1, d):
        res *= Fraction((d - a) * wi + a * wj, d)
    return res


def stable_vertex_factor(graph, v, lambdas):
    inc = graph.incidence()[v]
    g = graph.genera[v]
    ci = graph.colors[v]
    has_mark = v == graph.mark
    edge_val = len(inc)
    nslots = edge_val + (1 if has_mark else 0)
    dim = 3 * g - 3 + nslots
    poly = {(0, 0): Fraction(1)}
    for j in range(3):
        if j == ci:
            continue
        a = lambdas[ci] - lambdas[j]
        poly = poly_mul(poly, hodge_factor_poly_for_weight(g, a), g)
    w = -lambdas[ci]
    twist_poly = hodge_factor_poly_for_weight(g, w)
    power = edge_val - 1
    if power < 0:
        scalar = Fraction(1, w ** (-power))
    else:
        scalar = w ** power
    twist_poly = {k: scalar * c for k, c in twist_poly.items()}
    poly = poly_mul(poly, twist_poly, g)
    base = Fraction(1, 1) / eT(ci, lambdas)
    if has_mark:
        base *= lambdas[ci] ** MARK_H_POWER
    total = Fraction(0)
    ranges = [range(dim + 1) for _ in inc]
    for flag_pows in product(*ranges):
        coeff = base
        psi_exps = []
        for powa, (_, nb, d) in zip(flag_pows, inc):
            omega = Fraction(lambdas[ci] - lambdas[graph.colors[nb]], d)
            coeff *= Fraction(1, omega ** (powa + 1))
            psi_exps.append(powa)
        if has_mark:
            psi_exps.append(MARK_DESCENDANT)
        psisum = sum(psi_exps)
        for (l1, l2), pc in poly.items():
            hdeg = l1 + 2 * l2
            if psisum + hdeg != dim:
                continue
            val = hodge_int(g, tuple(psi_exps), l1, l2)
            if val:
                total += coeff * pc * val
    return total


def unstable_vertex_factor(graph, v, lambdas):
    inc = graph.incidence()[v]
    g = graph.genera[v]
    ci = graph.colors[v]
    has_mark = v == graph.mark
    assert g == 0
    edge_val = len(inc)
    normal = Fraction(1, 1) / eT(ci, lambdas)
    twist = Fraction(1)
    insertion = Fraction(1)
    if (not has_mark) and edge_val == 1:
        _, nb, d = inc[0]
        omega = Fraction(lambdas[ci] - lambdas[graph.colors[nb]], d)
        normal *= omega
    elif (not has_mark) and edge_val == 2:
        omega_sum = Fraction(0)
        for _, nb, d in inc:
            omega_sum += Fraction(lambdas[ci] - lambdas[graph.colors[nb]], d)
        normal *= Fraction(1, omega_sum)
        twist *= -lambdas[ci]
    elif has_mark and edge_val == 1:
        _, nb, d = inc[0]
        omega = Fraction(lambdas[ci] - lambdas[graph.colors[nb]], d)
        insertion *= (lambdas[ci] ** MARK_H_POWER) * ((-omega) ** MARK_DESCENDANT)
    else:
        raise ValueError(("bad unstable", graph, v, edge_val, has_mark))
    return normal * twist * insertion


def graph_contribution(graph, lambdas):
    inc = graph.incidence()
    prodv = Fraction(1)
    for v in range(graph.nv):
        edge_val = len(inc[v])
        has_mark = v == graph.mark
        g = graph.genera[v]
        if stable(g, edge_val + (1 if has_mark else 0)):
            vf = stable_vertex_factor(graph, v, lambdas)
        else:
            vf = unstable_vertex_factor(graph, v, lambdas)
        prodv *= vf
        if prodv == 0:
            break
    prode = Fraction(1)
    for u, v, d in graph.edges:
        prode *= edge_normal_factor(graph.colors[u], graph.colors[v], d, lambdas)
        prode *= edge_twist_factor(graph.colors[u], graph.colors[v], d, lambdas)
    aut = graph.aut_graph() * graph.deck_factor()
    return prodv * prode / Fraction(aut, 1)


def evaluate(lambdas):
    graphs = gen_graphs()
    total = Fraction(0)
    nz = 0
    bytype = Counter()
    for graph in graphs:
        contribution = graph_contribution(graph, lambdas)
        if contribution:
            nz += 1
            bytype[len(graph.edges)] += contribution
        total += contribution
    return graphs, nz, total, bytype


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--descendant", type=int, default=4)
    parser.add_argument("--h-power", type=int, default=1)
    args = parser.parse_args()
    MARK_DESCENDANT = args.descendant
    MARK_H_POWER = args.h_power

    print("psi <tau4>_2", psi_int(2, (4,)))
    print("lambda2 psi2 g2 n1", lambda_g_int(2, (2,)))
    print("lambda1 psi3 g2 n1", lambda1_int(2, (3,)))
    print("lambda1lambda2 psi1 g2 n1", lambda1lambda2_int_g2((1,)))
    graphs = gen_graphs()
    print("num graphs", len(graphs))
    print(Counter(len(graph.edges) for graph in graphs))
    for lambdas in [
        (Fraction(1), Fraction(2), Fraction(4)),
        (Fraction(1), Fraction(3), Fraction(7)),
        (Fraction(2), Fraction(5), Fraction(11)),
    ]:
        _, nz, total, bytype = evaluate(lambdas)
        print("lambdas", lambdas, "nz", nz, "total", total, "float", float(total))
        print("by edge count", bytype)

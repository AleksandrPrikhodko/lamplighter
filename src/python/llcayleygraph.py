import numpy as np
from typing import List

from math.extra_math import random_3d_u
from graphs.graph import NonOrientedGraph
from lamplighter import LLElement

GOLDEN_RATIO = (np.sqrt(5) + 1)/2


class LLCayleyGraph(NonOrientedGraph):
    def __init__(self, *, secondary=False):
        super().__init__()
        self.n_neighbours = {}
        self.distance_to_origin = {}
        self.coo = {}
        self.W = {}
        self.n_moves = 0
        self.n_expand = 0
        self.origin = None
        self.z0 = None
        self.z1 = None
        if not secondary:
            self.g0 = LLCayleyGraph(secondary=True)
            self.g0.create_origin()

    def create_origin(self):
        origin = LLElement(0, 0)
        self.vertexes[origin] = self._empty_vertex_info()
        self.distance_to_origin[origin] = 0
        self.n_neighbours[origin] = 4
        self.coo[origin] = np.zeros(3)
        self.W[origin] = 1
        self.origin = origin

    def node_is_expanded(self, node):
        return self.n_neighbours[node] == len(self.vertex_edges(node))

    def assign_coo(self, u, v):
        if self.distance_to_origin[u] == 1:
            x1 = GOLDEN_RATIO * u.t
            y1 = -1 + 2 * u.mask
            if u.t < 0:
                y1 *= -1
            self.coo[u] = np.array([x1, y1, 0])
        else:
            dp = np.array(self.coo[v])
            n1 = np.linalg.norm(dp)
            if n1 > 0.001:
                dp /= n1
            dp += random_3d_u() * 0.25
            self.coo[u] = self.coo[v] + dp

    def expand_border(self):
        nodes_to_expand = []
        for v, info in self.vertexes.items():
            v: LLElement
            # print("?", v)
            if self.node_is_expanded(v):
                continue
            nodes_to_expand.append(v)
        for v in nodes_to_expand:
            # print("Expanding {}".format(v))
            ngb = v.neighbours()
            for u in ngb:
                if u not in self.vertexes:
                    self.connect(v, u)
                    self.distance_to_origin[u] = self.distance_to_origin[v] + 1
                    self.n_neighbours[u] = 4
                    self.W[u] = 0
                    self.assign_coo(u, v)
                    for v2 in u.neighbours():
                        if v2 in self.vertexes:
                            self.connect(u, v2)

    def expand_border1(self):
        print("~~")
        print(self.vertexes)
        v_to_add = []
        for v in self.vertexes:
            for u in v.neighbours():
                _approved = False
                _v2l = []
                for v2 in v.neighbours():
                    if v2 == v:
                        continue
                    if v2 in self.vertexes:
                        _v2l.append(v2)
                        _approved = True
                if _approved:
                    v_to_add.append((u, v, _v2l))
        for u, v, v2l in v_to_add:
            self.vertexes[u] = self._empty_vertex_info()
            self.n_neighbours[u] = 4
            self.distance_to_origin[u] = 2
            self.connect(u, v)
            for v2 in v2l:
                self.connect(u, v2)
                _c = 0.5 * (self.coo[v] + self.coo[v2] + 0.5 * random_3d_u())
                _c = random_3d_u()
            self.coo[u] = _c

        print(list(self.vertexes.keys()))
        # exit()

    def propagate(self, alpha, *, verbose=False):
        W2 = {v: 0 for v in self.vertexes}
        for v in self.vertexes:
            w1 = self.W[v]
            _S = 0
            for u in v.neighbours():
                if u in self.vertexes:
                    dw = alpha * w1
                    W2[u] += dw
                    _S += dw
            W2[v] += w1 - _S
        for v in self.vertexes:
            self.W[v] = W2[v]
        L = [(v, self.W[v]) for v in self.vertexes]
        L.sort(key=lambda r: -r[1])
        if verbose:
            print(L[:8])

    def expand_border2(self):
        self.n_expand += 1
        # if self.n_expand > 20:
        #     return
        print("expand2: {}".format(self.n_expand))
        if self.n_expand < 11:
            self.g0.expand_border()
        self.g0.propagate(0.075)
        max_d0 = 2 * self.n_expand ** 0.5
        max_d = max(3, np.floor(max_d0))
        max_d = self.n_expand
        v_to_add = []
        w0 = self.g0.W[self.g0.origin]
        for v in self.g0.vertexes:
            if self.g0.W[v] >= 0.33 * w0:
                if v not in self.vertexes:
                    v_to_add.append(v)
        print("max_d: {} ({:.1f}),  g0: {},  adding: {},  current: {}".format(
            max_d, max_d0,
            len(self.g0), len(v_to_add), len(self)
        ))
        v_to_add2 = []
        for u in v_to_add:
            siblings = []
            for v2 in u.neighbours():
                if v2 in self.vertexes:
                    if v2 in self.coo:
                        siblings.append([v2, self.distance_to_origin[v2]])
            if len(siblings) == 0:
                # raise RuntimeError("?")
                continue
            v_to_add2.append(u)
            siblings.sort(key=lambda r: r[1])
            v1 = siblings[0][0]
            self.distance_to_origin[u] = siblings[0][1] + 1
            self.assign_coo(u, v1)
        for u in v_to_add2:
            self.vertexes[u] = self._empty_vertex_info()
            self.n_neighbours[u] = 4
            self.W[u] = 0
        for u in v_to_add2:
            _n = self.vertexes[u]["neighbours"]
            for v2 in u.neighbours():
                if v2 in self.vertexes:
                    if v2 not in _n:
                        self.connect(u, v2)

    def binary_expand(self):
        if self.n_expand >= 3:
            return
        self.n_expand += 1
        print("binary expand: {}".format(self.n_expand))

        for v in self.vertexes:
            print("v: {}  ({}, {})".format(v, v.left, v.mask))

        vertexes2 = {}
        _coo = {}
        _distance_to_origin = {}
        for v in self.vertexes:
            v2 = v ** 2
            _coo[v2] = self.coo[v]
            _distance_to_origin[v2] = self.distance_to_origin[v]
            vertexes2[v2] = self._empty_vertex_info()
        # print(list(vertexes2.keys()))

        _vertexes = self.vertexes
        self.vertexes = {v: i for v, i in vertexes2.items()}
        self.coo = _coo
        self.distance_to_origin = _distance_to_origin

        for v in vertexes2:
            for u in v.neighbours():
                if u not in self.vertexes:
                    print()
                    print("u", u)
                    _approved = False
                    _c = None
                    for v2 in u.neighbours():
                        if v2 in vertexes2:
                            if v2 != v:
                                _approved = True
                                _c = 0.5 * (self.coo[v] + self.coo[v2] + 0.25 * random_3d_u())
                    if _approved:
                        print(u, self.n_expand)
                        self.distance_to_origin[u] = self.n_expand + 1
                        self.vertexes[u] = self._empty_vertex_info()
                        self.connect(v, u)
                        print("{} -- {}".format(v, u))
                        if u not in self.coo:
                            self.coo[u] = _c
                        for v2 in u.neighbours():
                            if v2 in vertexes2:
                                if v2 != v:
                                    self.connect(u, v2)
                                    print("    {} -- {}".format(u, v2))
        print("edges:", len(self.get_list_of_edges()))

        for v in self.vertexes:
            self.n_neighbours[v] = 4

        print(list(self.vertexes.keys()))
        print("expanded: {} => {}".format(len(vertexes2), len(self.vertexes)))
        # exit()

    @staticmethod
    def get_avg_t_set(v_set):
        S0, S1 = 0, 0
        for v in v_set:
            S1 += v.t
            S0 += 1
        return S1/S0 if S0 > 0 else 0

    def build_graph_from_v_set(self, v_set):
        for v in v_set:
            self.vertexes[v] = self._empty_vertex_info()
            self.coo[v] = np.array([v.t - 0.5 * self.get_avg_t_set(v_set), 0, 0]) + 0.5 * random_3d_u()
            self.distance_to_origin[v] = 0
        for v in self.vertexes:
            for u in v.neighbours():
                if u in self.vertexes:
                    self.connect(v, u)

    def build1(self, m):
        v_set = []
        for _m in range(2**m):
            for t in range(m):
                v = LLElement(0, _m, t)
                v.reduce()
                v_set.append(v)
        self.build_graph_from_v_set(v_set)

    def get_weighted_o(self, h):
        v_list = []
        w0 = self.W[self.origin] * h
        for v in self.vertexes:
            if self.W[v] >= w0:
                v_list.append(v)
        return v_list

    def build_normalized_ball(self, R, h):
        g0 = LLCayleyGraph()
        g0.create_origin()
        for r in range(R):
            g0.expand_border()
        for it in range(133):
            g0.propagate(0.025)
        g0.propagate(0.025, verbose=True)
        v_list = g0.get_weighted_o(h)
        # print("v_list", v_list)
        print(len(g0.vertexes), len(v_list))

        self.build_graph_from_v_set(v_list)
        # print("restrict", len(self))
        self.origin = g0.origin
        for v in v_list:
            self.distance_to_origin[v] = g0.distance_to_origin[v]

    def structure_init(self, *, z=None, z0=None, z1=None):
        if z is not None:
            self.z0 = -z-1
            self.z1 = z
        else:
            self.z0 = z0
            self.z1 = z1
        vertexes2 = {}
        d2 = {}
        for v in self.vertexes:
            v2 = LLElement(v.left, v.mask, v.t, z0=self.z0, z1=self.z1)
            vertexes2[v2] = self._empty_vertex_info()
            _d = self.distance_to_origin[v]
            if v2 in d2:
                d2[v2] = min(d2[v2], _d)
            else:
                d2[v2] = _d
        print("{} --> {}".format(len(self.vertexes), len(vertexes2)))
        self.vertexes = vertexes2
        self.distance_to_origin = d2
        for v2 in self.vertexes:
            for u in v2.neighbours():
                if u in self.vertexes:
                    self.connect(v2, u)
            self.coo[v2] = random_3d_u()


    def structure_expand(self):
        ...

    def create_rhombus(self):
        v0 = LLElement(0, 0)
        v_set = [v0, v0.b(), v0.a(), v0.a().bi()]
        self.build_graph_from_v_set(v_set)

    def rhombus_expand(self):
        edges = self.get_list_of_edges()
        print(edges)

    def update_edges(self):
        for v in self.vertexes:
            _a = []
            for u in v.neighbours():
                if u in self.vertexes:
                    _a.append(u)
            self.vertexes[v]["neighbours"] = {}
            for u in _a:
                self.vertexes[v]["neighbours"][u] = v

    def remove_hanging1(self):
        self.update_edges()
        n1 = len(self.vertexes)
        r = []
        for v in self.vertexes:
            if len(self.vertexes[v]["neighbours"]) == 1:
                r.append(v)
        found = (len(r) > 0)
        for v in r:
            del self.vertexes[v]
        for v in self.vertexes:
            _a = []
            for u in self.vertexes[v]["neighbours"]:
                if u not in self.vertexes:
                    _a.append(u)
            for u in _a:
                del self.vertexes[v]["neighbours"][u]
        print("rh: {} --> {}".format(n1, len(self.vertexes)))
        return found

    def remove_hanging(self):
        while self.remove_hanging1():
            pass

    def create_simple(self, K):
        self.origin = LLElement(0, 0)
        v_list = set()
        for x in range(2**K):
            v = LLElement(-K+1, x, 0)
            v.reduce()
            v_list.add(v)
        for it in range(K-1):
            set2 = set()
            for v1 in v_list:
                set2.add(v1)
                for u in v1.neighbours():
                    set2.add(u)
            v_list = set2
        self.build_graph_from_v_set(v_list)
        for v in self.vertexes:
            self.distance_to_origin[v] = 0
            self.coo[v] = random_3d_u()
        print(list(self.vertexes.keys()))

    def expand_via_propagate_init(self, h0):
        self.h0 = h0
        self.g0 = LLCayleyGraph()
        self.g0.create_origin()
        for it in range(8):
            self.g0.expand_border()
        for it in range(25):
            self.g0.propagate(0.05, verbose=False)
        self.copy_g0()

    def expand_via_propagate(self, alpha, *, m=10):
        for it in range(m):
            self.g0.propagate(alpha)
        self.copy_g0()

    def copy_g0(self):
        w0 = self.h0 * self.g0.W[self.g0.origin]
        v_list = [v for v in self.g0.vertexes if self.g0.W[v] >= w0]
        if len(v_list) == self.size():
            return
        self.n_expand += 1
        _coo = self.coo.copy()
        _dto = self.distance_to_origin.copy()
        self.build_graph_from_v_set(v_list)
        for v in self.vertexes:
            if v in _coo:
                continue
            _dto[v] = self.n_expand
            _S0, _S1 = 0, np.zeros(3)
            for u in v.neighbours():
                if u in _coo:
                    _S1 += _coo[u]
                    _S0 += 1
            if _S0 == 0:
                _coo[v] = random_3d_u()
            else:
                _coo[v] = _S1/_S0 + 0.25 * random_3d_u()
        self.coo = _coo
        self.distance_to_origin = _dto
        print(list(self.vertexes.keys()))


    def __repr__(self):
        s = "  ".join(["{}".format(x) for x in self.get_list_of_vertexes()])
        return s

    def t_distribution(self) -> List[int]:
        d = {}
        for v in self.vertexes:
            t = v.t
            if t not in d:
                d[t] = 0
            d[t] += 1
        ks = list(d.keys())
        ks.sort()
        distr = [d[k] for k in ks]
        return distr

    def move(self, dt):
        # self.n_moves += 1
        # if self.n_moves >= 2:
        #     return
        deltas = {}
        for v, info in self.vertexes.items():
            p0 = self.coo[v]
            dp = np.zeros(3)
            neighbours = info["neighbours"]
            for u in neighbours:
                p1 = self.coo[u]
                # print(p0, p1)
                e1 = p1 - p0
                n2 = max(1e-2, e1.dot(e1))
                dp += (1 - n2**(-2)) * e1
            for v2 in self.vertexes:
                p2 = self.coo[v2]
                e2 = p2 - p0
                dp += -0.8 * e2 * np.exp(-np.linalg.norm(e2))
            dp *= 2 * dt
            deltas[v] = dp
        for v in self.vertexes:
            self.coo[v] += deltas[v]


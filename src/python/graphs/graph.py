

class Graph:
    def __init__(self):
        self.vertexes = {}
        self.edges = []

    def vertex_edges(self, v):
        return self.vertexes[v]["neighbours"]

    def get_list_of_vertexes(self):
        l = list(self.vertexes.keys())
        l.sort()
        return l

    def get_list_of_edges(self):
        edges = []
        edges_d = {}
        for v in self.vertexes:
            for u in self.vertexes[v]["neighbours"]:
                _a = [u, v]
                _a.sort()
                _k = "{}_{}".format(_a[0], _a[1])
                if _k in edges_d:
                    continue
                edges_d[_k] = 1
                edges.append((v, u))
        return edges

    @staticmethod
    def _empty_vertex_info():
        return {
                "neighbours": {}
            }

    def size(self) -> int:
        return len(self.vertexes)

    def __len__(self):
        return self.size()


class NonOrientedGraph(Graph):
    def __init__(self):
        super().__init__()

    def connect(self, v, u):
        if u not in self.vertexes:
            self.vertexes[u] = self._empty_vertex_info()
        d = self.vertexes[v]["neighbours"]
        if u not in d:
            d[u] = v
        d = self.vertexes[u]["neighbours"]
        if v not in d:
            d[v] = u




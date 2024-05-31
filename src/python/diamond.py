

class Diamond:
    def __init__(self, *, elements=None, start=None):
        self.degree1_inverted = False
        if elements is not None:
            self.elements = elements
            if type(self.first()) is Diamond:
                self.degree = self.first().degree + 1
            else:
                self.degree = 1
        elif start is not None:
            if type(start) is Diamond:
                n1 = start.next()
                n2 = n1.next()
                n3 = n2.next()
                self.elements = [
                    start,
                    n1, n2, n3,
                ]
                self.degree = self.first().degree + 1
            else:
                self.elements = [
                    start, start.b(), start.b().ai(), start.a()
                ]
                self.degree = 1

    def first(self):
        return self.elements[0]

    def invert(self):
        _elements = self.elements[1:]
        _elements.append(self.elements[0])
        d = Diamond(elements=_elements)
        d.degree1_inverted = not self.degree1_inverted
        return d

    def fit(self, d2):
        ...

    def copy(self):
        if self.degree == 1:
            d = Diamond(elements=self.elements.copy())
        else:
            L = [x.copy() for x in self.elements.copy()]
            d = Diamond(elements=L)
        d.degree1_inverted = self.degree1_inverted
        return d

    def next(self):
        right = self.elements[2].copy() if type(self.elements[2]) is Diamond else self.elements[2]
        if self.degree == 1:
            if not self.degree1_inverted:
                d = Diamond(elements=[
                    right, right.ai(), right.ai().b(), right.bi()
                ])
                d.degree1_inverted = True
            else:
                d = Diamond(start=right)
            return d
        else:
            right_i = right.invert()
            return Diamond(start=right_i)

    def vertexes(self):
        if self.degree == 1:
            return set(self.elements)
        else:
            if self.degree <= 2 or True:
                return self.elements[0].vertexes().union(
                    self.elements[1].vertexes().union(
                        self.elements[2].vertexes().union(self.elements[3].vertexes())
                    )
                )
            else:
                # return self.elements[0].vertexes().union(
                #     self.elements[1].vertexes()
                # )
                r = self.elements[2].vertexes().copy()
                print("V", r)
                return r

    def __repr__(self):
        if self.degree == 1:
            return ", ".join([str(x) for x in self.elements]) + (" I" if self.degree1_inverted else "")
        else:
            r = ""
            for i in range(4):
                r += "{}\n".format(i)
                t = self.elements[i].__repr__()
                lines = t.split("\n")
                _a = []
                for line in lines:
                    _a.append("    {}".format(line))
                r += "\n".join(_a) + "\n"
            return r

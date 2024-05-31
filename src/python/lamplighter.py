from __future__ import annotations

from typing import List


class LLElement:
    def __init__(self, left=0, mask=0, t=0, *, z0=None, z1=None):
        self.left = left
        self.mask = mask
        self.t = t
        self.rmsk = None
        self.z0 = z0
        self.z1 = z1
        self.apply_restrictions()
        if z0 is not None:
            print("{} ~> {}".format(LLElement(left, mask, t), self))

    def __hash__(self):
        # if self.rmsk is None:
        #     return (self.left, self.mask, self.t).__hash__()
        # else:
        #     return (self.left, self.mask & self.rmsk, self.t).__hash__()
        return (self.left, self.mask, self.t).__hash__()

    def __eq__(self, other):
        if (self.mask % 2 == 0 and self.mask != 0) or (other.mask % 2 == 0 and other.mask != 0):
            raise RuntimeError("me: {} ({}, {}),  other: {}".format(self, self.left, self.mask, other))
        return (
            self.left == other.left and
            self.mask == other.mask and
            self.t == other.t
        )

    def __pow__(self, power, modulo=None):
        if power != 2:
            raise RuntimeError("Power 3 is only supported")
        t2 = self.t * 2
        left2 = self.left * 2
        h = 1
        x = self.mask
        x2 = 0
        while x > 0:
            if x & 1 != 0:
                x2 += h
            x = x >> 1
            h *= 4
        v2 = LLElement(left2, x2, t2)
        print("{} --> {} ({}, {})".format(self, v2, v2.left, v2.mask))
        return v2

    def __ne__(self, other):
        return not(self == other)

    def __lt__(self, other):
        if self.t < other.t:
            return True
        elif self.t > other.t:
            return False
        if self.left < other.left:
            return True
        elif self.left > other.left:
            return False
        if self.mask < other.mask:
            return True
        elif self.mask > other.mask:
            return False
        return False

    def __gt__(self, other):
        if self == other:
            return False
        if self < other:
            return False
        return True

    def get_copy(self):
        return LLElement(self.left, self.mask, self.t)

    def apply_restrictions(self):
        if self.z0 is None:
            return
        _z0 = self.z0 - self.left
        _z1 = self.z1 - self.left
        if _z1 < 0:
            self.mask = 0
        else:
            _z0 = max(0, _z0)
            _m = ((1 << (_z1 - _z0 + 1)) - 1) << _z0
            self.mask = self.mask & _m
        if self.mask == 0:
            self.left = 0
        self._reduce()

    def b(node) -> LLElement:
        x = node.get_copy()
        # if x.mask != 0:
        #     x.left += 1
        x.t += 1
        x.apply_restrictions()
        return x

    def br(node) -> LLElement:
        x = node.get_copy()
        if x.mask != 0:
            x.left += 1
        x.t += 1
        x.apply_restrictions()
        return x

    def bi(node) -> LLElement:
        x = node.get_copy()
        # if x.mask != 0:
        #     x.left -= 1
        x.t -= 1
        x.apply_restrictions()
        return x

    def _reduce(self):
        if self.mask == 0:
            self.left = 0
        else:
            while self.mask % 2 == 0:
                self.mask //= 2
                self.left += 1

    def reduce(self):
        self._reduce()
        self.apply_restrictions()

    def c(node) -> LLElement:
        x = node.get_copy()
        x.left -= x.t
        if x.left >= 0:
            x.mask = (x.mask << x.left) ^ 1
            x.left = 0
        else:
            x.mask = x.mask ^ (2**(-x.left))
        x.left += x.t
        x.reduce()
        return x

    def a(node) -> LLElement:
        x = node.c().b()
        return x

    def ai(node) -> LLElement:
        return node.bi().c()

    def neighbours(self) -> List[LLElement]:
        return [self.b(), self.bi(), self.a(), self.ai()]

    def compact_repr(self) -> str:
        y = max(0, self.left)
        s = "{0:b}".format(self.mask * (2 ** y))
        h = max(0, -self.left)
        if h == 0:
            r = "|" + s
        else:
            r = s[:h] + "|" + s[h:]
        return "{}({})".format(r, self.t, )

    def repr_with_hash(self) -> str:
        r = self.compact_repr()
        return "{}[{}]".format(r, self.__hash__())

    def __repr__(self):
        r = self.compact_repr()
        return r


class LLGroup:
    def __init__(self):
        self._unity = LLElement(0, 0)

    def unity(self) -> LLElement:
        return self._unity


from abc import ABC, abstractmethod


class Semantics(ABC):

    @abstractmethod
    def conjunction(self, a, b):
        pass

    @abstractmethod
    def disjunction(self, a, b):
        pass

    @abstractmethod
    def negation(self, a):
        pass


class SumProductSemiring(Semantics):
    def conjunction(self, a):
        term = 1
        for elem in a:
            term *= elem
        return term

    def disjunction(self, a):
        return max(a)

    def negation(self, a):
        return 1 - a

class LukasieviczTNorm(Semantics):
    def conjunction(self, a):
        return max(0 * a[0], sum(a) - (len(a) - 1))

    def disjunction(self, a):
        return min(0 * a[0] + 1, sum(a))

    def negation(self, a):
        return 1 - a

class GodelTNorm(Semantics):
    def conjunction(self, a):
        return min(a)

    def disjunction(self, a):
        return max(a)

    def negation(self, a):
        return 1-a

class ProductTNorm(Semantics):
    def conjunction(self, a):
        term = 1
        for elem in a:
            term *= elem
        return term
    
    def disjunction(self, a, b):
        return (a+b)-(a*b)

    def negation(self, a):
        return 1 - a
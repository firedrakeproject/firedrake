"""Linear analysis of the flat indices of variables, shared by the Impero passes."""

from __future__ import annotations

import dataclasses
import functools
import numbers
import operator
from typing import Any

from immutabledict import immutabledict as idict

from gem import gem

from pyop3.compile.passes.impero.utils import is_static, static_extent


@dataclasses.dataclass(frozen=True)
class LinearIndex:
    """A flat index `sum_i k·i + constant`."""

    coefficients: idict
    constant: int

    def without(self, indices: tuple[gem.Index, ...]) -> LinearIndex:
        return LinearIndex(idict({i: k for i, k in self.coefficients.items() if i not in indices}), self.constant)

    @property
    def size(self) -> int:
        """The number of entries from 0 to the largest value."""
        return self.constant + 1 + sum(k * (static_extent(i) - 1) for i, k in self.coefficients.items())

    @property
    def bounds(self) -> tuple[int, int] | None:
        if not all(is_static(i.extent) for i in self.coefficients):
            return None
        return self.constant, self.size - 1

    def is_injective(self, indices: tuple[gem.Index, ...]) -> bool:
        """Whether the value determines the point of `indices`, whatever the other indices are.

        This is a sufficient condition: sorted by coefficient, the varying terms
        must form a mixed-radix number, so the value determines every varying
        index. A dynamic index (the item index) has no bound, so it must have
        the largest coefficient. An index of extent 1 takes one value, so it is
        ignored.
        """
        def varies(i: gem.Index) -> bool:
            return not is_static(i.extent) or static_extent(i) != 1

        if not set(filter(varies, indices)) <= self.coefficients.keys():
            return False
        span = 0
        terms = sorted(((i, k) for i, k in self.coefficients.items() if varies(i)), key=operator.itemgetter(1))
        for n, (index, k) in enumerate(terms):
            if k <= span:
                return False
            if not is_static(index.extent):
                return n == len(terms) - 1
            span += k * (static_extent(index) - 1)
        return True

    def to_gem(self) -> int | gem.Node:
        terms = [i if k == 1 else i * gem.as_gem_uint(k) for i, k in self.coefficients.items()]
        if not terms:
            return self.constant
        expr = functools.reduce(operator.add, terms)
        return expr + gem.as_gem_uint(self.constant) if self.constant else expr


# {{{ helpers

@functools.singledispatch
def affine(expr: Any) -> tuple[dict, int]:
    """Return `expr` as `sum k·term + constant`, where a term is not a sum, a constant, or a product with a constant."""
    return {expr: 1}, 0

@affine.register
def _(expr: numbers.Integral) -> tuple[dict, int]:
    return {}, int(expr)

@affine.register
def _(expr: gem.VariableIndex) -> tuple[dict, int]:
    return affine(expr.expression)

@affine.register
def _(expr: gem.Constant) -> tuple[dict, int]:
    return ({expr: 1}, 0) if expr.shape else ({}, int(expr.value))

@affine.register
def _(expr: gem.Sum) -> tuple[dict, int]:
    (a, c), (b, d) = map(affine, expr.children)
    terms = dict(a)
    for term, k in b.items():
        terms[term] = terms.get(term, 0) + k
    return {t: k for t, k in terms.items() if k}, c + d

@affine.register
def _(expr: gem.Product) -> tuple[dict, int]:
    (a, c), (b, d) = map(affine, expr.children)
    if not a:
        return {t: c * k for t, k in b.items() if c * k}, c * d
    if not b:
        return {t: d * k for t, k in a.items() if d * k}, c * d
    return {expr: 1}, 0


def flat_entry(multiindex: tuple) -> Any:
    """The only entry of `multiindex`, with a `gem.VariableIndex` unwrapped."""
    if len(multiindex) != 1:
        raise NotImplementedError("Only flat variables are supported")
    entry, = multiindex
    return entry.expression if isinstance(entry, gem.VariableIndex) else entry


def linear(access: gem.Indexed) -> LinearIndex | None:
    """The flat index of `access` as a linear index, or `None` if it is not linear in indices with positive coefficients."""
    terms, constant = affine(flat_entry(access.multiindex))

    # A negative coefficient or constant would make size and to_gem wrong, so it is treated as nonlinear
    if constant < 0 or not all(isinstance(term, gem.Index) and k > 0 for term, k in terms.items()):
        return None

    return LinearIndex(idict(terms), constant)


# }}}

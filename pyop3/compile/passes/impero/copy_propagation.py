"""Removal of temporaries that only copy other data, on the loop nests of `pyop3.compile.passes.impero.thread_transposition`.

This is a popular compiler pass known as "copy propagation" or maybe "advanced copy propagation" for array-based.

It removes situations like
A = B + 1
B = 2*A
C = B + 1 

->
C = (2 * (A + 1) + 1

This removes the intermediate B variable, reducing storage. 

I would love to have a separate compiler or an existing library be used for this compiler pass but 
there is no generic one that can be applied to GEM, and thread transposition cannot be applied properly
as some of the intermediate temporaries (that would be removed) can influence the TT phases and parameter choices
"""

from __future__ import annotations

import operator
from typing import Any

from gem import gem, replace

from pyop3.compile.passes.impero.linear import LinearIndex, affine, flat_entry, linear
from pyop3.compile.passes.impero.utils import is_static, is_temporary, num_points, static_extent


def propagate_copies(nests):
    """Remove temporaries that only copy other data, one at a time until none is left.

    pyop3 packs the data of an item by copies such as `t_1[2*i + j] =
    dat[map[i] + j]` (gather) or `t_2[2*i + j] = t_1[2*perm[i] + j]`
    (permutation), and each would otherwise be a phase with barriers.
    """
    while True:
        for name in dict.fromkeys(n.write.aggregate.name for n in nests if is_temporary(n.write.aggregate)):
            if (propagated := _propagate_copy(nests, name)) is not None:
                nests = propagated
                break
        else:
            return nests


def _propagate_copy(nests, name: str) -> list | None:
    """Replace the reads of temporary `name` by the sources of its copies, or return `None` if this is not legal.

    Every writer must be a copy with an injective linear write over its whole
    domain, with disjoint ranges. Every writer must precede every reader, with
    no write to a source in between. The readers must not read more entries
    than are written, so that no load is repeated (the coordinates that the
    geometry reads many times are kept).
    """
    writers = [k for k, n in enumerate(nests) if n.write.aggregate.name == name]
    readers = [k for k, n in enumerate(nests) if k not in writers and n.accesses_to(name)]
    if not readers or max(writers) > min(readers):
        return None

    copies = [nests[k] for k in writers]
    for copy in copies:
        w = linear(copy.write)
        if (
            not _is_copy(copy) or w is None or set(w.coefficients) != set(copy.domain)
            or not w.is_injective(copy.domain) or any(a.aggregate.name == name for a in copy.reads)
        ):
            return None
    bounds = sorted(linear(copy.write).bounds for copy in copies)
    if any(a[1] >= b[0] for a, b in zip(bounds, bounds[1:])):
        return None

    inputs = {a.aggregate.name for copy in copies for a in copy.reads}
    if any(nests[k].write.aggregate.name in inputs for k in range(min(writers), max(readers) + 1) if k not in writers):
        return None

    num_written = sum(num_points(copy.domain) for copy in copies)
    num_read = sum(len(nests[k].accesses_to(name)) * num_points(nests[k].loops) for k in readers)
    if num_read > num_written:
        return None
    sources = {a: _copy_source(copies, a) for k in readers for a in nests[k].accesses_to(name)}
    if None in sources.values():
        return None
    return [nest.rewritten(sources) if k in readers else nest for k, nest in enumerate(nests) if k not in writers]


def _copy_source(copies, access: gem.Indexed) -> gem.Node | None:
    """The source of `access`, from the only copy or the copy whose range contains it."""
    if len(copies) == 1:
        copy, = copies
    else:
        read = linear(access)
        if read is None or read.bounds is None:
            return None
        lo, hi = read.bounds
        matches = [c for c in copies if linear(c.write).bounds[0] <= lo and hi <= linear(c.write).bounds[1]]
        if len(matches) != 1:
            return None
        copy, = matches
    indices = _copy_indices(flat_entry(access.multiindex), linear(copy.write))
    return None if indices is None else replace(copy.statement.expression, indices)


def _is_copy(nest) -> bool:
    e = nest.statement.expression
    return (
        nest.statement.mode == "write" and not nest.reductions
        and isinstance(e, gem.Indexed) and isinstance(e.aggregate, gem.Variable)
    )


def _copy_indices(index: Any, write: LinearIndex) -> dict[gem.Index, gem.Node] | None:
    """The values of the indices of a copy that writes entry `index`, or `None` if they cannot be found.

    `write` is injective, so its coefficients are the radices of a
    mixed-radix number, and `index` is split into its digits: each term must
    have the coefficient of one index, and the constant is split from the most
    significant digit down. A digit other than the most significant must not
    carry, which is known for a constant or a static index plus a constant.
    An index of extent 1 is 0, and is not a digit.
    """
    unit = tuple(i for i in write.coefficients if static_extent(i) == 1)
    write = write.without(unit)
    indices = dict.fromkeys(unit, gem.as_gem_uint(0))
    terms, rest = affine(index)
    rest -= write.constant
    if rest < 0:
        return None
    if not write.coefficients:
        return indices if not terms and rest == 0 else None

    by_coefficient = {k: i for i, k in write.coefficients.items()}
    digit_terms = {}
    for term, k in terms.items():
        i = by_coefficient.get(k)
        if i is None or i in digit_terms:
            return None
        digit_terms[i] = term

    order = sorted(write.coefficients.items(), key=operator.itemgetter(1))
    digits = {}
    for i, k in reversed(order):
        digits[i], rest = divmod(rest, k)
    if rest:
        return None

    for i, _ in order:
        term, c = digit_terms.get(i), digits[i]
        if i != order[-1][0]:
            if term is None:
                fits = c < static_extent(i)
            else:
                fits = isinstance(term, gem.Index) and is_static(term.extent) and static_extent(term) + c <= static_extent(i)
            if not fits:
                return None
        if term is None:
            indices[i] = gem.as_gem_uint(c)
        else:
            indices[i] = term + gem.as_gem_uint(c) if c else term
    return indices

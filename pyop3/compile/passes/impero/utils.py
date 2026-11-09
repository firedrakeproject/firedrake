"""Helpers that the Impero passes share: kernel temporaries, static extents, and rewriting terminals."""

from __future__ import annotations

import functools
import numbers
import operator
from collections.abc import Mapping
from typing import Any

import numpy as np

from gem import gem, impero as imp, replace
from gem.node import pre_traversal

from pyop3 import utils
from pyop3.dtypes import ScalarType


TEMPORARY_DATA = "null"
"""The `data` of a `gem.Variable` that pyop3 declares as a kernel temporary
Ideally we have a nicer way of dealing with this in future 

All is_temporary checks happen in this file as a result 
"""

SHARED_DATA = "shared"
"""The `data` of a temporary in the shared memory of the block of an `gem.impero.ItemBatch`"""


# TODO: Reduce the number of helper functions.
# This has ballooned and feels really wrong. Many of these are just to maintain a single-source-of-truth
# i.e. is_item_loop or is_temporary; all assumptions belong to these functions ONLY  

def is_static(extent: Any) -> bool:
    return isinstance(extent, (numbers.Integral, gem.Literal))


def static_extent(index: gem.Index) -> int:
    if not is_static(index.extent):
        raise NotImplementedError(f"Temporaries over the dynamic index {index} are not supported")
    return int(index.extent if isinstance(index.extent, numbers.Integral) else index.extent.value)


def num_points(indices: tuple[gem.Index, ...]) -> int:
    """The number of points of the static `indices`."""
    return int(np.prod([static_extent(i) for i in indices], dtype=int))


def temporary(name: str, shape: tuple, dtype, *, shared: bool = False) -> gem.Variable:
    """A kernel temporary, in the shared memory of the block of an `gem.impero.ItemBatch` if `shared`"""
    return gem.Variable(name, shape, dtype=dtype, data=SHARED_DATA if shared else TEMPORARY_DATA)


def as_index(index: Any) -> int | gem.IndexBase:
    """A flat index (an `int`, an index or a GEM expression) as an entry of a multi-index. An index is returned unchanged."""
    if isinstance(index, numbers.Integral):
        return int(index)
    if isinstance(index, gem.Constant) and not index.shape:
        return int(index.value)
    return gem.VariableIndex(index)


def new_temporary(name: str, indices: tuple[gem.Index, ...], dtype) -> gem.Indexed:
    """The entry of a new row-major temporary, with one entry per point of `indices`, that stores and loads a value."""
    strides = utils.row_major_strides([static_extent(index) for index in indices])
    terms = [i if s == 1 else i * gem.as_gem_uint(s) for i, s in zip(indices, strides)]
    index = functools.reduce(operator.add, terms) if terms else 0
    variable = temporary(name, (num_points(indices),), dtype or ScalarType)
    return gem.Indexed(variable, (as_index(index),))


def is_temporary(variable: gem.Node) -> bool:
    """Whether `variable` is a kernel temporary, in shared memory or not"""
    return isinstance(variable, gem.Variable) and isinstance(variable.data, str) and variable.data in (TEMPORARY_DATA, SHARED_DATA)


def is_shared(variable: gem.Node) -> bool:
    """Whether `variable` is a kernel temporary in shared memory"""
    return is_temporary(variable) and variable.data == SHARED_DATA


def is_item_loop(node: imp.Node) -> bool:
    """Whether `node` is a loop over the items of a pyop3 loop, as `GemCompiler` marks them"""
    return isinstance(node, imp.For) and node.is_parallel


def name_generator(tree: imp.Node) -> utils.UniqueNameGenerator:
    """Names for new temporaries that no variable of `tree` already has"""
    return utils.UniqueNameGenerator(existing_names={e.name for e in gem_nodes(tree) if isinstance(e, gem.Variable)})


def gem_nodes(*nodes: imp.Node):
    """Every GEM node that the terminals below `nodes` refer to"""
    terminals = (n for n in pre_traversal(nodes) if isinstance(n, imp.Terminal))
    values = (getattr(t, slot) for t in terminals for slot in t.__slots__)
    return pre_traversal([v for v in values if isinstance(v, gem.Node)])


@functools.singledispatch
def assign_value(terminal: imp.Terminal, temporaries: Mapping) -> imp.Assignment | None:
    """Return the terminal that computes a value of `temporaries` as an assignment to its temporary

    The values that it reads become loads from their temporaries. Return `None` for any other terminal
    """
    return None


@assign_value.register
def _(terminal: imp.Evaluate, temporaries: Mapping) -> imp.Assignment | None:
    if (expr := terminal.expression) not in temporaries:
        return None
    # Without expr, so that it is not replaced by its own temporary
    others = {value: t for value, t in temporaries.items() if value != expr}
    return imp.Assignment(temporaries[expr], replace(expr, others), "write")


@assign_value.register
def _(terminal: imp.Initialise, temporaries: Mapping) -> imp.Assignment | None:
    if (expr := terminal.indexsum) not in temporaries:
        return None
    return imp.Assignment(temporaries[expr], gem.Zero(dtype=expr.dtype), "write")


@assign_value.register
def _(terminal: imp.Accumulate, temporaries: Mapping) -> imp.Assignment | None:
    if (expr := terminal.indexsum) not in temporaries:
        return None
    return imp.Assignment(temporaries[expr], replace(expr.children[0], temporaries), "inc")

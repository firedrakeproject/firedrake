from __future__ import annotations

import functools
import itertools
import numbers
from collections.abc import Hashable
from types import NoneType
from typing import Any

from immutabledict import immutabledict as idict

import pyop3.axis_tree
from pyop3 import utils
from pyop3.cache import memory_cache
from pyop3.collections import OrderedFrozenSet
from pyop3.labeled_tree import parent_path
from pyop3.node import Visitor, postorder

from .layout import compute_layouts  # noqa: F401
from .size import compute_axis_tree_component_size, compute_axis_tree_size  # noqa: F401


def get_block_shape(axis_tree: AbstractAxisTree) -> tuple[int, ...]:
    """Detect any common innermost integer shape in an axis tree."""
    if axis_tree.depth < 2:
        return ()

    axis_tree = axis_tree.materialize()

    block_shape = []
    while not axis_tree.is_empty:
        if not utils.is_single_valued(axis_tree.leaves):
            break
        leaf_axis = utils.single_valued(axis_tree.leaves)

        if not isinstance(leaf_axis.size, numbers.Integral):
            break
        block_shape.insert(0, leaf_axis.size)

        for leaf_path in axis_tree.leaf_paths:
            axis_tree = axis_tree.drop_node(parent_path(leaf_path))
    return tuple(block_shape)

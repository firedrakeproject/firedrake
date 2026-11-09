# Note that this file contains imports from all of pyop3. This module should
# therefore only be imported after all other modules.

import numbers
from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from petsc4py import PETSc

# {{{ imports we need in the namespace to resolve type hints

import immutabledict

import pyop3

# }}}


# TODO: When we drop Python 3.11, use the 'type' keyword here so Sphinx will
# document these properly

KwargsT: TypeAlias = Mapping[str, Any]
PetscSizeT: TypeAlias = tuple[int | None, int | None]

# {{{ tree types

LabelT: TypeAlias = Hashable
NodeLabelT: TypeAlias = LabelT
NodeComponentLabelT: TypeAlias = LabelT
PathT: TypeAlias = Mapping[NodeLabelT, NodeComponentLabelT]
"""Type alias for things can be parsed into a tree path."""
ConcretePathT: TypeAlias = immutabledict.immutabledict[NodeLabelT, NodeComponentLabelT]
"""Type alias for a tree path."""

# }}}


# {{{ axis tree types

AxisLabelT: TypeAlias = NodeLabelT
"""Type alias for axis labels."""
AxisComponentLabelT: TypeAlias = NodeComponentLabelT
"""Type alias for axis component labels."""
AxisComponentRegionLabelT: TypeAlias = LabelT
"""Type alias for axis component region labels."""

AxisComponentLike: TypeAlias = pyop3.AxisComponent | AxisComponentLabelT
"""Type alias for things that can be parsed into axis components."""
AxisComponentRegionSizeT: TypeAlias = numbers.Integral | pyop3.expr.LinearDatBufferExpression

# }}}


# {{{ index tree types

SliceComponentsT: TypeAlias = (
    Sequence[pyop3.SliceComponent]
    | pyop3.SliceComponent
    | Mapping[NodeComponentLabelT, Any]
    | NodeComponentLabelT
)

# }}}

# {{{ expr types

ExpressionT: TypeAlias = Any

# }}}

# {{{ array types

ArrayT: TypeAlias = np.ndarray | pyop3.arrayref.ArrayReference

# DeviceArrayT = np.ndarray | cp.array
DeviceArrayT: TypeAlias = np.ndarray

MatInsertMode: TypeAlias = Literal[PETSc.InsertMode.INSERT_VALUES, PETSc.InsertMode.ADD_VALUES]

# }}}

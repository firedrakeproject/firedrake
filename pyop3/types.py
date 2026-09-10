# Note that this file contains imports from all of pyop3. This module should
# therefore only be imported inside a 'typing.TYPE_CHECKING' block.
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
from immutabledict import immutabledict as idict
from petsc4py import PETSc

import pyop3

IntType = PETSc.IntType


KwargsT = Mapping[str, Any]
PetscSizeT = tuple[int|None, int|None]

# {{{ tree types

LabelT = Hashable
NodeLabelT = LabelT
NodeComponentLabelT = LabelT
PathT = Mapping[NodeLabelT, NodeComponentLabelT]
ConcretePathT = idict[NodeLabelT, NodeComponentLabelT]

# }}}


# {{{ axis tree types

AxisComponentRegionSizeT = IntType | LinearDatBufferExpression
AxisLabelT = NodeLabelT
IteratorIndexT = tuple[ConcretePathT, idict[AxisLabelT, int]]

# }}}

# {{{ array types

ArrayT = np.ndarray | pyop3.arrayref.ArrayReference

DeviceArrayT = np.ndarray | cp.array

MatInsertMode = Literal[PETSc.InsertMode.INSERT_VALUES, PETSc.InsertMode.ADD_VALUES]

# }}}

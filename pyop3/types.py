# Note that this file contains imports from all of pyop3. This module should
# therefore only be imported inside a 'typing.TYPE_CHECKING' block.
from collections.abc import Mapping
from typing import Any
from typing import Hashable, Mapping

import numpy as np
from immutabledict import immutabledict as idict

import pyop3




KwargsT = Mapping[str, Any]
PetscSizeT = tuple[int|None, int|None]


# {{{ axis tree types

AxisComponentRegionSizeT = IntType | LinearDatBufferExpression
AxisLabelT = NodeLabelT
IteratorIndexT = tuple[ConcretePathT, idict[AxisLabelT, int]]

# }}}

# {{{ array types

ArrayT = np.ndarray | pyop3.arrayref.ArrayReference

# }}}

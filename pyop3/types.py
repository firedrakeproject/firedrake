# NOTE: I still am not convinced that this is the best approach
from typing import Hashable, Mapping

from immutabledict import immutabledict as idict



# Axis tree types
AxisComponentRegionSizeT = IntType | LinearDatBufferExpression
AxisLabelT = NodeLabelT
IteratorIndexT = tuple[ConcretePathT, idict[AxisLabelT, int]]

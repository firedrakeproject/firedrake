# NOTE: I still am not convinced that this is the best approach
from typing import Hashable, Mapping

from immutabledict import immutabledict as idict


# Labelled tree types
LabelT = Hashable
NodeLabelT = Hashable
ComponentLabelT = Hashable
ComponentRegionLabelT = Hashable
ComponentT = ComponentLabelT  # | ComponentT
PathT = Mapping[NodeLabelT, ComponentLabelT]
ConcretePathT = idict[NodeLabelT, ComponentLabelT]

NodeMapT = Mapping[PathT, Node | None]
ConcreteNodeMapT = idict[ConcretePathT, Node | None]

# ParentT = tuple[PathT, ComponentT] | PathT | | None

# Axis tree types
AxisComponentRegionSizeT = IntType | LinearDatBufferExpression
AxisLabelT = NodeLabelT
IteratorIndexT = tuple[ConcretePathT, idict[AxisLabelT, int]]

# Import order is specially managed in this file, don't complain about it
# flake8: noqa: I001

# Before anything else, initialise pyop3
from pyop3.config import config_ as config
__all__ = ["config"]


# think the command line is a better way to do this.
def _init_likwid():
    import os

    if "LIKWID_MODE" in os.environ:
        # TODO: nice error message if import fails
        import atexit

        import pylikwid

        pylikwid.markerinit()
        atexit.register(pylikwid.markerclose)


_init_likwid()
del _init_likwid


from pyop3.axis_tree import (
    Axis,
    AxisComponent,
    AxisComponentRegion,
    AxisForest,
    AxisTarget,
    AxisTree,
    IndexedAxisTree,
)
__all__ += [
    "Axis",
    "AxisComponent",
    "AxisComponentRegion",
    "AxisForest",
    "AxisTarget",
    "AxisTree",
    "IndexedAxisTree",
]
from pyop3.buffer import (
    ArrayBuffer,
    DensePythonMatContext,
    NonNestedPetscMatBufferSpec,
    NullBuffer,
    PetscMatBuffer,
    PetscMatNestBufferSpec,
)
__all__ += [
    "ArrayBuffer",
    "DensePythonMatContext",
    "NonNestedPetscMatBufferSpec",
    "NullBuffer",
    "PetscMatBuffer",
    "PetscMatNestBufferSpec",
]
from pyop3.constants import (
    DECIDE,
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    WRITE,
)
__all__ += [
    "DECIDE",
    "INC",
    "MAX_RW",
    "MAX_WRITE",
    "MIN_RW",
    "MIN_WRITE",
    "READ",
    "RW",
    "WRITE",
]
from pyop3.device import CUDAGPU, HOST_DEVICE, offloading
__all__ += ["CUDAGPU", "HOST_DEVICE", "offloading"]
from pyop3.dtypes import IntType, ScalarType
__all__ += ["IntType", "ScalarType"]
from pyop3.expr import (
    AxisVar,
    LinearDatBufferExpression,
    NaN,
    OpaqueTerminal,
    as_linear_buffer_expression,
)
__all__ += [
    "AxisVar",
    "LinearDatBufferExpression",
    "NaN",
    "OpaqueTerminal",
    "as_linear_buffer_expression",
]
from pyop3.expr.tensor import (
    AggregateDat,
    AggregateMat,
    Dat,
    Mat,
    OutOfPlaceCallableTensorTransform,
    Scalar,
    Tensor,
)
__all__ += [
    "AggregateDat",
    "AggregateMat",
    "Dat",
    "Mat",
    "OutOfPlaceCallableTensorTransform",
    "Scalar",
    "Tensor",
]
from pyop3.expr.visitors import (
    collect_axis_vars,
    evaluate,
    replace_terminals,
)
__all__ += [
    "collect_axis_vars",
    "evaluate",
    "replace_terminals",
]
from pyop3.index_tree import (
    AffineSliceComponent,
    Index,
    IndexTree,
    LoopIndex,
    Map,
    ScalarIndex,
    ScalarMap,
    Slice,
    SliceComponent,
    Subset,
    SubsetSliceComponent,
    TabulatedMapComponent,
)
__all__ += [
    "AffineSliceComponent",
    "Index",
    "IndexTree",
    "LoopIndex",
    "Map",
    "ScalarIndex",
    "ScalarMap",
    "Slice",
    "SliceComponent",
    "Subset",
    "SubsetSliceComponent",
    "TabulatedMapComponent",
]
from pyop3.insn import (
    Assignment,
    AssignmentType,
    Function,
    Loop,
    exscan,
    loop_ as loop,
)
__all__ += [
    "Assignment",
    "AssignmentType",
    "Function",
    "Loop",
    "exscan",
    "loop",
]
from pyop3.sf import StarForest, local_sf, single_star_sf
__all__ += ["StarForest", "local_sf", "single_star_sf"]
from pyop3.utils import atom
__all__ += ["atom"]
from pyop3.visitors import replace
__all__ += ["replace"]

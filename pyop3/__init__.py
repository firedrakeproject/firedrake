from pyop3.config import config_ as config

def _fixup_pytools():
    # Many pyop3 objects inherit from pytools.RecordWithoutPickling.
    # RecordWithoutPickling sets __getattr__ for linting purposes but this breaks
    # tracebacks for @property methods so we remove it here.
    import pytools

    try:
        del pytools.RecordWithoutPickling.__getattr__
    except AttributeError:
        pass


_fixup_pytools()
del _fixup_pytools


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


import pyop3.dtypes
import pyop3.lower
import pyop3.insn.visitors as insn_visitors

from pyop3.constants import (   # noqa: F401
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
from pyop3.expr.tensor import (  # noqa: F401
    Tensor, Dat, Scalar, Mat, AggregateMat, AggregateDat,
    OutOfPlaceCallableTensorTransform
)
from pyop3.expr import as_linear_buffer_expression, AxisVar, LinearDatBufferExpression, OpaqueTerminal, NaN
from pyop3.axis_tree import (  # noqa: F401
    Axis,
    AxisForest,
    AxisTarget,
    AxisComponent,
    AxisComponentRegion,
    AxisTree,
    IndexedAxisTree,
)
from pyop3.expr.visitors import collect_axis_vars, evaluate, replace, replace_terminals  # noqa: F401
from pyop3.buffer import (  # noqa: F401
    ArrayBuffer, NullBuffer, NonNestedPetscMatBufferSpec, PetscMatNestBufferSpec, PetscMatBuffer
)
from pyop3.dtypes import IntType, ScalarType  # noqa: F401
from pyop3.index_tree import (  # noqa: F401
    AffineSliceComponent,
    Index,
    IndexTree,
    LoopIndex,
    Map,
    ScalarMap,
    Slice,
    SliceComponent,
    Subset,
    SubsetSliceComponent,
    TabulatedMapComponent,
    ScalarIndex,
    as_slice,
)
from pyop3.insn import (  # noqa: F401
    Function,
    Loop,
    Assignment,
    do_loop,
    loop_ as loop,
    exscan,
    AssignmentType,
)
from pyop3.device import ( # noqa: F401
    HOST_DEVICE,
    CUDAGPU,
    offloading
)
from pyop3.sf import StarForest, single_star_sf, local_sf
import pyop3.sf
from pyop3.index_tree.parse import as_index_forest
from pyop3.lower import LOOPY_TARGET, LOOPY_LANG_VERSION

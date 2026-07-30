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
import pyop3.insn.visitors as insn_visitors
import pyop3.lower
import pyop3.sf
from pyop3.axis_tree import (  # noqa: F401
    Axis,
    AxisComponent,
    AxisComponentRegion,
    AxisForest,
    AxisTarget,
    AxisTree,
    IndexedAxisTree,
)
from pyop3.buffer import (  # noqa: F401
    ArrayBuffer,
    DensePythonMatContext,
    NonNestedPetscMatBufferSpec,
    NullBuffer,
    PetscMatBuffer,
    PetscMatNestBufferSpec,
)
from pyop3.constants import (  # noqa: F401
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
from pyop3.device import CUDAGPU, HOST_DEVICE, offloading  # noqa: F401
from pyop3.dtypes import IntType, ScalarType  # noqa: F401
from pyop3.expr import (
    AxisVar,
    LinearDatBufferExpression,
    NaN,
    OpaqueTerminal,
    as_linear_buffer_expression,
)
from pyop3.expr.tensor import (  # noqa: F401
    AggregateDat,
    AggregateMat,
    Dat,
    Mat,
    OutOfPlaceCallableTensorTransform,
    Scalar,
    Tensor,
)
from pyop3.expr.visitors import (  # noqa: F401
    collect_axis_vars,
    evaluate,
    replace_terminals,
)
from pyop3.index_tree import (  # noqa: F401
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
from pyop3.index_tree.parse import as_index_forest
from pyop3.insn import (  # noqa: F401
    Assignment,
    AssignmentType,
    Function,
    Loop,
    do_loop,
    exscan,
)
from pyop3.insn import (
    loop_ as loop,
)
from pyop3.lower import LOOPY_LANG_VERSION, LOOPY_TARGET
from pyop3.sf import StarForest, local_sf, single_star_sf
from pyop3.utils import atom
from pyop3.visitors import replace

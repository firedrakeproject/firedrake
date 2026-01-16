import os as _os
from pyop3.config import config

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
    if "LIKWID_MODE" in _os.environ:
        # TODO: nice error message if import fails
        import atexit
        import pylikwid

        pylikwid.markerinit()
        atexit.register(pylikwid.markerclose)


_init_likwid()
del _init_likwid


# UNDO ME
import pyop3.extras


from pyop3.insn import Intent
import pyop3.dtypes, pyop3.tree
import pyop3.ir
import pyop3.insn.visitors as insn_visitors
from pyop3.expr.tensor import (  # noqa: F401
    Tensor, FancyIndexWriteException, Dat, Scalar, Mat, AggregateMat,
    RowDatPythonMatContext, ColumnDatPythonMatContext, OutOfPlaceCallableTensorTransform
)
from pyop3.expr import as_linear_buffer_expression, AxisVar, LinearDatBufferExpression
from pyop3.tree.axis_tree import (  # noqa: F401
    Axis,
    AxisForest,
    AxisTarget,
    AxisComponent,
    AxisComponentRegion,
    AxisTree,
    IndexedAxisTree,
)
from pyop3.expr.base import NAN  # noqa: F401
from pyop3.expr.visitors import collect_axis_vars, replace  # noqa: F401
from pyop3.buffer import (  # noqa: F401
    ArrayBuffer, NullBuffer, NonNestedPetscMatBufferSpec, PetscMatNestBufferSpec, LGMap
)
from pyop3.dtypes import IntType, ScalarType  # noqa: F401
from pyop3.expr.visitors import evaluate  # noqa: F401
from pyop3.tree.index_tree import (  # noqa: F401
    AffineSliceComponent,
    Index,
    IndexTree,
    LoopIndex,
    Map,
    Slice,
    SliceComponent,
    Subset,
    SubsetSliceComponent,
    TabulatedMapComponent,
    ScalarIndex,
    as_slice,
)
from pyop3.insn import (  # noqa: F401
    INC,
    MAX_RW,
    MAX_WRITE,
    MIN_RW,
    MIN_WRITE,
    READ,
    RW,
    WRITE,
    DummyKernelArgument,
    Function,
    Loop,
    OpaqueKernelArgument,
    ArrayAssignment,
    do_loop,
    loop_ as loop,
    exscan,
    AssignmentType,
)
from pyop3.sf import StarForest, single_star_sf, local_sf
import pyop3.sf
from pyop3.tree.index_tree.parse import as_index_forest
from pyop3.tree import accumulate_path
from pyop3.ir import LOOPY_TARGET, LOOPY_LANG_VERSION

del _os

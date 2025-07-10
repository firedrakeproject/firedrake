import os as _os
from pyop3.config import config as _config
from pyop3.itree.parse import as_index_forest

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


import pyop3.dtypes
import pyop3.ir
import pyop3.insn_visitors
from pyop3.tensor import (  # noqa: F401
    Tensor, FancyIndexWriteException, Dat, Scalar, Mat,
    RowDatPythonMatContext, ColumnDatPythonMatContext,
)
from pyop3.tensor.dat import as_linear_buffer_expression
from pyop3.axtree import (  # noqa: F401
    Axis,
    AxisComponent,
    AxisTree,
    AxisVar,
    IndexedAxisTree,
)
from pyop3.axtree.tree import NAN  # noqa: F401
from pyop3.buffer import (  # noqa: F401
    ArrayBuffer, NullBuffer, NonNestedPetscMatBufferSpec, PetscMatNestBufferSpec,
)
from pyop3.dtypes import IntType, ScalarType  # noqa: F401
from pyop3.itree import (  # noqa: F401
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
    as_index_forest,
)
from pyop3.lang import (  # noqa: F401
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
    _loop as loop,
)
from pyop3.sf import StarForest, single_star_sf, local_sf
import pyop3.sf
from . import utils

del _os
del _config

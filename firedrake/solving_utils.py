from pyop2 import op2
from petsc import PETSc


def update_parameters(obj, petsc_obj):
    """Update parameters on a petsc object

    :arg obj: An object with a parameters dict (mapping to petsc options).
    :arg petsc_obj: The PETSc object to set parameters on."""
    # Skip if parameters haven't changed
    if hasattr(obj, '_set_parameters') and obj.parameters == obj._set_parameters:
        return
    opts = PETSc.Options(obj._opt_prefix)
    for k, v in obj.parameters.iteritems():
        if type(v) is bool:
            if v:
                opts[k] = None
        else:
            opts[k] = v
    petsc_obj.setFromOptions()
    obj._set_parameters = obj.parameters.copy()


def set_fieldsplits(mat, pc, names=None):
    """Set up fieldsplit splits

    :arg mat: a :class:`~.Matrix` (the operator)
    :arg pc: a PETSc PC to set splits on
    :kwarg names: (optional) list of names for each split.
           If not provided, splits are numbered from 0.

    Returns a list of (name, IS) pairs (for later use with nullspace),
    or None if no fieldsplit was set up."""

    # No splits if not mixed
    if mat.sparsity.shape == (1, 1):
        return None

    rows, cols = mat.sparsity.shape
    ises = []
    nlocal_rows = 0
    for i in range(rows):
        if i < cols:
            nlocal_rows += mat[i, i].sparsity.nrows * mat[i, i].dims[0]
    offset = 0
    if op2.MPI.comm.rank == 0:
        op2.MPI.comm.exscan(nlocal_rows)
    else:
        offset = op2.MPI.comm.exscan(nlocal_rows)

    for i in range(rows):
        if i < cols:
            nrows = mat[i, i].sparsity.nrows * mat[i, i].dims[0]
            name = names[i] if names is not None else str(i)
            ises.append((name, PETSc.IS().createStride(nrows, first=offset, step=1)))
            offset += nrows

    pc.setFieldSplitIS(*ises)
    return ises

import pytest
import time
from firedrake import *
import ufl
from pyop2 import op2
from tsfc.finatinterface import create_element
from firedrake.preconditioners.fdm import diff_blocks, petsc_sparse, block_mat, get_base_elements, is_restricted, restricted_dofs, unrestrict_element
from firedrake.preconditioners.pmg import evaluate_dual
from firedrake.preconditioners.hypre_ams import chop
from firedrake.petsc import PETSc

def insert_kernel(name, triu=False, addv=None):
    """Insert one sparse matrix into another sparse matrix.
       Done in C for efficiency, since it loops over rows."""
    if triu:
        select_cols = "icol -= (icol < irow) * (1 + icol);"
    else:
        select_cols = ""
    if addv is None:
        addv = PETSc.InsertMode.Insert

    code = f"""
#include <petsc.h>

PetscErrorCode {name}(Mat A,
                      Mat B,
                      PetscInt *rindices,
                      PetscInt *cindices)
{{
    PetscInt ncols, irow, icol;
    PetscInt *cols, *indices;
    PetscScalar *vals;
    InsertMode addv = {addv};
    PetscInt m, n;
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    MatGetSize(B, &m, NULL);

    n = 0;
    for (PetscInt i = 0; i < m; i++) {{
        ierr = MatGetRow(B, i, &ncols, NULL, NULL);CHKERRQ(ierr);
        n = ncols > n ? ncols : n;
        ierr = MatRestoreRow(B, i, &ncols, NULL, NULL);CHKERRQ(ierr);
    }}
    PetscMalloc1(n, &indices);
    for (PetscInt i = 0; i < m; i++) {{
        ierr = MatGetRow(B, i, &ncols, &cols, &vals);CHKERRQ(ierr);
        irow = rindices[i];
        for (PetscInt j = 0; j < ncols; j++) {{
            icol = cindices[cols[j]];
            {select_cols}
            indices[j] = icol;
        }}
        ierr = MatSetValues(A, 1, &irow, ncols, indices, vals, addv);CHKERRQ(ierr);
        ierr = MatRestoreRow(B, i, &ncols, &cols, &vals);CHKERRQ(ierr);
    }}
    PetscFree(indices);
    PetscFunctionReturn(0);
}}
"""
    return code


def local_tensor(Vc, Vf, cbcs=[], fbcs=[], comm=None):
    """
    Tabulate exterior derivative: Vc -> Vf as an explicit sparse matrix.
    Works for any tensor-product basis. These are the same matrices one needs for HypreAMS and friends.
    """
    if comm is None:
        comm = Vf.comm
    ec = Vc.finat_element
    ef = Vf.finat_element
    if ef.formdegree - ec.formdegree != 1:
        raise ValueError("Expecting Vf = d(Vc)")

    elements = sorted(get_base_elements(ec), key=lambda e: e.formdegree)
    try:
        c0, = elements
        c1 = None
    except ValueError:
        c0, c1 = elements[::len(elements)-1]
        if c1.formdegree != 1:
            c1 = None

    elements = sorted(get_base_elements(ef), key=lambda e: e.formdegree)
    try:
        f1, = elements
        f0 = None
    except ValueError:
        f0, f1 = elements[::len(elements)-1]
        if f0.formdegree != 0:
            f0 = None

    tdim = Vc.mesh().topological_dimension()
    zero = PETSc.Mat()
    A00 = petsc_sparse(evaluate_dual(c0, f0), comm=PETSc.COMM_SELF) if f0 else zero
    A11 = petsc_sparse(evaluate_dual(c1, f1), comm=PETSc.COMM_SELF) if c1 else zero
    A10 = petsc_sparse(evaluate_dual(c0, f1, alpha=(1,)), comm=PETSc.COMM_SELF)
    Dhat = block_mat(diff_blocks(tdim, ec.formdegree, A00, A11, A10), destroy_blocks=True)
    #A00.destroy()
    #A10.destroy()
    #A11.destroy()

    if any(is_restricted(ec)) or any(is_restricted(ef)):
        scalar_element = lambda e: e._sub_element if isinstance(e, (ufl.TensorElement, ufl.VectorElement)) else e
        fdofs = restricted_dofs(ef, create_element(unrestrict_element(scalar_element(Vf.ufl_element()))))
        cdofs = restricted_dofs(ec, create_element(unrestrict_element(scalar_element(Vc.ufl_element()))))
        temp = Dhat
        fises = PETSc.IS().createGeneral(fdofs, comm=temp.getComm())
        cises = PETSc.IS().createGeneral(cdofs, comm=temp.getComm())
        Dhat = temp.createSubMatrix(fises, cises)
        temp.destroy()
        fises.destroy()
        cises.destroy()

    if Vf.value_size > 1:
        temp = Dhat
        eye = petsc_sparse(numpy.eye(Vf.value_size, dtype=PETSc.RealType), comm=temp.getComm())
        Dhat = temp.kron(eye)
        temp.destroy()
        eye.destroy()

    return Dhat


@pytest.mark.parallel()
def test_assemble_div():
    N = 2
    degree = 1
    mesh = UnitSquareMesh(N, N, quadrilateral=True)
    mesh = ExtrudedMesh(mesh, N)

    cell = mesh.ufl_cell()
    variant = "fdm"
    e0 = FiniteElement("NCF", cell, degree=degree, variant=variant)
    e1 = FiniteElement("DQ", cell, degree=degree-1, variant=variant)

    V0 = FunctionSpace(mesh, e0)
    V1 = FunctionSpace(mesh, e1)

    Dlocal = local_tensor(V0, V1)
    #Dlocal.view()


    tstart = time.time()
    Dgold = assemble(inner(TestFunction(V1), div(TrialFunction(V0))) * dx).petscmat
    tdense = time.time() - tstart
    print("Time dense", tdense)
    Dgold = chop(Dgold, tol=1E-10)


    Dglobal = Dgold.copy()
    Dglobal.zeroEntries()
    #Dgold.view()

    addv = PETSc.InsertMode.INSERT
    name = "insertKernel"
    kernel = op2.Kernel(insert_kernel(name=name, addv=addv), name=name)

    Dglobal_arg = op2.PassthroughArg(op2.PetscMatType(), Dglobal.handle)
    Dlocal_arg = op2.PassthroughArg(op2.PetscMatType(), Dlocal.handle)

    rindices = op2.Dat(V1.dof_dset, V1.dof_dset.lgmap.indices)
    cindices = op2.Dat(V0.dof_dset, V0.dof_dset.lgmap.indices)

    print(mesh.comm.rank, kernel.cache_key)

    tstart = time.time()
    #op2.par_loop(kernel,
    parloop = op2.LegacyParloop(kernel,
                 mesh.cell_set,
                 Dglobal_arg,
                 Dlocal_arg,
                 rindices(op2.READ, V1.cell_node_map()),
                 cindices(op2.READ, V0.cell_node_map()))
    print(parloop.global_kernel.cache_key)
    parloop()

    tsparse = time.time() - tstart
    print("Time sparse", tsparse)

    Dglobal.assemble()
    B = chop(Dgold - Dglobal, tol=1E-10)

    Z = B.copy()
    Z.zeroEntries()
    assert B.equal(Z)

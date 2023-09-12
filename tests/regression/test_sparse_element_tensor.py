import pytest
import time
from firedrake import *
import ufl
import numpy
from pyop2 import op2
from tsfc.finatinterface import create_element
from firedrake.preconditioners.fdm import mass_blocks, diff_blocks, petsc_sparse, block_mat, get_base_elements, is_restricted, restricted_dofs, unrestrict_element
from firedrake.preconditioners.pmg import evaluate_dual
from firedrake.preconditioners.hypre_ams import chop
from firedrake.petsc import PETSc
from firedrake.formmanipulation import ExtractSubBlock
from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.expand_indices import expand_indices
from pyop2.sparsity import get_preallocation


def insert_mat(A, B, rindices, cindices, triu=False, addv=None):
    if triu:
        select_cols = "icol -= (icol < irow) * (1 + icol);"
    else:
        select_cols = ""
    if addv is None:
        addv = PETSc.InsertMode.INSERT

    return f"""
    InsertMode addv = {addv};
    PetscInt ncols, irow, icol;
    PetscInt *cols, *indices;
    PetscScalar *vals;
    PetscInt m, n;

    MatGetSize({B}, &m, NULL);
    n = 0;
    for (PetscInt i = 0; i < m; i++) {{
        ierr = MatGetRow({B}, i, &ncols, NULL, NULL);CHKERRQ(ierr);
        n = ncols > n ? ncols : n;
        ierr = MatRestoreRow({B}, i, &ncols, NULL, NULL);CHKERRQ(ierr);
    }}
    PetscMalloc1(n, &indices);
    for (PetscInt i = 0; i < m; i++) {{
        ierr = MatGetRow({B}, i, &ncols, &cols, &vals);CHKERRQ(ierr);
        irow = {rindices}[i];
        for (PetscInt j = 0; j < ncols; j++) {{
            icol = {cindices}[cols[j]];
            {select_cols}
            indices[j] = icol;
        }}
        ierr = MatSetValues({A}, 1, &irow, ncols, indices, vals, addv);CHKERRQ(ierr);
        ierr = MatRestoreRow({B}, i, &ncols, &cols, &vals);CHKERRQ(ierr);
    }}
    PetscFree(indices);
    """


def constant_kernel(Vrow, Vcol, name, triu=False, addv=None):
    indices = ("rindices",) if Vrow == Vcol else ("rindices", "cindices")
    declare_indices = ", ".join(["PetscInt *%s" % s for s in indices])
    insert_code = insert_mat("A", "B", indices[0], indices[-1], triu=triu, addv=addv)
    return f"""
#include <petsc.h>

PetscErrorCode {name}(Mat A, Mat B,
                      {declare_indices})
{{
    PetscErrorCode ierr;
    PetscFunctionBeginUser;
    // A[rindices, cindices] += B
    {insert_code}
    PetscFunctionReturn(0);
}}
"""


def stiffness_kernel(Vrow, Vcol, name, triu=False, addv=None):
    indices = ("rindices",) if Vrow == Vcol else ("rindices", "cindices")
    declare_indices = ", ".join(["PetscInt *%s" % s for s in indices])
    insert_code = insert_mat("A", "B", indices[0], indices[-1], triu=triu, addv=addv)
    return f"""
#include <petsc.h>

PetscErrorCode {name}(Mat A, Mat B, Mat L, Mat D, Mat R,
                      PetscScalar *values,
                      {declare_indices})
{{
    PetscErrorCode ierr;
    PetscFunctionBeginUser;

    // D = diag(values)
    PetscInt nn;
    Vec vec = NULL;
    MatGetSize(D, &nn, NULL);
    ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, 1, nn, values, &vec);CHKERRQ(ierr);
    ierr = MatDiagonalSet(D, vec, INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecDestroy(&vec);CHKERRQ(ierr);

    // B = L * D * R
    ierr = MatMatMatMult(L, D, R, MAT_REUSE_MATRIX, PETSC_DEFAULT, &B);CHKERRQ(ierr);

    // A[rindices, cindices] += B
    {insert_code}
    PetscFunctionReturn(0);
}}
"""


def local_exterior_derivative(Vc, Vf, cbcs=[], fbcs=[], comm=None):
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
    A00.destroy()
    A11.destroy()
    if Dhat != A10:
        A10.destroy()

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


def assemble_reference_tensor(V, coefficients, Dtensor, transpose=False, sort_interior=False, cache=None):
    """
    Return the reference tensor used in the diagonal factorisation of the
    sparse cell matrices.  See Section 3.2 of Brubeck2022b.

    :arg V: a :class:`.FunctionSpace`

    :returns: a :class:`PETSc.Mat` interpolating V^k * d(V^k) onto
              broken(V^k) * broken(V^{k+1}) on the reference element.
    """
    value_size = V.value_size
    fe = V.finat_element
    tdim = fe.cell.get_spatial_dimension()
    formdegree = fe.formdegree
    degree = fe.degree
    if type(degree) != int:
        degree, = set(degree)
    if formdegree == tdim:
        degree = degree + 1
    is_interior, is_facet = is_restricted(fe)
    key = (value_size, tdim, degree, formdegree, is_interior, is_facet, transpose, sort_interior)
    if cache is None:
        cache = {}
    try:
        return cache[key]
    except KeyError:
        pass

    if transpose:
        result = assemble_reference_tensor(V, coefficients, Dtensor, transpose=False, sort_interior=sort_interior, cache=cache)
        result = PETSc.Mat().createTranspose(result).convert(result.getType())
        return cache.setdefault(key, result)

    if sort_interior:
        assert is_interior and not is_facet and not transpose
        # Sort DOFs to make A00 block diagonal with blocks of increasing dimension along the diagonal
        result = assemble_reference_tensor(V, coefficients, Dtensor, transpose=transpose, sort_interior=False, cache=cache)
        if formdegree != 0:
            # Compute the stiffness matrix on the interior of a cell
            A00 = Dtensor.PtAP(result)
            indptr, indices, _ = A00.getValuesCSR()
            degree = numpy.diff(indptr)
            # Sort by blocks
            uniq, u_index = numpy.unique(indices, return_index=True)
            perm = uniq[u_index.argsort(kind='stable')]
            # Sort by degree
            degree = degree[perm]
            perm = perm[degree.argsort(kind='stable')]
            A00.destroy()

            isperm = PETSc.IS().createGeneral(perm, comm=result.getComm())
            result = get_submat(result, iscol=isperm, permute=True)
            isperm.destroy()
        return cache.setdefault(key, result)

    short_key = key[:-3] + (False,) * 3
    try:
        result = cache[short_key]
    except KeyError:
        # Get CG(k) and DG(k-1) 1D elements from V
        elements = sorted(get_base_elements(fe), key=lambda e: e.formdegree)
        e0 = elements[0] if elements[0].formdegree == 0 else None
        e1 = elements[-1] if elements[-1].formdegree == 1 else None
        if e0 and is_interior:
            e0 = FIAT.RestrictedElement(e0, restriction_domain="interior")

        # Get broken(CG(k)) and DG(k-1) 1D elements from the coefficient spaces
        Q0 = coefficients.subfunctions[0].function_space().finat_element.element
        elements = sorted(get_base_elements(Q0), key=lambda e: e.formdegree)
        q0 = elements[0] if elements[0].formdegree == 0 else None
        q1 = elements[-1]
        if q1.formdegree != 1:
            Q1 = coefficients.subfunctions[1].function_space().finat_element.element
            q1 = sorted(get_base_elements(Q1), key=lambda e: e.formdegree)[-1]

        # Interpolate V * d(V) -> space(beta) * space(alpha)
        comm = PETSc.COMM_SELF
        zero = PETSc.Mat()
        A00 = petsc_sparse(evaluate_dual(e0, q0), comm=comm) if e0 and q0 else zero
        A11 = petsc_sparse(evaluate_dual(e1, q1), comm=comm) if e1 else zero
        A10 = petsc_sparse(evaluate_dual(e0, q1, alpha=(1,)), comm=comm) if e0 else zero
        B_blocks = mass_blocks(tdim, formdegree, A00, A11)
        A_blocks = diff_blocks(tdim, formdegree, A00, A11, A10)
        result = block_mat(B_blocks + A_blocks, destroy_blocks=True)
        A00.destroy()
        A10.destroy()
        A11.destroy()
        if value_size != 1:
            eye = petsc_sparse(numpy.eye(value_size), comm=result.getComm())
            temp = result
            result = temp.kron(eye)
            temp.destroy()
            eye.destroy()

    if is_facet:
        cache[short_key] = result
        result = get_submat(result, iscol=fises)
    return cache.setdefault(key, result)


def assemble_coefficients(J, fcp, block_diagonal=True):
    """
    Obtain coefficients for the auxiliary operator as the diagonal of a
    weighted mass matrix in broken(V^k) * broken(V^{k+1}).
    See Section 3.2 of Brubeck2022b.

    :arg J: the Jacobian bilinear :class:`ufl.Form`,
    :arg fcp: form compiler parameters to assemble the diagonal of the mass matrices.
    :arg block_diagonal: are we assembling the block diagonal of the mass matrices?

    :returns: a 2-tuple of a `dict` with the zero-th order and second
              order coefficients keyed on ``"beta"`` and ``"alpha"``,
              and a list of assembly callables.
    """
    coefficients = {}
    assembly_callables = []
    # Basic idea: take the original bilinear form and
    # replace the exterior derivatives with arguments in broken(V^{k+1}).
    # Then, replace the original arguments with arguments in broken(V^k).
    # Where the broken spaces have L2-orthogonal FDM basis functions.
    index = len(J.arguments()[-1].function_space())-1
    if index:
        splitter = ExtractSubBlock()
        J = splitter.split(J, argument_indices=(index, index))

    args_J = J.arguments()
    e = args_J[0].ufl_element()
    mesh = args_J[0].function_space().mesh()
    tdim = mesh.topological_dimension()
    if isinstance(e, (ufl.VectorElement, ufl.TensorElement)):
        e = e._sub_element
    e = unrestrict_element(e)
    sobolev = e.sobolev_space()

    # Replacement rule for the exterior derivative = grad(arg) * eps
    map_grad = None
    if sobolev == ufl.H1:
        map_grad = lambda p: p
    elif sobolev in [ufl.HCurl, ufl.HDiv]:
        u = ufl.Coefficient(ufl.FunctionSpace(mesh, e))
        du = ufl.variable(ufl.grad(u))
        dku = ufl.div(u) if sobolev == ufl.HDiv else ufl.curl(u)
        eps = expand_derivatives(ufl.diff(ufl.replace(expand_derivatives(dku), {ufl.grad(u): du}), du))
        if sobolev == ufl.HDiv:
            map_grad = lambda p: ufl.outer(p, eps/tdim)
        elif len(eps.ufl_shape) == 3:
            map_grad = lambda p: ufl.dot(p, eps/2)
        else:
            map_grad = lambda p: p*(eps/2)

    # Construct Z = broken(V^k) * broken(V^{k+1})
    V = args_J[0].function_space()
    fe = V.finat_element
    formdegree = fe.formdegree
    degree = fe.degree
    if type(degree) != int:
        degree, = set(degree)
    qdeg = degree
    if formdegree == tdim:
        qfam = "DG" if tdim == 1 else "DQ"
        qdeg = 0
    elif formdegree == 0:
        qfam = "DG" if tdim == 1 else "RTCE" if tdim == 2 else "NCE"
    elif formdegree == 1 and tdim == 3:
        qfam = "NCF"
    else:
        qfam = "DQ L2"
        qdeg = degree - 1

    qvariant = "fdm_quadrature"
    elements = [e.reconstruct(variant=qvariant),
                ufl.FiniteElement(qfam, cell=mesh.ufl_cell(), degree=qdeg, variant=qvariant)]
    elements = list(map(ufl.BrokenElement, elements))
    if V.shape:
        elements = [ufl.TensorElement(ele, shape=V.shape) for ele in elements]
    Z = FunctionSpace(mesh, ufl.MixedElement(elements))

    # Transform the exterior derivative and the original arguments of J to arguments in Z
    args = (TestFunctions(Z), TrialFunctions(Z))
    repargs = {t: v[0] for t, v in zip(args_J, args)}
    repgrad = {ufl.grad(t): map_grad(v[1]) for t, v in zip(args_J, args)} if map_grad else {}
    Jcell = expand_indices(expand_derivatives(ufl.Form(J.integrals_by_type("cell"))))
    mixed_form = ufl.replace(ufl.replace(Jcell, repgrad), repargs)

    # Return coefficients and assembly callables
    from firedrake.assemble import OneFormAssembler
    coefficients = Function(Z)
    assembly_callables.append(OneFormAssembler(mixed_form, tensor=coefficients, diagonal=True,
                                               form_compiler_parameters=fcp).assemble)
    return coefficients, assembly_callables


def preallocate(Vrow, Vcol, Alocal):
    mesh = Vrow.mesh()
    comm = Vrow.comm
    sizes = tuple(V.dof_dset.layout_vec.getSizes() for V in (Vrow, Vcol))
    preallocator = PETSc.Mat().create(comm=comm)
    preallocator.setType(PETSc.Mat.Type.PREALLOCATOR)
    preallocator.setSizes(sizes)
    preallocator.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, False)
    preallocator.setUp()

    indices = lambda V: op2.Dat(V.dof_dset, V.dof_dset.lgmap.indices)
    spaces = (Vrow,) if Vrow == Vcol else (Vrow, Vcol)

    name = "preallocatorKernel"
    kernel = op2.Kernel(constant_kernel(Vrow, Vcol, name), name=name)
    op2.par_loop(kernel, mesh.cell_set,
                 *(op2.PassthroughArg(op2.PetscMatType(), mat.handle) for mat in (preallocator, Alocal)),
                 *(indices(V)(op2.READ, V.cell_node_map()) for V in spaces))
    preallocator.assemble()

    nnz = get_preallocation(preallocator, sizes[0][0])

    A = PETSc.Mat().createAIJ(sizes, preallocator.getBlockSizes(), nnz=nnz, comm=comm)
    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
    preallocator.destroy()
    return A


@pytest.mark.parallel()
def test_assemble_div():
    N = 3
    degree = 7

    dim = 3
    if dim == 1:
        mesh = UnitIntervalMesh(N)
        fam0 = "Lagrange"
        fam1 = "DG L2"

    elif dim >= 2:
        mesh = UnitSquareMesh(N, N, quadrilateral=True)
        fam0 = "RTCF"
        fam1 = "DQ"
        if dim == 3:
            fam0 = "NCF"
            mesh = ExtrudedMesh(mesh, N)

    cell = mesh.ufl_cell()
    variant = "fdm"
    e0 = FiniteElement(fam0, cell, degree=degree, variant=variant)
    e1 = FiniteElement(fam1, cell, degree=degree-1, variant=variant)

    V0 = FunctionSpace(mesh, e0)
    V1 = FunctionSpace(mesh, e1)

    tstart = time.time()
    if dim == 1:
        Dgold = chop(Interpolator(grad(TestFunction(V0))[0], V1).callable().handle)
    else:
        detJ = JacobianDeterminant(mesh)
        s = detJ/abs(detJ)
        Dgold = assemble(inner(TestFunction(V1), s*div(TrialFunction(V0))) * dx).petscmat

    tdense = time.time() - tstart
    print("Time dense", tdense)
    Dgold = chop(Dgold, tol=1E-10)

    #Dgold.view()

    #Dlocal.view()

    tstart = time.time()
    Dlocal = local_exterior_derivative(V0, V1)
    Dglobal = preallocate(V1, V0, Dlocal)
    Dglobal.zeroEntries()

    mats = [Dglobal, Dlocal]
    rindices = op2.Dat(V1.dof_dset, V1.dof_dset.lgmap.indices)
    cindices = op2.Dat(V0.dof_dset, V0.dof_dset.lgmap.indices)

    name = "insertKernel"
    kernel = op2.Kernel(constant_kernel(V1, V0, name=name), name=name)
    op2.par_loop(kernel,
                 mesh.cell_set,
                 *(op2.PassthroughArg(op2.PetscMatType(), mat.handle) for mat in mats),
                 rindices(op2.READ, V1.cell_node_map()),
                 cindices(op2.READ, V0.cell_node_map()))

    Dglobal.assemble()
    tsparse = time.time() - tstart
    print("Time sparse", tsparse)

    #Dglobal.view()
    B = chop(Dgold - Dglobal, tol=1E-10)

    Z = B.copy()
    Z.zeroEntries()
    assert B.equal(Z)


@pytest.mark.parallel()
def test_stiffness():
    tol = 1E-8
    N = 3
    degree = 7

    dim = 3
    if dim == 1:
        mesh = UnitIntervalMesh(N)
        family = "Lagrange"
    elif dim >= 2:
        mesh = UnitSquareMesh(N, N, quadrilateral=True)
        family = "RTCF"
        if dim == 3:
            family = "NCF"
            mesh = ExtrudedMesh(mesh, N)

    #family = "Lagrange"
    cell = mesh.ufl_cell()
    variant = "fdm"
    e = FiniteElement(family, cell, degree=degree, variant=variant)
    V = FunctionSpace(mesh, e)

    formdegree = V.finat_element.formdegree
    d = [grad, curl, div][formdegree]
    beta = Constant(1)
    alpha = Constant(2)

    u = TestFunction(V)
    v = TrialFunction(V)
    a = (inner(u, beta*v) + inner(d(u), alpha*d(v))) * dx

    tstart = time.time()
    Agold = chop(assemble(a).petscmat)
    tdense = time.time() - tstart
    print("Time dense", tdense)
    #Agold.view()


    fcp = None
    coefficients, assembly_callables = assemble_coefficients(a, fcp)
    Z = coefficients.function_space()
    for c in assembly_callables:
        c()

    n = sum(Zsub.finat_element.space_dimension() for Zsub in Z)
    data = numpy.ones((n,), dtype=PETSc.RealType)
    ai = numpy.arange(n+1, dtype=PETSc.IntType)
    aj = ai[:-1]
    Dtensor = PETSc.Mat().createAIJ((n, n), csr=(ai, aj, data), comm=PETSc.COMM_SELF)

    cache = {}
    Rtensor = assemble_reference_tensor(V, coefficients, Dtensor, cache=cache)
    Ltensor = assemble_reference_tensor(V, coefficients, Dtensor, transpose=True, cache=cache)
    Alocal = Ltensor.matMatMult(Dtensor, Rtensor)
    rindices = op2.Dat(V.dof_dset, V.dof_dset.lgmap.indices)

    tstart = time.time()

    Aglobal = preallocate(V, V, Alocal)
    Aglobal.zeroEntries()
    mats = [Aglobal, Alocal, Ltensor, Dtensor, Rtensor]
    name = "stiffnessKernel"
    addv = PETSc.InsertMode.ADD
    kernel = op2.Kernel(stiffness_kernel(V, V, name, addv=addv), name=name)
    op2.par_loop(kernel, mesh.cell_set,
                 *(op2.PassthroughArg(op2.PetscMatType(), mat.handle) for mat in mats),
                 coefficients.dat(op2.READ, Z.cell_node_map()),
                 rindices(op2.READ, V.cell_node_map()))
    Aglobal.assemble()

    tsparse = time.time() - tstart
    print("Time sparse", tsparse)

    #Aglobal.view()
    B = Agold - Aglobal

    #B.view()
    assert B.norm(norm_type=PETSc.NormType.FROBENIUS)/numpy.prod(B.getSize()) < tol


if __name__ == "__main__":
    test_stiffness()

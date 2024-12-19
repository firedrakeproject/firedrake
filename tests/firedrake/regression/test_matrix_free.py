from firedrake import *
from firedrake.petsc import PETSc
from firedrake.utils import ScalarType
from firedrake.solving_utils import DEFAULT_KSP_PARAMETERS
import pytest
import numpy as np
from mpi4py import MPI


@pytest.fixture
def mesh():
    return UnitSquareMesh(4, 4)


@pytest.fixture(params=["scalar", "vector"])
def V(mesh, request):
    if request.param == "scalar":
        return FunctionSpace(mesh, "CG", 1)
    elif request.param == "vector":
        return VectorFunctionSpace(mesh, "CG", 1)


@pytest.fixture
def L(V):
    x = SpatialCoordinate(V.mesh())
    v = TestFunction(V)
    if V.shape == ():
        return inner(sin(x[0]*2*pi)*sin(x[1]*2*pi), v)*dx
    elif V.shape == (2, ):
        return inner(as_vector([sin(x[0]*2*pi)*sin(x[1]*2*pi),
                                cos(x[0]*2*pi)*cos(x[1]*2*pi) - 1]),
                     v)*dx


@pytest.fixture(params=["Poisson", "Mass"])
def problem(request):
    return request.param


@pytest.fixture
def a(problem, V):
    u = TrialFunction(V)
    v = TestFunction(V)
    if problem == "Poisson":
        return inner(grad(u), grad(v))*dx
    elif problem == "Mass":
        return inner(u, v)*dx


@pytest.fixture
def bcs(problem, V):
    if problem == "Poisson":
        return DirichletBC(V, zero(V.shape), (1, 2, 3, 4))
    elif problem == "Mass":
        return None


@pytest.mark.parametrize("pc_type", ("none",
                                     "ilu",
                                     "lu"))
@pytest.mark.parametrize("pmat_type", ("matfree", "aij"))
def test_assembled_pc_equivalence(V, a, L, bcs, tmpdir, pc_type, pmat_type):

    u = Function(V)

    assembled = str(tmpdir.join("assembled"))
    matrixfree = str(tmpdir.join("matrixfree"))

    assembled_parameters = {"ksp_type": "cg",
                            "pc_type": pc_type,
                            "ksp_monitor_short": "ascii:%s:" % assembled}
    u.assign(0)
    solve(a == L, u, bcs=bcs, solver_parameters=assembled_parameters)

    matrixfree_parameters = {"mat_type": "matfree",
                             "pmat_type": pmat_type,
                             "ksp_type": "cg",
                             "ksp_monitor_short": "ascii:%s:" % matrixfree}

    if pmat_type == "aij":
        matrixfree_parameters["pc_type"] = pc_type
    else:
        matrixfree_parameters["pc_type"] = "python"
        matrixfree_parameters["pc_python_type"] = "firedrake.AssembledPC"
        matrixfree_parameters["assembled_pc_type"] = pc_type

    u.assign(0)
    solve(a == L, u, bcs=bcs, solver_parameters=matrixfree_parameters)

    with open(assembled, "r") as f:
        f.readline()            # Skip over header
        expect = f.read()

    with open(matrixfree, "r") as f:
        f.readline()            # Skip over header
        actual = f.read()

    assert expect == actual


@pytest.mark.parametrize("bcs", [False, True],
                         ids=["no bcs", "bcs"])
def test_matrixfree_action(a, V, bcs):
    f = Function(V)
    expect = Function(V)
    actual = Function(V)

    x = SpatialCoordinate(V.mesh())
    if V.shape == ():
        f.interpolate(x[0]*sin(x[1]*2*pi))
    elif V.shape == (2, ):
        f.interpolate(as_vector([x[0]*sin(x[1]*2*pi),
                                 x[1]*cos(x[0]*2*pi)]))

    if bcs:
        bcs = DirichletBC(V, zero(V.shape), (1, 2))
    else:
        bcs = None
    A = assemble(a, bcs=bcs)
    Amf = assemble(a, mat_type="matfree", bcs=bcs)

    with f.dat.vec_ro as x:
        with expect.dat.vec as y:
            A.petscmat.mult(x, y)
        with actual.dat.vec as y:
            Amf.petscmat.mult(x, y)

    assert np.allclose(expect.dat.data_ro, actual.dat.data_ro)


@pytest.mark.parametrize("preassembled", [False, True],
                         ids=["variational", "preassembled"])
@pytest.mark.parametrize("rhs", ["form_rhs", "cofunc_rhs"])
@pytest.mark.parametrize("parameters",
                         [{"ksp_type": "preonly",
                           "pc_type": "python",
                           "pc_python_type": "firedrake.AssembledPC",
                           "assembled_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "fieldsplit_pc_type": "python",
                           "fieldsplit_pc_python_type": "firedrake.AssembledPC",
                           "fieldsplit_assembled_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "1",
                           "pc_fieldsplit_1_fields": "0,2",
                           "fieldsplit_0_pc_type": "python",
                           "fieldsplit_0_pc_python_type": "firedrake.MassInvPC",
                           "fieldsplit_0_Mp_pc_type": "lu",
                           "fieldsplit_1_pc_type": "python",
                           "fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
                           "fieldsplit_1_assembled_pc_type": "lu"},
                          {"ksp_type": "preonly",
                           "pc_type": "fieldsplit",
                           "pc_fieldsplit_type": "additive",
                           "pc_fieldsplit_0_fields": "1",
                           "pc_fieldsplit_1_fields": "0,2",
                           "fieldsplit_0_pc_type": "python",
                           "fieldsplit_0_pc_python_type": "firedrake.MassInvPC",
                           "fieldsplit_0_Mp_pc_type": "lu",
                           "fieldsplit_1_pc_type": "fieldsplit",
                           "fieldsplit_1_pc_fieldsplit_type": "additive",
                           "fieldsplit_1_fieldsplit_0_pc_type": "python",
                           "fieldsplit_1_fieldsplit_0_pc_python_type": "firedrake.MassInvPC",
                           "fieldsplit_1_fieldsplit_0_Mp_pc_type": "lu",
                           "fieldsplit_1_fieldsplit_1_pc_type": "python",
                           "fieldsplit_1_fieldsplit_1_pc_python_type": "firedrake.AssembledPC",
                           "fieldsplit_1_fieldsplit_1_assembled_pc_type": "lu"}])
def test_fieldsplitting(mesh, preassembled, parameters, rhs):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)
    Q = VectorFunctionSpace(mesh, "DG", 1)
    W = V*P*Q

    expect = Function(W)
    expect.sub(0).assign(1)
    expect.sub(1).assign(2)
    expect.sub(2).assign(Constant((3, 4)))

    u = TrialFunction(W)
    v = TestFunction(W)

    a = inner(u, v)*dx

    L = inner(expect, v)*dx
    if rhs == 'cofunc_rhs':
        L = assemble(L)
    elif rhs != 'form_rhs':
        raise ValueError("Unknown right hand side type")

    f = Function(W)

    if preassembled:
        A = assemble(a, mat_type="matfree")
        b = assemble(L)
        solve(A, f, b, solver_parameters=parameters)
    else:
        parameters["mat_type"] = "matfree"
        solve(a == L, f, solver_parameters=parameters)

    f -= expect

    for d in f.dat.data_ro:
        assert np.allclose(d, 0.0)


@pytest.mark.parallel(nprocs=4)
def test_matrix_free_split_communicators():

    wcomm = COMM_WORLD

    if wcomm.rank == 0:
        # On rank zero, we build a unit triangle,
        wcomm.Split(MPI.UNDEFINED)

        m = UnitTriangleMesh(comm=COMM_SELF)
        V = FunctionSpace(m, 'DG', 0)

        u = TrialFunction(V)
        v = TestFunction(V)

        volume = assemble(inner(u, v)*dx).M.values

        assert np.allclose(volume, 0.5)
    else:
        # On the other ranks, we'll build a collective mesh
        comm = wcomm.Split(0)

        m = UnitSquareMesh(4, 4, quadrilateral=True, comm=comm)

        V = VectorFunctionSpace(m, 'DG', 0)

        f = Function(V)

        u = TrialFunction(V)
        v = TestFunction(V)

        const = Constant((1, 0), domain=m)
        solve(inner(u, v)*dx == inner(const, v)*dx, f,
              solver_parameters={"mat_type": "matfree"})

        expect = Function(V).interpolate(const)
        assert np.allclose(expect.dat.data, f.dat.data)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize("infotype",
                         ["local", "sum", "max"])
def test_get_info(a, bcs, infotype):
    A = assemble(a, mat_type="matfree")
    ctx = A.petscmat.getPythonContext()

    itype = {"local": A.petscmat.InfoType.LOCAL,
             "sum": A.petscmat.InfoType.GLOBAL_SUM,
             "max": A.petscmat.InfoType.GLOBAL_MAX}[infotype]
    info = ctx.getInfo(A.petscmat, info=itype)
    test, trial = a.arguments()
    expect = ((test.function_space().dof_dset.total_size
               * test.function_space().value_size)
              + (trial.function_space().dof_dset.total_size
                 * trial.function_space().value_size))

    expect *= ScalarType.itemsize

    if infotype == "sum":
        expect = A.comm.allreduce(expect, op=MPI.SUM)
    elif infotype == "max":
        expect = A.comm.allreduce(expect, op=MPI.MAX)

    assert info["memory"] == expect

    if bcs is not None:
        A = assemble(a, mat_type="matfree", bcs=bcs)
        ctx = A.petscmat.getPythonContext()
        info = ctx.getInfo(A.petscmat, info=itype)
        assert info["memory"] == 2*expect


def test_duplicate(a, bcs):

    test, trial = a.arguments()

    if test.function_space().shape == ():
        rhs_form = inner(Constant(1), test)*dx
    elif test.function_space().shape == (2, ):
        rhs_form = inner(Constant((1, 1)), test)*dx

    if bcs is not None:
        Af = assemble(a, mat_type="matfree", bcs=bcs)
        rhs = assemble(rhs_form, bcs=bcs)
    else:
        Af = assemble(a, mat_type="matfree")
        rhs = assemble(rhs_form)

    # matrix-free duplicate creates a matrix-free copy of Af
    # we have not implemented the default copy = False
    B_petsc = Af.petscmat.duplicate(copy=True)

    ksp = PETSc.KSP().create()
    ksp.setOperators(Af.petscmat)
    ksp.setFromOptions()

    solution1 = Function(test.function_space())
    solution2 = Function(test.function_space())

    # Solve system with original matrix A
    with rhs.dat.vec_ro as b, solution1.dat.vec as x:
        ksp.solve(b, x)

    # Multiply with copied matrix B
    with solution1.dat.vec_ro as x, solution2.dat.vec_ro as y:
        B_petsc.mult(x, y)
    # Check if original rhs is equal to BA^-1 (rhs)
    assert np.allclose(rhs.vector().array(), solution2.vector().array())


def test_matrix_free_fieldsplit_with_real():
    mesh = RectangleMesh(10, 10, 1, 1)

    U = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    R = FunctionSpace(mesh, 'R', 0)
    V = MixedFunctionSpace([U, Q, R])

    eps = 1e-8

    w_, p_, l_water_ = TrialFunctions(V)
    v_, q_, m_water_ = TestFunctions(V)

    f = as_vector([0, -9.8])

    A = 2 * inner(sym(grad(w_)), sym(grad(v_))) * dx \
        - inner(p_, div(v_)) * dx \
        + inner(div(w_), q_) * dx \
        + eps * inner(p_, q_) * dx
    A += inner(dot(w_, f), m_water_) * dx - inner(l_water_, v_[1]) * dx
    L = inner(f, v_) * dx(domain=mesh)

    u_bdy = Function(U)
    bc = DirichletBC(V.sub(0), u_bdy, 'on_boundary')

    sol = Function(V)
    stokes_problem = LinearVariationalProblem(A, L, sol, bcs=bc)
    # Full Schur complement eliminating onto Real blocks.
    # The following is currently automatically set by `solving_utils.set_defaults()`.
    opts = {"mat_type": "matfree",
            "ksp_type": "fgmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": '0,1',
            "pc_fieldsplit_1_fields": '2',
            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled": DEFAULT_KSP_PARAMETERS,
            },
            "fieldsplit_1": {
                "ksp_type": "gmres",
                "pc_type": "none",
            }}
    stokes_solver = LinearVariationalSolver(stokes_problem, solver_parameters=opts)
    stokes_solver.solve()

import pytest
from firedrake import *
from firedrake.petsc import PETSc
from numpy.linalg import norm as np_norm
import gc


def count_refs(cls):
    """ Counts references of type `cls`
    """
    gc.collect()
    # A list comprehension here can trigger:
    #  > ReferenceError: weakly-referenced object no longer exists
    # So we count references the "slow" way and ignore `ReferenceError`s
    count = 0
    object_list = gc.get_objects()
    for obj in object_list:
        try:
            if isinstance(obj, cls):
                count += 1
        except ReferenceError:
            pass
    return count


@pytest.fixture
def a_L_out():
    mesh = UnitCubeMesh(1, 1, 1)
    fs = FunctionSpace(mesh, 'CG', 1)

    f = Function(fs)
    out = Function(fs)

    u = TrialFunction(fs)
    v = TestFunction(fs)
    return inner(u, v) * dx, inner(f, v) * dx, out


def test_linear_solver_api(a_L_out):
    a, L, out = a_L_out
    p = LinearVariationalProblem(a, L, out)
    solver = LinearVariationalSolver(p, solver_parameters={'ksp_type': 'cg'})

    assert solver.parameters['snes_type'] == 'ksponly'
    assert solver.parameters['ksp_rtol'] == 1e-7
    assert solver.snes.getType() == solver.snes.Type.KSPONLY
    assert solver.snes.getKSP().getType() == solver.snes.getKSP().Type.CG
    rtol, _, _, _ = solver.snes.getKSP().getTolerances()
    assert rtol == solver.parameters['ksp_rtol']


def test_petsc_options_cleared(a_L_out):
    a, L, out = a_L_out
    opts = PETSc.Options()
    original = {}
    original.update(opts.getAll())

    solve(a == L, out, solver_parameters={'foo': 'bar'})

    assert original == opts.getAll()


def test_linear_solver_gced(a_L_out):
    a, L, out = a_L_out

    before = count_refs(LinearVariationalSolver)
    solve(a == L, out)
    out.dat.data_ro  # force evaluation
    after = count_refs(LinearVariationalSolver)

    assert before == after


def test_assembled_solver_gced(a_L_out):
    a, L, out = a_L_out

    A = assemble(a)
    b = assemble(L)

    before = count_refs(LinearSolver)
    solve(A, out, b)
    out.dat.data_ro
    after = count_refs(LinearSolver)

    assert before == after


def test_nonlinear_solver_gced(a_L_out):
    a, L, out = a_L_out

    before = count_refs(NonlinearVariationalSolver)
    F = action(a, out) - L
    solve(F == 0, out)
    out.dat.data_ro  # force evaluation
    after = count_refs(NonlinearVariationalSolver)

    assert before == after


def test_nonlinear_solver_api(a_L_out):
    a, L, out = a_L_out
    J = a
    F = action(a, out) - L
    p = NonlinearVariationalProblem(F, out, J=J)
    solver = NonlinearVariationalSolver(p, solver_parameters={'snes_type': 'ksponly'})

    assert solver.snes.getType() == solver.snes.Type.KSPONLY
    rtol, _, _, _ = solver.snes.getTolerances()
    assert rtol == 1e-8


def test_nonlinear_solver_flattens_params(a_L_out):
    a, L, out = a_L_out
    J = a
    F = action(a, out) - L
    p = NonlinearVariationalProblem(F, out, J=J)

    solver1 = NonlinearVariationalSolver(
        p, solver_parameters={'snes_type': 'ksponly', 'ksp_rtol': 1e-10}
    )
    solver2 = NonlinearVariationalSolver(
        p, solver_parameters={'snes_type': 'ksponly', 'ksp': {'rtol': 1e-10}}
    )

    assert solver1.parameters["ksp_rtol"] == 1e-10
    assert solver2.parameters["ksp_rtol"] == 1e-10


def test_linear_solves_equivalent():
    """solve(a == L, out) should return the same as solving with the assembled objects.

    This relies on two different code paths agreeing on the same set of solver parameters."""
    mesh = UnitSquareMesh(50, 50)

    V = FunctionSpace(mesh, "CG", 1)

    f = Function(V)
    f.assign(1)
    f.vector()[:] = 1.
    t = TestFunction(V)
    q = TrialFunction(V)

    a = inner(q, t) * dx
    L = inner(f, t) * dx

    # Solve the system using forms
    sol = Function(V)
    solve(a == L, sol)

    # And again
    sol2 = Function(V)
    solve(a == L, sol2)
    assert np_norm(sol.vector()[:] - sol2.vector()[:]) == 0

    # Solve the system using preassembled objects
    sol3 = Function(V)
    solve(assemble(a), sol3, assemble(L))
    assert np_norm(sol.vector()[:] - sol3.vector()[:]) < 5e-14

    # Same, solving into vector
    sol4 = sol3.vector()
    solve(assemble(a), sol4, assemble(L))
    assert np_norm(sol.vector()[:] - sol4[:]) < 5e-14


def test_linear_solver_flattens_params(a_L_out):
    a, _, _ = a_L_out
    A = assemble(a)
    solver1 = LinearSolver(A, solver_parameters={"ksp_rtol": 1e-10})
    solver2 = LinearSolver(A, solver_parameters={"ksp": {"rtol": 1e-10}})

    assert solver1.parameters["ksp_rtol"] == 1e-10
    assert solver2.parameters["ksp_rtol"] == 1e-10


def test_constant_jacobian_lvs():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)

    q = Function(V)
    q.assign(1)
    a = q * inner(u, v) * dx

    f = Function(V)
    f.assign(1)
    L = inner(f, v) * dx

    out = Function(V)

    # Non-constant jacobian set
    lvp = LinearVariationalProblem(a, L, out, constant_jacobian=False)
    lvs = LinearVariationalSolver(lvp)

    lvs.solve()

    assert norm(assemble(out - f)) < 1e-7

    q.assign(5)

    lvs.solve()

    assert norm(assemble(out*5 - f)) < 2e-7

    q.assign(1)

    # This one should fail (because Jac is wrong)
    lvp = LinearVariationalProblem(a, L, out, constant_jacobian=True)
    lvs = LinearVariationalSolver(lvp)

    lvs.solve()

    assert norm(assemble(out - f)) < 1e-7

    q.assign(5)

    lvs.solve()

    assert not (norm(assemble(out*5 - f)) < 2e-7)


@pytest.fixture
def mesh(request):
    return UnitIntervalMesh(10)


def test_solve_cofunction_rhs(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    x, = SpatialCoordinate(mesh)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = Cofunction(V.dual())
    bcs = [DirichletBC(V, x, "on_boundary")]
    # Set the wrong BCs on the RHS
    for bc in bcs:
        bc.set(L, 888)
    Lold = L.copy()

    w = Function(V)
    solve(a == L, w, bcs=bcs)
    assert errornorm(x, w) < 1E-10
    assert np.allclose(L.dat.data, Lold.dat.data)


def test_solve_empty_form_rhs(mesh):
    V = FunctionSpace(mesh, "CG", 1)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = rhs(a)
    assert L.empty()
    x, = SpatialCoordinate(mesh)
    bcs = DirichletBC(V, x, "on_boundary")

    w = Function(V)
    solve(a == L, w, bcs)
    assert errornorm(x, w) < 1E-10


@pytest.mark.skipif(utils.complex_mode, reason="Differentiation of energy not defined in Complex.")
@pytest.mark.parametrize("mixed", (False, True), ids=("primal", "mixed"))
def test_solve_pre_apply_bcs(mesh, mixed):
    """Solve a 1D hyperelasticity problem with linear exact solution.
       The default DirichletBC treatment would raise NaNs if the problem is
       linearised around an initial guess that satisfies the bcs and is zero on the interior.
       Here we test that we can linearise around an initial guess that
       does not satisfy the DirichletBCs by passing pre_apply_bcs=True."""

    V = VectorFunctionSpace(mesh, "CG", 1)
    if mixed:
        Q = FunctionSpace(mesh, "DG", 0)
        Z = V * Q
        V = Z.sub(0)
        z = Function(Z)
        u, p = split(z)
    else:
        u = Function(V)
        z = u
        p = None

    # Boundary conditions
    eps = Constant(0.1)
    x = SpatialCoordinate(mesh)
    g = -eps * x
    bcs = [DirichletBC(V, g, "on_boundary")]

    # Hyperelastic energy functional
    lam = Constant(1E3)
    dim = mesh.geometric_dimension()
    F = grad(u) + Identity(dim)
    J = det(F)
    logJ = 0.5*ln(J**2)
    if p is None:
        p = lam*logJ

    W = (1/2)*(inner(F, F) - dim - 2*logJ) * dx + (inner(p, logJ - (1/(2*lam))*p)) * dx
    uh = z.subfunctions[0]

    # Raises NaNs if pre_apply_bcs=True
    F = derivative(W, z)
    z.zero()
    solve(F == 0, z, bcs, pre_apply_bcs=False)
    assert errornorm(g, uh) < 1E-10

    # Test that pre_apply_bcs=False works with a linear problem
    a = derivative(F, z)
    L = Form([])
    z.zero()
    solve(a == L, z, bcs, pre_apply_bcs=False)
    assert errornorm(g, uh) < 1E-10

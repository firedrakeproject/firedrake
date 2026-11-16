"""Solve a mixed Helmholtz problem

sigma + grad(u) = 0,
u + div(sigma) = f,
u = 42 on the exterior boundary,

using hybridisation with SLATE performing the forward elimination and
backwards reconstructions. The finite element variational problem is:

(sigma, tau) - (u, div(tau)) = -<42*tau, n>
(div(sigma), v) + (u, v) = (f, v)

for all tau and v. The forcing function is chosen as:

(1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2),

and the strong boundary condition along the boundary is:

u = 42

which appears in the variational form as the term: -<42*tau, n>
"""
import pytest

from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


@pytest.fixture(scope="module")
def setup_poisson():
    p = 3
    n = 2
    mesh = UnitSquareMesh(n, n)
    S = FunctionSpace(mesh, "RT", p+1)
    V = FunctionSpace(mesh, "DG", p)
    W = S * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    rg = RandomGenerator(PCG64(seed=1234))
    f = rg.uniform(V)

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = -inner(f, v) * dx
    return a, L, W


@pytest.fixture(scope="module")
def setup_poisson_3D():
    p = 3
    n = 2
    mesh = UnitCubeMesh(n, n, n)
    S = FunctionSpace(mesh, "RT", p+1)
    V = FunctionSpace(mesh, "DG", p)
    W = S * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    x, y, z = SpatialCoordinate(mesh)
    expr = (1+12*pi*pi)*cos(100*pi*x)*cos(100*pi*y)*cos(100*pi*z)
    f = Function(V).interpolate(expr)

    # Define the variational forms
    a = inner(sigma, tau) * dx(degree=8) + (inner(u, div(tau)) + inner(div(sigma), v)) * dx(degree=6)
    L = -inner(f, v)*dx(degree=8)
    return a, L, W


@pytest.fixture(params=[("RT", "DG", "DG"),
                        ("DG", "RT", "BDM"),
                        ("RT", "DG", "BDM", "DG"),
                        ("DG", "BDM", "DG", "RT")])
def multi_field_problem(request: pytest.FixtureRequest) -> tuple:
    mesh = UnitSquareMesh(2, 2)
    families = request.param
    W = MixedFunctionSpace([FunctionSpace(mesh, family, 0 if family == "DG" else 1)
                            for family in families])
    trial = TrialFunctions(W)
    test = TestFunctions(W)
    hdiv_fields = tuple(i for i, family in enumerate(families) if family != "DG")
    l2_fields = tuple(i for i, family in enumerate(families) if family == "DG")
    a = sum(inner(trial[i], test[i]) for i in range(len(W))) * dx
    for fields in (hdiv_fields, l2_fields):
        a += sum(0.1 * inner(trial[i], test[j]) * dx
                 for i in fields for j in fields if i != j)
    a += sum((inner(div(trial[i]), test[j]) - inner(trial[j], div(test[i])))
             for i in hdiv_fields for j in l2_fields) * dx
    x, y = SpatialCoordinate(mesh)
    L = sum((i + 1) * (1 + x + y) * test[i] for i in l2_fields) * dx
    return a, L, W, hdiv_fields


@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("strong_bcs", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_slate_hybridization_multi_field(multi_field_problem: tuple, strong_bcs: bool, nested: bool) -> None:
    a, L, W, hdiv_fields = multi_field_problem
    bcs = [DirichletBC(W.sub(i), 0, j + 1)
           for j, i in enumerate(hdiv_fields)] if strong_bcs else []
    hybridization = {"ksp_type": "preonly", "pc_type": "lu",
                     "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER}
    if nested:
        hybridization["localsolve"] = {"ksp_type": "preonly", "pc_type": "fieldsplit",
                                       "pc_fieldsplit_type": "schur"}
    solution = Function(W)
    solve(a == L, solution, bcs=bcs,
          solver_parameters={"mat_type": "matfree", "ksp_type": "preonly",
                             "pc_type": "python", "pc_python_type": "firedrake.HybridizationPC",
                             "hybridization": hybridization})
    reference = Function(W)
    solve(a == L, reference, bcs=bcs,
          solver_parameters={"mat_type": "aij", "ksp_type": "preonly", "pc_type": "lu",
                             "pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER})
    for i in range(len(W)):
        assert errornorm(solution.sub(i), reference.sub(i)) < 1.e-11


def options_check(builder, expected):
    return all(bool(getattr(builder, k)) == v for k, v in expected.items())


@pytest.mark.parametrize(("degree", "hdiv_family", "quadrilateral"),
                         [(1, "RT", False), (1, "RTCF", True),
                          (2, "RT", False), (2, "RTCF", True)])
def test_slate_hybridization(degree, hdiv_family, quadrilateral):
    # Create a mesh
    mesh = UnitSquareMesh(6, 6, quadrilateral=quadrilateral)
    RT = FunctionSpace(mesh, hdiv_family, degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define the variational forms
    a = inner(sigma, tau)*dx + (-inner(u, div(tau)) + inner(u, v) + inner(div(sigma), v)) * dx(degree=2*(degree-1))
    L = inner(f, v) * dx - 42 * inner(n, tau)*ds

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu'}}
    solve(a == L, w, solver_parameters=params)
    sigma_h, u_h = w.subfunctions

    # (Non-hybrid) Need to slam it with preconditioning due to the
    # system's indefiniteness
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'schur',
                             'ksp_type': 'cg',
                             'ksp_rtol': 1e-14,
                             'pc_fieldsplit_schur_fact_type': 'FULL',
                             'fieldsplit_0_ksp_type': 'cg',
                             'fieldsplit_1_ksp_type': 'cg'})
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11


def test_slate_hybridization_wrong_option(setup_poisson, petsc_raises):
    a, L, W = setup_poisson

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'frog'}}}
    problem = LinearVariationalProblem(a, L, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)

    with petsc_raises(ValueError):
        solver.solve()


def test_slate_hybridization_nested_schur(setup_poisson):
    a, L, W = setup_poisson

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur'}}}

    problem = LinearVariationalProblem(a, L, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': False,
                'preonly_Shat': False, 'jacobi_Shat': False}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."
    sigma_h, u_h = w.subfunctions

    w2 = Function(W)
    solve(a == L, w2, solver_parameters={'ksp_type': 'preonly',
                                         'pc_type': 'python',
                                         'mat_type': 'matfree',
                                         'pc_python_type': 'firedrake.HybridizationPC',
                                         'hybridization': {'ksp_type': 'preonly',
                                                           'pc_type': 'lu'}})
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11


class DGLaplacian(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        W = trial.function_space()
        n = FacetNormal(W.mesh())
        alpha = Constant(3**3)
        gamma = Constant(4**3)
        h = CellSize(W.mesh())
        h_avg = (h('+') + h('-'))/2
        a_dg = -(inner(grad(trial), grad(test)) * dx
                 + (- inner(jump(trial, n), avg(grad(test)))
                    - inner(avg(grad(trial)), jump(test, n))
                    + (alpha/h_avg) * inner(jump(trial, n), jump(test, n))) * dS
                 + (- inner(trial*n, grad(test))
                    - inner(grad(trial), test*n)
                    + (gamma/h)*inner(trial, test)) * ds)
        bcs = None
        return (a_dg, bcs)


def test_mixed_poisson_approximated_schur(setup_poisson):
    """A test, which compares a solution to a 2D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with uses a user supplied operator as preconditioner to the
    Schur solver.

    NOTE: With the setup in this test, using the approximated schur complemement
    defined as DGLaplacian as a preconditioner to the schur complement,
    reduces the condition number of the local solve from 16.77 to 6.06.
    """
    a, L, W = setup_poisson

    # setup first solver
    w = Function(W)
    params = {'ksp_type': 'preonly',
              'pc_type': 'python',
              'mat_type': 'matfree',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-8,
                                'mat_type': 'matfree',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur',
                                               'fieldsplit_1': {'ksp_type': 'default',
                                                                'pc_type': 'python',
                                                                'pc_python_type': __name__ + '.DGLaplacian'}}}}

    problem = LinearVariationalProblem(a, L, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # double-check options are set as expected
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': True,
                'preonly_Shat': False, 'jacobi_Shat': False}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.subfunctions

    # setup second solver
    w2 = Function(W)
    aij_params = {'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'mat_type': 'matfree',
                  'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'cg',
                                    'pc_type': 'none',
                                    'ksp_rtol': 1e-9,
                                    'mat_type': 'matfree'}}
    solve(a == L, w2, solver_parameters=aij_params)
    _sigma, _u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_slate_hybridization_jacobi_prec_A00(setup_poisson_3D):
    """A test, which compares a solution to a 3D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with the same solver but which has a nested schur complement
    in the trace solve operator and a jacobi preconditioner on the A00 block.

    NOTE: With the setup in this test, using jacobi as a preconditioner to the
    schur complement matrix, the condition number of the matrix of the local solve
    P.inv * A.solve(...) is reduced from 36.59 to 3.06.
    """
    a, L, W = setup_poisson_3D

    # setup first solver
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-12,
                                'mat_type': 'matfree',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur',
                                               'fieldsplit_0': {'ksp_type': 'default',
                                                                'pc_type': 'jacobi'}}}}
    problem = LinearVariationalProblem(a, L, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # double-check options are set as expected
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': True,
                'schur_approx': False,
                'preonly_Shat': False, 'jacobi_Shat': False,
                'preonly_S': False, 'jacobi_S': False}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.subfunctions

    # setup second solver
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'mat_type': 'matfree',
                             'ksp_type': 'preonly',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization': {'ksp_type': 'cg',
                                               'pc_type': 'none',
                                               'ksp_rtol': 1e-9,
                                               'mat_type': 'matfree'}})
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)
    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_slate_hybridization_jacobi_prec_schur(setup_poisson_3D):
    """A test, which compares a solution to a 3D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with the same solver but which has a nested schur complement
    in the trace solve operator and a jacobi preconditioner on the schur
    complement.

    NOTE With the setup in this test, using jacobi as apreconditioner to the
    schur complement matrix the condition number of the matrix of the local solve
    P.inv * A.solve(...) is reduced from 17.13 to 16.71
    """
    a, L, W = setup_poisson_3D

    # setup first solver
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-12,
                                'mat_type': 'matfree',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur',
                                               'fieldsplit_1': {'ksp_type': 'default',
                                                                'pc_type': 'jacobi'}}}}
    problem = LinearVariationalProblem(a, L, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # double-check options are set as expected
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': False,
                'preonly_S': False, 'jacobi_S': True}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.subfunctions

    # setup second solver
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'mat_type': 'matfree',
                             'ksp_type': 'preonly',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization': {'ksp_type': 'cg',
                                               'pc_type': 'none',
                                               'ksp_rtol': 1e-9,
                                               'mat_type': 'matfree'}})
    nh_sigma, nh_u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_mixed_poisson_approximated_schur_jacobi_prec(setup_poisson):
    """A test, which compares a solution to a 2D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system where
    the a user supplied operator is used as preconditioner to the
    Schur solver to a solution where the user supplied operator is replaced
    with the jacobi preconditioning operator.
    """
    a, L, W = setup_poisson

    # setup first solver
    w = Function(W)
    params = {'ksp_type': 'preonly',
              'pc_type': 'python',
              'mat_type': 'matfree',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-8,
                                'mat_type': 'matfree',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur',
                                               'fieldsplit_1': {'ksp_type': 'default',
                                                                'pc_type': 'python',
                                                                'pc_python_type': __name__ + '.DGLaplacian',
                                                                'aux_ksp_type': 'preonly',
                                                                'aux_pc_type': 'jacobi'}}}}

    problem = LinearVariationalProblem(a, L, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # double-check options are set as expected
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': True,
                'preonly_S': False, 'jacobi_S': False,
                'preonly_Shat': True, 'jacobi_Shat': True}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.subfunctions

    # setup second solver
    w2 = Function(W)
    aij_params = {'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'mat_type': 'matfree',
                  'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'cg',
                                    'pc_type': 'none',
                                    'ksp_rtol': 1e-8,
                                    'mat_type': 'matfree'}}
    solve(a == L, w2, solver_parameters=aij_params)
    _sigma, _u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


@pytest.mark.parametrize('counts', [(10001, 10002), (10002, 10001)])
def test_slate_hybridization_count_safe(counts):
    g_count, c_count = counts
    mesh = UnitTriangleMesh()
    BDM = FunctionSpace(mesh, "BDM", 2)
    DG = FunctionSpace(mesh, "DG", 1)
    V = BDM * DG
    VectorDG = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(V)
    v = TestFunction(V)
    g = Function(VectorDG, count=g_count)
    c = Function(DG, count=c_count)
    a = (
        inner(u, v) * dx
        + inner(g[0] * u[0], v[0]) * dx
        + inner(c * grad(u[0]), grad(v[0])) * dx
    )
    sol = Function(V)
    solver_parameters = {
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
    }
    solve(lhs(a) == rhs(a), sol, solver_parameters=solver_parameters)

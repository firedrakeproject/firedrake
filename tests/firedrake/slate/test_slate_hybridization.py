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
    f = Function(V)
    import numpy as np
    fvector = f.vector()
    fvector.set_local(np.random.uniform(size=fvector.local_size()))

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
    a = dot(sigma, tau) * dx(degree=8) + (inner(u, div(tau)) + inner(div(sigma), v)) * dx(degree=6)
    L = -f*v*dx(degree=8)
    return a, L, W


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
    def form(self, pc, u, v):
        W = u.function_space()
        n = FacetNormal(W.mesh())
        alpha = Constant(3**3)
        gamma = Constant(4**3)
        h = CellSize(W.mesh())
        h_avg = (h('+') + h('-'))/2
        a_dg = -(inner(grad(u), grad(v)) * dx
                 + (- inner(jump(u, n), avg(grad(v)))
                    - inner(avg(grad(u)), jump(v, n))
                    + (alpha/h_avg) * inner(jump(u, n), jump(v, n))) * dS
                 + (- inner(u*n, grad(v))
                    - inner(grad(u), v*n)
                    + (gamma/h)*inner(u, v)) * ds)
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


def test_slate_hybridization_gusto():

    # ---------------------------------------------------------------------------- #
    # Set up core objects
    # ---------------------------------------------------------------------------- #

    degree = 1
    radius = Constant(6371220.)
    dt = Constant(3000.0)
    H = Constant(3e4/9.80616)
    g = Constant(9.80616)
    Omega = Constant(7.292e-5)
    alpha = Constant(0.5)
    beta_u = alpha*dt
    beta_d = alpha*dt

    mesh = IcosahedralSphereMesh(float(radius), refinement_level=0, degree=2)
    x = SpatialCoordinate(mesh)

    # Function spaces
    VDG = FunctionSpace(mesh, "DG", degree)
    VHDiv = FunctionSpace(mesh, "BDM", degree+1)
    mesh.init_cell_orientations(x)

    VCG1 = FunctionSpace(mesh, 'CG', 1)
    Vmixed = MixedFunctionSpace((VHDiv, VDG, VDG))
    Vreduced = MixedFunctionSpace((VHDiv, VDG))

    # Mixed RHS and LHS functions
    x_rhs = Function(Vmixed)

    # Components of various mixed functions
    u_rhs, D_rhs, b_rhs = split(x_rhs)

    uD_lhs = Function(Vreduced)
    u_test, D_test = TestFunctions(Vreduced)
    u_trial, D_trial = TrialFunctions(Vreduced)
    u_lhsr, D_lhsr = uD_lhs.subfunctions

    # Reference profiles
    uDb_bar = Function(Vmixed)
    D_bar = split(uDb_bar)[1]
    b_bar = split(uDb_bar)[2]
    D_bar_subfunc = uDb_bar.subfunctions[1]
    b_bar_subfunc = uDb_bar.subfunctions[2]

    # Set up perp function
    sphere_degree = mesh.coordinates.function_space().ufl_element().degree()
    VecDG = VectorFunctionSpace(mesh, 'DG', sphere_degree)
    N = Function(VecDG).interpolate(CellNormal(mesh))
    perp = lambda u: cross(N, u)

    # Coriolis
    f = Function(VCG1)

    # ---------------------------------------------------------------------------- #
    # Set up problem and solvers
    # ---------------------------------------------------------------------------- #

    # We have a 3-function mixed space for wind, depth and buoyancy
    # Strategy is to approximately eliminate buoyancy, leaving a wind-depth problem
    # The wind-depth problem is solved using the hydridization preconditioner

    # Approximate elimination of b
    b = -0.5 * dt * dot(u_trial, grad(b_bar)) + b_rhs

    solver_parameters = {
        'ksp_type': 'preonly',
        'mat_type': 'matfree',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {
            'ksp_type': 'cg',
            'pc_type': 'gamg',
            'ksp_rtol': 1e-1,
            'mg_levels': {
                'ksp_type': 'chebyshev',
                'ksp_max_it': 1,
                'pc_type': 'bjacobi',
                'sub_pc_type': 'ilu'
            }
        }
    }

    n = FacetNormal(mesh)

    # Linear thermal shallow water problem
    uD_eqn = (
        inner(u_test, (u_trial - u_rhs)) * dx
        - beta_u * (D_trial - D_bar) * div(u_test * b_bar) * dx
        + beta_u * jump(u_test * b_bar, n) * avg(D_trial - D_bar) * dS
        - beta_u * 0.5 * D_bar * b_bar * div(u_test) * dx
        - beta_u * 0.5 * D_bar * b * div(u_test) * dx
        - beta_u * 0.5 * b_bar * div(u_test*(D_trial - D_bar)) * dx
        + beta_u * 0.5 * jump((D_trial - D_bar)*u_test, n) * avg(b_bar) * dS
        + inner(D_test, (D_trial - D_rhs)) * dx
        + beta_d * D_test * div(D_bar*u_trial) * dx
        + beta_u * f * inner(u_test, perp(u_trial)) * dx
    )

    # Boundary conditions
    bcs = []

    # Solver for u, D
    uD_problem = LinearVariationalProblem(
        lhs(uD_eqn), rhs(uD_eqn), uD_lhs, bcs=bcs, constant_jacobian=True
    )

    # Provide callback for the nullspace of the trace system
    def trace_nullsp(T):
        return VectorSpaceBasis(constant=True)

    appctx = {"trace_nullspace": trace_nullsp}
    uD_solver = LinearVariationalSolver(
        uD_problem, solver_parameters=solver_parameters, appctx=appctx
    )

    # ---------------------------------------------------------------------------- #
    # Set some initial conditions
    # ---------------------------------------------------------------------------- #

    f.interpolate(2*Omega*x[2]/radius)
    D_bar_subfunc.assign(H)
    b_bar_subfunc.assign(g)

    # Simplest test is to give a right-hand side that is zero
    x_rhs.assign(1.0)

    # ---------------------------------------------------------------------------- #
    # Run
    # ---------------------------------------------------------------------------- #

    uD_solver.solve()


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

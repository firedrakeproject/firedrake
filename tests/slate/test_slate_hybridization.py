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


def setup_poisson(p, quad):
    n = 2
    mesh = UnitSquareMesh(n, n, quadrilateral=quad)
    space1, space2 = ("RTCF", "DQ") if quad else ("RT", "DG")
    U = FunctionSpace(mesh, space1, p+1)
    V = FunctionSpace(mesh, space2, p)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    x = SpatialCoordinate(mesh)
    exact = (10)*x[0]*(1-x[0])*x[1]*(1-x[1])
    f = -div(grad(exact))

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = -inner(f, v) * dx
    return a, L, W


def setup_poisson_3D(p):
    n = 2
    mesh = SquareMesh(n, n, 1, quadrilateral=True)
    mesh = ExtrudedMesh(mesh, n)
    RT = FiniteElement("RTCF", quadrilateral, p+1)
    DG_v = FiniteElement("DG", interval, p)
    DG_h = FiniteElement("DQ", quadrilateral, p)
    CG = FiniteElement("CG", interval, p+1)
    HDiv_ele = EnrichedElement(HDiv(TensorProductElement(RT, DG_v)),
                               HDiv(TensorProductElement(DG_h, CG)))
    V = FunctionSpace(mesh, HDiv_ele)
    U = FunctionSpace(mesh, "DQ", p)
    W = V * U
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    V, U = W.split()
    x = SpatialCoordinate(mesh)
    exact = 100*x[0]*(1-x[0])*x[1]*(1-x[1])*x[2]*(1-x[2])
    f = -div(grad(exact))
    a = (dot(sigma, tau) + div(tau)*u + div(sigma)*v)*dx(degree=8)
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
    a = (inner(sigma, tau) - inner(u, div(tau)) + inner(u, v) + inner(div(sigma), v)) * dx
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
    sigma_h, u_h = w.split()

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
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11


def test_slate_hybridization_wrong_option():
    a, L, W = setup_poisson()

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
    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    with pytest.raises(ValueError):
        # HybridizationPC isn't called directly from the Python interpreter,
        # it's a callback that PETSc calls. This means that the call stack from pytest
        # down to HybridizationPC goes via PETSc C code, which interferes with the exception
        # before it is observed outside. Hence removing PETSc's error handler
        # makes the problem go away, because PETSc stops interfering.
        # We need to repush the error handler because popErrorHandler globally changes
        # the system state for all future tests.
        from firedrake.petsc import PETSc
        PETSc.Sys.pushErrorHandler("ignore")
        solver.solve()
        PETSc.Sys.popErrorHandler("ignore")


@pytest.mark.parametrize("local_matfree", [True, False])
def test_slate_hybridization_nested_schur(local_matfree):
    # Take lower order for local matrix-free solve
    # so that the test does not run too long
    s = (1, True) if local_matfree else (3, False)
    a, L, W = setup_poisson(*s)

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'mat_type': 'matfree',
                                'rtol': 1e-8,
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur'}}}
    if local_matfree:
        params['hybridization']['localsolve']['mat_type'] = 'matfree'

    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': False,
                'preonly_Shat': False, 'jacobi_Shat': False}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."
    sigma_h, u_h = w.split()

    w2 = Function(W)
    solve(a == L, w2, solver_parameters={'ksp_type': 'preonly',
                                         'pc_type': 'python',
                                         'mat_type': 'matfree',
                                         'pc_python_type': 'firedrake.HybridizationPC',
                                         'hybridization': {'ksp_type': 'preonly',
                                                           'pc_type': 'lu'}})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


class DGLaplacian(AuxiliaryOperatorPC):
    def form(self, pc, u, v):
        W = u.function_space()
        n = FacetNormal(W.mesh())
        alpha = Constant(3**5)
        gamma = Constant(4**5)
        h = CellVolume(W.mesh())/FacetArea(W.mesh())
        h_avg = (h('+') + h('-'))/2
        a_dg = -(inner(grad(u), grad(v))*dx
                 - inner(jump(u, n), avg(grad(v)))*dS
                 - inner(avg(grad(u)), jump(v, n), )*dS
                 + alpha/h_avg * inner(jump(u, n), jump(v, n))*dS
                 - inner(u*n, grad(v))*ds
                 - inner(grad(u), v*n)*ds
                 + (gamma/h)*inner(u, v)*ds)
        bcs = None
        return (a_dg, bcs)


class DGLaplacian3D(AuxiliaryOperatorPC):
    def form(self, pc, u, v):
        W = u.function_space()
        n = FacetNormal(W.mesh())
        gamma = Constant(4.**3)
        h = CellVolume(W.mesh())/FacetArea(W.mesh())

        a_dg = -(dot(grad(v), grad(u))*dx(degree=8)
                 - dot(grad(v), (u)*n)*ds_v(degree=8)
                 - dot(v*n, grad(u))*ds_v(degree=8)
                 + gamma/h*dot(v, u)*ds_v(degree=8)
                 - dot(grad(v), (u)*n)*ds_t(degree=8)
                 - dot(v*n, grad(u))*ds_t(degree=8)
                 + gamma/h*dot(v, u)*ds_t(degree=8)
                 - dot(grad(v), (u)*n)*ds_b(degree=8)
                 - dot(v*n, grad(u))*ds_b(degree=8)
                 + gamma/h*dot(v, u)*ds_b(degree=8))

        bcs = []
        return (a_dg, bcs)


@pytest.mark.parametrize("local_matfree", [True, False])
def test_mixed_poisson_approximated_schur(local_matfree):
    """A test, which compares a solution to a 2D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with uses a user supplied operator as preconditioner to the
    Schur solver.
    """
    # setup FEM
    # Take lower order for local matrix-free solve
    # so that the test does not run too long
    s = (1, True) if local_matfree else (3, True)
    a, L, W = setup_poisson(*s)

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
    if local_matfree:
        params['hybridization']['localsolve']['mat_type'] = 'matfree'

    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # double-check options are set as expected
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': True,
                'preonly_Shat': False, 'jacobi_Shat': False}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.split()

    # setup second solver
    w2 = Function(W)
    aij_params = {'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'mat_type': 'matfree',
                  'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'preonly',
                                    'pc_type': 'lu'}}
    solve(a == L, w2, solver_parameters=aij_params)
    _sigma, _u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


@pytest.mark.parametrize("local_matfree", [True, False])
def test_slate_hybridization_jacobi_prec_A00(local_matfree):
    """A test, which compares a solution to a 3D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with the same solver but which has a nested schur complement
    in the trace solve operator and a jacobi preconditioner on the A00 block.

    NOTE: With the setup of this test on RTCF4-DQ3,
    using jacobi as a preconditioner to the
    A00 block, the condition number of the matrix of the local solve
    P.inv * A.solve(...) is reduced from 36.59 to 3.06.
    """
    # Take lower order for local matrix-free solve
    # so that the test does not run too long
    p = 0 if local_matfree else 3
    a, L, W = setup_poisson_3D(p)

    # setup first solver
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-8,
                                'mat_type': 'matfree',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur',
                                               'fieldsplit_0': {'ksp_type': 'default',
                                                                'pc_type': 'jacobi'}}}}
    if local_matfree:
        params['hybridization']['localsolve']['mat_type'] = 'matfree'
    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
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

    sigma_h, u_h = w.split()

    # setup second solver
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'mat_type': 'matfree',
                             'ksp_type': 'preonly',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization': {'ksp_type': 'cg',
                                               'pc_type': 'none',
                                               'ksp_rtol': 1e-12,
                                               'mat_type': 'matfree'}})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)
    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_slate_hybridization_jacobi_prec_schur():
    """A test, which compares a solution to a 3D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with the same solver but which has a nested schur complement
    in the trace solve operator and a jacobi preconditioner on the schur
    complement.

    NOTE With the setup in this test, using jacobi as apreconditioner to the
    schur complement matrix the condition number of the matrix of the local solve
    P.inv * A.solve(...) is reduced from 17.13 to 16.71
    
    NOTE We can't do this locally matfree because we don't know
    how to implement diag(Schur complement) in a matrix-free way
    """
    # setup FEM
    a, L, W = setup_poisson_3D(3)

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
    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # double-check options are set as expected
    expected = {'nested': True,
                'preonly_A00': False, 'jacobi_A00': False,
                'schur_approx': False,
                'preonly_S': False, 'jacobi_S': True}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.split()

    # setup second solver
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'mat_type': 'matfree',
                             'ksp_type': 'preonly',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization': {'ksp_type': 'cg',
                                               'pc_type': 'none',
                                               'ksp_rtol': 1e-8,
                                               'mat_type': 'matfree'}})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


@pytest.mark.parametrize("local_matfree", [True, False])
def test_mixed_poisson_approximated_schur_jacobi_prec(local_matfree):
    """A test, which compares a solution to a 2D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system where
    an operator carrying the diagonal of a user supplied operator is preconditioning (inner)
    Schur complement solver.
    """
    # setup FEM
    s = (1, True) if local_matfree else (3, True)
    a, L, W = setup_poisson_3D(*s)

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

    if local_matfree:
        params['hybridization']['localsolve']['mat_type'] = 'matfree'

    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
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

    sigma_h, u_h = w.split()

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
    _sigma, _u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_slate_hybridization_full_local_prec():
    """A test, which compares a solution to a 3D mixed Poisson problem solved
    globally matrixfree with a HybridizationPC and CG on the trace system to
    a solution with the same solver but which has a nested schur complement
    in the trace solve operator, a jacobi preconditioner on the A00 block, 
    and an operator carrying the diagonal of a user supplied operator
    is preconditioning the (inner) Schur complement solver.

    """
    # setup FEM
    a, L, W = setup_poisson_3D(1)

    # setup first solver
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-8,
                                'ksp_atol': 1e-8,
                                'mat_type': 'matfree',
                                'localsolve': {'ksp_type': 'preonly',
                                               'pc_type': 'fieldsplit',
                                               'pc_fieldsplit_type': 'schur',
                                               'mat_type': 'matfree',
                                               'fieldsplit_0': {'ksp_type': 'default',
                                                                'pc_type': 'jacobi'},
                                               'fieldsplit_1': {'ksp_type': 'default',
                                                                'pc_type': 'python',
                                                                'pc_python_type': __name__ + '.DGLaplacian3D',
                                                                'aux_ksp_type': 'preonly',
                                                                'aux_pc_type': 'jacobi'}}}}

    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
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

    sigma_h, u_h = w.split()

    # setup second solver
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'mat_type': 'matfree',
                             'ksp_type': 'preonly',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization': {'ksp_type': 'cg',
                                               'pc_type': 'none',
                                               'ksp_rtol': 1e-16,
                                               'mat_type': 'matfree'}})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)
    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_slate_hybridization_global_matfree_jacobi():
    a, L, W = setup_poisson()

    w = Function(W)
    jacobi_matfree_params = {'mat_type': 'matfree',
                             'ksp_type': 'cg',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization': {'ksp_type': 'cg',
                                               'pc_type': 'jacobi',
                                               'mat_type': 'matfree',
                                               'ksp_rtol': 1e-8}}

    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=jacobi_matfree_params)
    solver.solve()
    sigma_h, u_h = w.split()

    w2 = Function(W)
    solve(a == L, w2, solver_parameters={'ksp_type': 'preonly',
                                         'pc_type': 'python',
                                         'mat_type': 'matfree',
                                         'pc_python_type': 'firedrake.HybridizationPC',
                                         'hybridization': {'ksp_type': 'preonly',
                                                           'pc_type': 'lu'}})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8

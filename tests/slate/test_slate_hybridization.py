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


def setup_poisson():
    mesh = UnitSquareMesh(1, 1)
    U = FunctionSpace(mesh, "RT", 4)
    V = FunctionSpace(mesh, "DG", 3)
    W = U * V
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


def options_check(builder, expected):
    all_good = True
    for k, v in expected.items():
        all_good = bool(getattr(builder, k)) == v
        if not all_good:
            return False
    return all_good


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


def test_slate_hybridization_nested_schur():
    a, L, W = setup_poisson()

    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'lmi': {'ksp_type': 'preonly',
                                        'pc_type': 'fieldsplit',
                                        'fieldsplit_type': 'schur'}}}
    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    expected = {'nested':True,
                'preonly_A00':False, 'jacobi_A00':False,
                'schur_approx':False,
                'preonly_Shat':False, 'jacobi_Shat':False}
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
        a_dg = -(inner(grad(u), grad(v))*dx
                 - inner(jump(u, n), avg(grad(v)))*dS
                 - inner(avg(grad(u)), jump(v, n), )*dS
                 + alpha/h_avg * inner(jump(u, n), jump(v, n))*dS
                 - inner(u*n, grad(v))*ds
                 - inner(grad(u), v*n)*ds
                 + (gamma/h)*inner(u, v)*ds)
        bcs = None
        return (a_dg, bcs)


def test_mixed_poisson_approximated_schur():
    # NOTE With the setup in this test, using the approximated schur complemement
    # defined as DGLaplacian as a preconditioner to the schur complement in the
    # reconstruction calls reduces the condition number of the local solve from
    # 16.77 to 5.95
    a, L, W = setup_poisson()

    # Compare hybridized solution with non-hybridized
    w = Function(W)
    bcs = []

    w = Function(W)
    params = {'ksp_type': 'preonly',
              'pc_type': 'python',
              'mat_type': 'matfree',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'cg',
                                'pc_type': 'none',
                                'ksp_rtol': 1e-8,
                                'mat_type': 'matfree',
                                'lmi': {'ksp_type': 'preonly',
                                        'pc_type': 'fieldsplit',
                                        'fieldsplit_type': 'schur',
                                        'fieldsplit_1': {'ksp_type': 'default',
                                                         'pc_type': 'python',
                                                         'pc_python_type': __name__ + '.DGLaplacian'}}}}

    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    expected = {'nested':True,
                'preonly_A00':False, 'jacobi_A00':False,
                'schur_approx':True,
                'preonly_Shat':False, 'jacobi_Shat':False}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."
    sigma_h, u_h = w.split()

    w2 = Function(W)
    aij_params = {'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'mat_type': 'matfree',
                  'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'cg',
                                    'pc_type': 'none',
                                    'ksp_rtol': 1e-8,
                                    'mat_type': 'matfree'}}
    solve(a == L, w2, bcs=bcs, solver_parameters=aij_params)
    _sigma, _u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


def test_slate_hybridization_jacobi_prec_schur():
    a, L, W = setup_poisson()

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    params = {'mat_type': 'matfree',
            'ksp_type': 'preonly',
            'pc_type': 'python',
            'pc_python_type': 'firedrake.HybridizationPC',
            'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'lmi': {'ksp_type': 'preonly',
                                        'pc_type': 'fieldsplit',
                                        'fieldsplit_type': 'schur',
                                        'fieldsplit_0': {'ksp_type': 'default',
                                                        'pc_type': 'jacobi'},
                                        'fieldsplit_1': {'ksp_type': 'default',
                                                        'pc_type': 'python',
                                                        'pc_python_type': __name__ + '.DGLaplacian',
                                                        'aux_ksp_type': 'preonly',
                                                        'aux_pc_type': 'jacobi'}}}}
    eq = a == L
    problem = LinearVariationalProblem(eq.lhs, eq.rhs, w)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    expected = {'nested':True,
                'preonly_A00':False, 'jacobi_A00':True,
                'schur_approx':True,
                'preonly_Shat':True, 'jacobi_Shat':True}
    builder = solver.snes.ksp.pc.getPythonContext().getSchurComplementBuilder()
    assert options_check(builder, expected), "Some solver options have not ended up in the PC as wanted."

    sigma_h, u_h = w.split()

    # (Non-hybrid) Need to slam it with preconditioning due to the
    # system's indefiniteness
    w2 = Function(W)
    solve(a == L, w2,
        solver_parameters={'mat_type': 'matfree',
                            'ksp_type': 'preonly',
                            'pc_type': 'python',
                            'pc_python_type': 'firedrake.HybridizationPC',
                            'hybridization': {'ksp_type': 'preonly',
                                            'pc_type': 'lu'}})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11

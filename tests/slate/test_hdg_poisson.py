import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS


def run_LDG_H_problem(r, degree, quads=False):
    """
    Solves the Dirichlet problem for the elliptic equation:

    -div(grad(u)) = f in [0, 1]^2, u = g on the domain boundary.

    The source function f and g are chosen such that the analytic
    solution is:

    u(x, y) = sin(x*pi)*sin(y*pi).

    This test uses an HDG discretization (specifically LDG-H by Cockburn)
    and Slate to perform the static condensation and local recovery.
    """

    # Set up problem domain
    mesh = UnitSquareMesh(2**r, 2**r, quadrilateral=quads)
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Set up function spaces
    U = VectorFunctionSpace(mesh, "DG", degree)
    V = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    # Mixed space and test/trial functions
    W = U * V * T
    s = Function(W).assign(0.0)
    q, u, uhat = split(s)
    v, w, mu = TestFunctions(W)

    # Analytical solutions for u and q
    V_a = FunctionSpace(mesh, "DG", degree + 3)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 3)

    u_a = Function(V_a)
    a_scalar = sin(pi*x[0])*sin(pi*x[1])
    u_a.interpolate(a_scalar)

    q_a = Function(U_a)
    a_flux = -grad(a_scalar)
    q_a.project(a_flux)

    Vh = FunctionSpace(mesh, "DG", degree + 3)
    f = Function(Vh).interpolate(-div(grad(a_scalar)))

    # Stability parameter
    tau = Constant(1)

    # Numerical flux
    qhat = q + tau*(u - uhat)*n

    # Formulate the LDG-H method in UFL
    a = ((inner(q, v) - inner(u, div(v)))*dx
         + inner(uhat('+'), jump(v, n=n))*dS
         + inner(uhat, dot(v, n))*ds
         - inner(grad(w), q)*dx
         + inner(jump(qhat, n=n), w('+'))*dS
         + inner(dot(qhat, n), w)*ds
         # Transmission condition
         + inner(jump(qhat, n=n), mu('+'))*dS)

    L = inner(f, w)*dx
    F = a - L
    params = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': '0, 1',
        'condensed_field': {
            'ksp_type': 'preonly',
            'pc_type': 'redundant',
            "redundant_pc_type": "lu",
            "redundant_pc_factor": DEFAULT_DIRECT_SOLVER_PARAMETERS,
        }
    }

    bcs = DirichletBC(W.sub(2), zero(), "on_boundary")
    problem = NonlinearVariationalProblem(F, s, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Computed flux, scalar, and trace
    q_h, u_h, uhat_h = s.subfunctions

    scalar_error = errornorm(a_scalar, u_h, norm_type="L2")
    flux_error = errornorm(a_flux, q_h, norm_type="L2")

    return scalar_error, flux_error


@pytest.mark.parallel
@pytest.mark.parametrize(('degree', 'quads', 'rate'),
                         [(1, False, 1.75),
                          (1, True, 1.75),
                          (2, False, 2.75),
                          (2, True, 2.75)])
def test_hdg_convergence(degree, quads, rate):
    import numpy as np
    diff = np.array([run_LDG_H_problem(r, degree, quads) for r in range(2, 5)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > rate).all()

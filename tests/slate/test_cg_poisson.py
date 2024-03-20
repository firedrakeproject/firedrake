import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER


def run_CG_problem(r, degree, quads=False):
    """
    Solves the Dirichlet problem for the elliptic equation:

    -div(grad(u)) = f in [0, 1]^2, u = g on the domain boundary.

    The source function f and g are chosen such that the analytic
    solution is:

    u(x, y) = sin(x*pi)*sin(y*pi).

    This test uses a CG discretization splitting interior and facet DOFs
    and Slate to perform the static condensation and local recovery.
    """

    # Set up problem domain
    mesh = UnitSquareMesh(2**r, 2**r, quadrilateral=quads)
    x = SpatialCoordinate(mesh)
    u_exact = sin(x[0]*pi)*sin(x[1]*pi)
    f = -div(grad(u_exact))

    # Set up function spaces
    e = FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degree)
    Z = FunctionSpace(mesh, MixedElement(RestrictedElement(e, "interior"), RestrictedElement(e, "facet")))
    z = Function(Z)
    u = sum(split(z))

    # Formulate the CG method in UFL
    U = (1/2)*inner(grad(u), grad(u))*dx - inner(u, f)*dx
    F = derivative(U, z, TestFunction(Z))

    params = {
        'snes_type': 'ksponly',
        'mat_type': 'matfree',
        'pmat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.SCPC',
        'pc_sc_eliminate_fields': '0',
        'condensed_field': {
            'ksp_type': 'preonly',
            'pc_type': 'redundant',
            "redundant_pc_type": "lu",
            "redundant_pc_factor_mat_solver_type": DEFAULT_DIRECT_SOLVER
        }
    }

    bcs = DirichletBC(Z.sub(1), zero(), "on_boundary")
    problem = NonlinearVariationalProblem(F, z, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    return norm(u_exact-u, norm_type="L2")


@pytest.mark.parallel
@pytest.mark.parametrize(('degree', 'quads', 'rate'),
                         [(3, False, 3.75),
                          (5, True, 5.75)])
def test_cg_convergence(degree, quads, rate):
    import numpy as np
    diff = np.array([run_CG_problem(r, degree, quads) for r in range(2, 5)])
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > rate).all()

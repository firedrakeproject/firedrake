import pytest
from firedrake import *


def run_CG_problem(r, degree, quads=False):
    """
    Solves the Dirichlet problem for the elliptic equation:

    -div(grad(u)) = f in [0, 1]^2, u = g on the domain boundary.

    The source function f and g are chosen such that the analytic
    solution is:

    u(x, y) = sin(x*pi)*sin(y*pi).

    This test uses a CG discretization splitting interior and facet DOFs
    and Slate to perform the static condensation and local recovery.
    This solver uses multigrid on a mesh hierarchy to test coarsening of
    Slate objects.
    """

    # Set up problem domain
    mesh = UnitSquareMesh(2, 2, quadrilateral=quads)
    mh = MeshHierarchy(mesh, r-1)
    mesh = mh[-1]
    x = SpatialCoordinate(mesh)
    u_exact = sin(x[0]*pi)*sin(x[1]*pi)
    f = -div(grad(u_exact))

    # Set up function spaces
    e = FiniteElement("Lagrange", cell=mesh.ufl_cell(), degree=degree)
    V = FunctionSpace(mesh, MixedElement(e["interior"], e["facet"]))
    uh = Function(V)

    # Formulate the CG method in UFL
    u = sum(TrialFunctions(V))
    v = sum(TestFunctions(V))
    a = inner(grad(v), grad(u)) * dx
    L = inner(v, f) * dx

    params = {
        "ksp_type": "preonly",
        "pc_type": "python",
        "mat_type": "matfree",
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0",
        "condensed_field": {
            "mat_type": "aij",
            "ksp_monitor": None,
            "ksp_type": "cg",
            "ksp_rtol": 1E-14,
            "ksp_atol": 0E-14,
            "ksp_norm_type": "natural",
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "chebyshev",
                "ksp_chebyshev_kind": "fourth",
                "ksp_chebyshev_esteig": "0,0.99,0,1.01",
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_construct_dim": 0,
                "pc_star_sub_sub_pc_type": "cholesky",
                "pc_star_sub_sub_pc_factor_mat_solver_type": "petsc"},
            "mg_coarse": {
                "ksp_type": "preonly",
                "pc_type": "redundant",
                "redundant_pc_type": "cholesky",
                "redundant_pc_factor_mat_solver_type": "mumps"}}}

    bcs = DirichletBC(V.sub(1), 0, "on_boundary")
    problem = LinearVariationalProblem(a, L, uh, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()
    ksp = solver.snes.ksp
    ksp = ksp.pc.getPythonContext().condensed_ksp
    its = ksp.getIterationNumber()
    error = norm(u_exact-sum(uh), norm_type="L2")
    return error, its


@pytest.mark.parallel
@pytest.mark.parametrize(('degree', 'quads', 'rate'),
                         [(3, False, 3.75),
                          (5, True, 5.75)])
def test_cg_convergence(degree, quads, rate):
    import numpy as np
    errors = []
    for r in range(2, 5):
        error, its = run_CG_problem(r, degree, quads)
        errors.append(error)
        assert its <= 21

    diff = np.array(errors)
    conv = np.log2(diff[:-1] / diff[1:])
    assert (np.array(conv) > rate).all()

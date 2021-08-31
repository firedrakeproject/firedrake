import pytest
from firedrake import *


@pytest.fixture(params=[2, 3],
                ids=["Rectangle", "Box"])
def mesh(request):
    distribution = {"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)}
    if request.param == 2:
        return RectangleMesh(10, 20, 2, 3, quadrilateral=True, distribution_parameters=distribution)
    if request.param == 3:
        base = RectangleMesh(5, 3, 1, 2, quadrilateral=True, distribution_parameters=distribution)
        return ExtrudedMesh(base, 5, layer_height=3/5)


@pytest.fixture
def expected(mesh):
    if mesh.geometric_dimension() == 2:
        return [5, 5, 5]
    elif mesh.geometric_dimension() == 3:
        return [6, 6, 6]


@pytest.mark.skipcomplex
def test_p_independence(mesh, expected):
    nits = []
    for p in range(2, 5):
        V = FunctionSpace(mesh, "Lagrange", p)

        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u), grad(v))*dx

        L = inner(Constant(1), v)*dx

        subs = ("on_boundary",)
        if mesh.topological_dimension() == 3:
            subs += ("top", "bottom")
        bcs = [DirichletBC(V, zero(V.ufl_element().value_shape()), sub) for sub in subs]

        uh = Function(V)
        problem = LinearVariationalProblem(a, L, uh, bcs=bcs)

        solver = LinearVariationalSolver(problem, solver_parameters={
            "mat_type": "matfree",
            "ksp_type": "gmres",
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.P1PC",
            "pmg_mg_levels_ksp_type": "chebyshev",
            "pmg_mg_levels_ksp_norm_type": "unpreconditioned",
            "pmg_mg_levels_ksp_monitor_true_residual": None,
            "pmg_mg_levels_pc_type": "python",
            "pmg_mg_levels_pc_python_type": "firedrake.FDMPC",
            "pmg_mg_levels_fdm": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.ASMStarPC",
                "pc_star_backend": "petscasm",
                "pc_star_sub_sub_ksp_type": "preonly",
                "pc_star_sub_sub_pc_type": "cholesky",
            },
            "pmg_mg_coarse": {
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "cholesky",
            },
            "ksp_monitor": None})

        solver.solve()

        nits.append(solver.snes.ksp.getIterationNumber())
    assert (nits == expected)

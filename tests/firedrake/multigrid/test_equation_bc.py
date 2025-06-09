import pytest
from firedrake import *


def test_poisson_NLVP():
    mesh = UnitIntervalMesh(10)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    mesh = mesh_hierarchy[-1]

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, = SpatialCoordinate(mesh)
    u_exact = Function(V).interpolate(x**2)

    F = inner(grad(u - u_exact), grad(v))*dx
    bcs = [EquationBC(inner(u - u_exact, v) * ds == 0, u, "on_boundary", V=V)]
    NLVP = NonlinearVariationalProblem(F, u, bcs=bcs)

    sp = {"ksp_rtol": 1E-10, "pc_type": "mg"}
    NLVS = NonlinearVariationalSolver(NLVP, solver_parameters=sp)
    NLVS.solve()

    assert errornorm(u_exact, u) < 1e-9
    assert NLVS.snes.getLinearSolveIterations() <= 9


def test_poisson_LVP():
    mesh = UnitIntervalMesh(10)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    mesh = mesh_hierarchy[-1]

    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    u_ = Function(V)
    v = TestFunction(V)

    x, = SpatialCoordinate(mesh)
    u_exact = Function(V).interpolate(x**2)

    a = inner(grad(u), grad(v))*dx
    L = action(a, u_exact)
    bcs = [EquationBC(inner(u, v) * ds == inner(u_exact, v) * ds, u_, "on_boundary", V=V)]
    LVP = LinearVariationalProblem(a, L, u_, bcs=bcs)

    sp = {"ksp_rtol": 1E-10, "pc_type": "mg"}
    LVS = LinearVariationalSolver(LVP, solver_parameters=sp)
    LVS.solve()

    assert errornorm(u_exact, u_) < 1e-9
    assert LVS.snes.getLinearSolveIterations() <= 9


@pytest.mark.parametrize("dim", (2, 3))
def test_nested_equation_bc(dim):
    Nbase = 2
    refine = 2
    if dim == 2:
        mesh = UnitSquareMesh(Nbase, Nbase)
    elif dim == 3:
        mesh = UnitCubeMesh(Nbase, Nbase, Nbase)

    mesh_hierarchy = MeshHierarchy(mesh, refine)
    mesh = mesh_hierarchy[-1]

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x = SpatialCoordinate(mesh)
    u_exact = Function(V).interpolate(x[0]*x[1])

    if dim == 2:
        ridges = [(1, 3), (1, 4), (2, 3), (2, 4)]
        num_ridge_dofs = 4
    elif dim == 3:
        ridges = [(1, 3), (1, 4), (1, 5), (1, 6),
                  (2, 3), (2, 4), (2, 5), (2, 6),
                  (3, 5), (3, 6), (4, 5), (4, 6)]
        num_ridge_dofs = 8 + 12*(Nbase*2**refine-1)

    F = inner(grad(u - u_exact), grad(v))*dx

    u_bc = Function(u_exact)
    n = FacetNormal(mesh)
    Fs = inner(dot(grad(u - u_bc), n), dot(grad(v), n)) * ds

    bcs_ridges = [DirichletBC(V, u_bc, ridges)]
    bcs_facets = [EquationBC(Fs == 0, u, "on_boundary", V=V, bcs=bcs_ridges)]
    problem = NonlinearVariationalProblem(F, u, bcs=bcs_ridges)

    nodes = set(sum((tuple(e.nodes) for e in problem.dirichlet_bcs()), ()))
    assert len(nodes) == num_ridge_dofs

    sp = {
        "ksp_type": "fgmres",
        "ksp_rtol": 1e-10,
        "ksp_monitor": None,
        "pc_type": "mg",
        "mg_levels": {
            "ksp_max_it": "2",
            "ksp_convergence_test": "skip",
            "ksp_type": "gmres",
            "pc_type": "jacobi",
        },
        "mg_coarse": {
            "ksp_type": "preonly",
            "pc_type": "lu",
        }
    }

    solver = NonlinearVariationalSolver(problem, solver_parameters=sp)
    solver.solve()
    assert errornorm(u_exact, u) < 1e-9
    assert solver.snes.getLinearSolveIterations() <= 12

    u.assign(0)
    solver = NonlinearVariationalSolver(problem, solver_parameters=sp, pre_apply_bcs=False)
    solver.solve()
    assert errornorm(u_exact, u) < 1e-9
    assert solver.snes.getLinearSolveIterations() <= 12

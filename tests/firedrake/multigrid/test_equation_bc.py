from firedrake import *
import pytest


def test_poisson():
    mesh = UnitIntervalMesh(10)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    mesh = mesh_hierarchy[-1]

    x = SpatialCoordinate(mesh)
    u_exact = x[0]*(x[0]-1)
    f = -div(grad(u_exact))

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v))*dx(degree=0) - inner(f, v)*dx
    bcs = [EquationBC(inner(u, v) * ds == 0, u, "on_boundary", V=V)]
    NLVP = NonlinearVariationalProblem(F, u, bcs=bcs)

    sp = {"pc_type": "mg"}

    NLVS = NonlinearVariationalSolver(NLVP, solver_parameters=sp)
    NLVS.solve()

    assert errornorm(u_exact, u) < 5e-4


def test_nested_equation_bc():
    mesh = UnitCubeMesh(2,2,2)
    mesh_hierarchy = MeshHierarchy(mesh, 2)
    mesh = mesh_hierarchy[-1]

    (x,y,z) = SpatialCoordinate(mesh)
    u_exact = x*y*z
    f = -div(grad(u_exact))

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v))*dx(degree=0) - inner(f, v)*dx
    bcs_edges = [DirichletBC(V, u_exact, "on_boundary")]
    bcs_facets = [EquationBC(inner(u - u_exact, v) * ds == 0, u, "on_boundary", V=V, bcs=bcs_edges)]
    NLVP = NonlinearVariationalProblem(F, u, bcs=bcs_facets)

    sp = {
            "ksp_rtol": 1e-10,
            "ksp_monitor": None,
            "pc_type": "mg",
            "mg_levels": {
                "ksp_max_it": "1",
                "ksp_convergence_test": "skip",
                "ksp_type": "chebyshev",
                "pc_type": "jacobi",
                },
            }

    NLVS = NonlinearVariationalSolver(NLVP, solver_parameters=sp)
    NLVS.solve()

    assert errornorm(u_exact, u) < 5e-3
    assert NLVS.snes.getLinearSolveIterations() <= 13

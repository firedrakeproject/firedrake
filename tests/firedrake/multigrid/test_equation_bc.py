from firedrake import *
import pytest


def test_poisson():
    mesh = UnitIntervalMesh(10)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    mesh = mesh_hierarchy[-1]

    x = SpatialCoordinate(mesh)
    u_exact = sin(pi*x[0])
    f = -div(grad(u_exact))

    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = (inner(grad(u), grad(v)) - inner(f, v)) * dx
    bcs = [EquationBC(inner(u, v) * ds == 0, u, "on_boundary", V=V)]
    NLVP = NonlinearVariationalProblem(F, u, bcs=bcs)

    sp = {"pc_type": "mg",}

    NLVS = NonlinearVariationalSolver(NLVP, solver_parameters=sp)
    NLVS.solve()

    assert sqrt(assemble(inner(u - u_exact, u - u_exact) * dx)) < 1e-2


def test_nested_equation_bc():
    mesh = UnitCubeMesh(10,10,10)
    mesh_hierarchy = MeshHierarchy(mesh, 1)
    mesh = mesh_hierarchy[-1]

    (x,y,z) = SpatialCoordinate(mesh)
    u_exact = sin(pi*x)*sin(pi*y)*sin(pi*z)
    f = -div(grad(u_exact))

    V = FunctionSpace(mesh, "CG", 1)
    print(f"# DoFs = {V.dim()}")
    u = Function(V)
    v = TestFunction(V)

    F = (inner(grad(u), grad(v)) - inner(f, v)) * dx
    bcs_vertices = [EquationBC(inner(u, v) * ds == 0, u, "on_boundary", V=V)]
    bcs_edges = [EquationBC(inner(u, v) * ds == 0, u, "on_boundary", V=V, bcs=bcs_vertices)]
    bcs_facets = [EquationBC(inner(u, v) * ds == 0, u, "on_boundary", V=V, bcs=bcs_edges)]
    NLVP = NonlinearVariationalProblem(F, u, bcs=bcs_facets)

    sp = {
            "pc_type": "mg",
            "mg_levels": {
                "ksp_type": "richardson",
                "pc_type": "ilu",
                },
            }

    NLVS = NonlinearVariationalSolver(NLVP, solver_parameters=sp)
    NLVS.solve()

    assert sqrt(assemble(inner(u - u_exact, u - u_exact) * dx)) < 1e-2

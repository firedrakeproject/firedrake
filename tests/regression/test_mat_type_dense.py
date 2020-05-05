from firedrake import *
from numpy import allclose


def solve_poisson(mat_type):
    mesh = UnitSquareMesh(3, 3, quadrilateral=True)
    V = VectorFunctionSpace(mesh, "CG", 2)

    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v))*dx - inner(Constant((1, 2)), v)*dx
    bcs = DirichletBC(V, Constant((0, 0)), "on_boundary")

    sp = {"snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "mat_type": mat_type}

    solve(F == 0, u, bcs, solver_parameters=sp)
    return u


def test_mat_type_dense():
    u_aij = solve_poisson("aij")
    u_dense = solve_poisson("dense")

    assert allclose(u_aij.dat.data_ro, u_dense.dat.data_ro)

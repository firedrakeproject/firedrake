from firedrake import *
from numpy import allclose
import pytest


def solve_poisson(mat_type, kind):
    mesh = UnitSquareMesh(3, 3, quadrilateral=True)
    if kind == "vector":
        V = VectorFunctionSpace(mesh, "CG", 2)
    elif kind == "mixed":
        W = FunctionSpace(mesh, "CG", 2)
        V = MixedFunctionSpace([W, W])

    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v))*dx - inner(Constant((1, 2)), v)*dx

    if kind == "vector":
        bcs = DirichletBC(V, Constant((0, 0)), "on_boundary")
    elif kind == "mixed":
        bcs = [DirichletBC(V.sub(i), Constant(0), "on_boundary") for i in range(2)]

    sp = {"snes_type": "ksponly",
          "ksp_type": "preonly",
          "pc_type": "lu",
          "mat_type": mat_type}

    solve(F == 0, u, bcs, solver_parameters=sp)
    return u


@pytest.mark.parametrize(('kind',), [('vector',), ('mixed',)])
def test_compare_mat_type_dense(kind):
    u_aij = solve_poisson("aij", kind)
    u_dense = solve_poisson("dense", kind)

    assert allclose(u_aij.dat.data_ro, u_dense.dat.data_ro)

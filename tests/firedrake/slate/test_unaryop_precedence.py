from firedrake import *


def test_unary_minus():
    mesh = UnitSquareMesh(1, 1)

    V = FunctionSpace(mesh, "CG", 1)

    uh = Function(V)

    v = TestFunction(V)

    u = TrialFunction(V)

    A = Tensor(inner(u, v)*dx)

    B = Tensor(inner(uh, v)*dx)

    uh.assign(1)

    expr = action(A, uh) - B

    assembled_expr = assemble(expr)
    assert assembled_expr.dat.norm < 1e-9

    assembled_expr = assemble(-expr)
    assert assembled_expr.dat.norm < 1e-9

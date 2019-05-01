import pytest
from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_pointwise_operator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)
    v = Function(V)
    u = Function(V)

    x, y = SpatialCoordinate(mesh)
    u.interpolate(cos(x))
    v.interpolate(sin(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: x*y)
    p2 = p(u, v, eval_space=P)
    a2 = p2*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0)
    assert p2.ufl_shape == ()
    assert p2.operator_data(u, v) == u*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-3  # Not evaluate on the same space whence the lack of precision

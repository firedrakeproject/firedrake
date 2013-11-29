import pytest

from firedrake import *


def test_facet_normal():

    m = UnitSquareMesh(2, 2)

    V = VectorFunctionSpace(m, 'CG', 1)

    y_hat = Function(V)

    y_hat.interpolate(Expression(('0', '1')))

    x_hat = Function(V)

    x_hat.interpolate(Expression(('1', '0')))

    n = FacetNormal(m.ufl_cell())

    y1 = assemble(dot(y_hat, n)*ds(1))   # -1
    y2 = assemble(dot(y_hat, n)*ds(2))   # 1
    y3 = assemble(dot(y_hat, n)*ds(3))   # 0
    y4 = assemble(dot(y_hat, n)*ds(4))   # 0

    x1 = assemble(dot(x_hat, n)*ds(1))   # 0
    x2 = assemble(dot(x_hat, n)*ds(2))   # 0
    x3 = assemble(dot(x_hat, n)*ds(3))   # 1
    x4 = assemble(dot(x_hat, n)*ds(4))   # -1

    assert y1 == -1.0
    assert y2 == 1.0
    assert y3 == 0.0
    assert y4 == 0.0

    assert x1 == 0.0
    assert x2 == 0.0
    assert x3 == 1.0
    assert x4 == -1.0


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

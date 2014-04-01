"""Testing assembly of scalars on facets of extruded meshes in 2D"""
import pytest

from firedrake import *
from tests.common import *


@pytest.fixture(scope='module', params=[1, 2])
def f(request):
    mesh = extmesh_2D(4, 4)
    fspace = FunctionSpace(mesh, "CG", request.param)
    return Function(fspace)


@pytest.fixture(scope='module')
def RT2():
    mesh = extmesh_2D(4, 4)
    U0 = FiniteElement("CG", "interval", 2)
    U1 = FiniteElement("DG", "interval", 1)
    W1 = HDiv(OuterProductElement(U1, U0)) + HDiv(OuterProductElement(U0, U1))
    return FunctionSpace(mesh, W1)


def test_scalar_area(f):
    f.assign(1)
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 1.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 2.0) < 1e-7
    assert abs(assemble(f*ds_v) - 2.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 3.0) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 3.0) < 1e-7
    assert abs(assemble(f('+')*dS_v) - 3.0) < 1e-7
    assert abs(assemble(f('-')*dS_v) - 3.0) < 1e-7


def test_scalar_expression(f):
    f.interpolate(Expression("x[1]"))
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 0.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 1.0) < 1e-7
    assert abs(assemble(f*ds_v) - 1.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('+')*dS_v) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_v) - 1.5) < 1e-7


def test_hdiv_area(RT2):
    f = project(Expression(("0.8", "0.6")), RT2)
    assert abs(assemble(dot(f, f)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_b) - 1.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_tb) - 2.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_v) - 2.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('+'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('-'), f('-'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('-'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('+'))*dS_v) - 3.0) < 1e-7
    assert abs(assemble(dot(f('-'), f('-'))*dS_v) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('-'))*dS_v) - 3.0) < 1e-7


def test_exterior_horizontal_normals(RT2):
    n = FacetNormal(RT2.mesh())
    f = project(Expression(("1.0", "0.0")), RT2)
    assert abs(assemble(dot(f, n)*ds_t) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - 0.0) < 1e-7
    f = project(Expression(("0.0", "1.0")), RT2)
    assert abs(assemble(dot(f, n)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - (-1.0)) < 1e-7


def test_exterior_vertical_normals(RT2):
    n = FacetNormal(RT2.mesh())
    f = project(Expression(("1.0", "0.0")), RT2)
    assert abs(assemble(dot(f, n)*ds_v(1)) - (-1.0)) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 1.0) < 1e-7
    f = project(Expression(("0.0", "1.0")), RT2)
    assert abs(assemble(dot(f, n)*ds_v(1)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 0.0) < 1e-7


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

"""Testing assembly of scalars on facets of extruded meshes in 2D"""
import pytest

from firedrake import *
from common import *


@pytest.mark.xfail(reason="waiting for extruded facet changes")
@pytest.mark.parametrize('degree', [1, 2])
def test_scalar_area(degree):
    mesh = extmesh_2D(4, 4)
    fspace = FunctionSpace(mesh, "CG", degree)
    f = Function(fspace)
    f.assign(1)
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 1.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 2.0) < 1e-7
    assert abs(assemble(f*ds_v) - 2.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 3.0) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 3.0) < 1e-7
    assert abs(assemble(f('+')*dS_v) - 3.0) < 1e-7
    assert abs(assemble(f('-')*dS_v) - 3.0) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
@pytest.mark.parametrize('degree', [1, 2])
def test_scalar_expression(degree):
    mesh = extmesh_2D(4, 4)
    fspace = FunctionSpace(mesh, "CG", degree)
    f = Function(fspace)
    f.interpolate(Expression("x[1]"))
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 0.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 1.0) < 1e-7
    assert abs(assemble(f*ds_v) - 1.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('+')*dS_v) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_v) - 1.5) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_hcurl_area():
    mesh = extmesh_2D(4, 4)
    U0 = FiniteElement("CG", "interval", 1)
    U1 = FiniteElement("DG", "interval", 0)
    W1 = HCurl(OuterProductElement(U1, U0)) + HCurl(OuterProductElement(U0, U1))
    fspace = FunctionSpace(mesh, W1)
    f = project(Expression(("0.8", "0.6")), fspace)
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


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_hdiv_area():
    mesh = extmesh_2D(4, 4)
    U0 = FiniteElement("CG", "interval", 1)
    U1 = FiniteElement("DG", "interval", 0)
    W1 = HDiv(OuterProductElement(U1, U0)) + HDiv(OuterProductElement(U0, U1))
    fspace = FunctionSpace(mesh, W1)
    f = project(Expression(("0.8", "0.6")), fspace)
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


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_exterior_horizontal_normals():
    mesh = extmesh_2D(4, 4)
    n = FacetNormal(mesh)
    U0 = FiniteElement("CG", "interval", 1)
    U1 = FiniteElement("DG", "interval", 0)
    W1 = HDiv(OuterProductElement(U1, U0)) + HDiv(OuterProductElement(U0, U1))
    fspace = FunctionSpace(mesh, W1)
    f = project(Expression(("1.0", "0.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_t) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - 0.0) < 1e-7
    f = project(Expression(("0.0", "1.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - (-1.0)) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_exterior_vertical_normals():
    mesh = extmesh_2D(4, 4)
    n = FacetNormal(mesh)
    U0 = FiniteElement("CG", "interval", 1)
    U1 = FiniteElement("DG", "interval", 0)
    W1 = HDiv(OuterProductElement(U1, U0)) + HDiv(OuterProductElement(U0, U1))
    fspace = FunctionSpace(mesh, W1)
    f = project(Expression(("1.0", "0.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_v(1)) - (-1.0)) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 1.0) < 1e-7
    f = project(Expression(("0.0", "1.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_v(1)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 0.0) < 1e-7


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

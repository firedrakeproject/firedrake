"""Testing assembly of scalars on facets of extruded meshes in 3D"""
import pytest

from firedrake import *
from tests.common import *


@pytest.mark.xfail(reason="waiting for extruded facet changes")
@pytest.mark.parametrize('degree', [1, 2])
def test_scalar_area(degree):
    mesh = extmesh(4, 4, 4)
    fspace = FunctionSpace(mesh, "CG", degree)
    f = Function(fspace)
    f.assign(1)
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 1.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 2.0) < 1e-7
    assert abs(assemble(f*ds_v) - 4.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 3.0) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 3.0) < 1e-7
    assert abs(assemble(f('+')*dS_v) - (6.0 + 4*sqrt(2))) < 1e-7
    assert abs(assemble(f('-')*dS_v) - (6.0 + 4*sqrt(2))) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
@pytest.mark.parametrize('degree', [1, 2])
def test_scalar_expression(degree):
    mesh = extmesh(4, 4, 4)
    fspace = FunctionSpace(mesh, "CG", degree)
    f = Function(fspace)
    f.interpolate(Expression("x[2]"))
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 0.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 1.0) < 1e-7
    assert abs(assemble(f*ds_v) - 2.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('+')*dS_v) - 0.5*(6.0 + 4*sqrt(2))) < 1e-7
    assert abs(assemble(f('-')*dS_v) - 0.5*(6.0 + 4*sqrt(2))) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_hcurl_area():
    mesh = extmesh(4, 4, 4)
    U0 = FiniteElement("CG", "triangle", 1)
    U1 = FiniteElement("RT", "triangle", 1)
    V0 = FiniteElement("CG", "interval", 1)
    V1 = FiniteElement("DG", "interval", 0)
    W1 = HCurl(OuterProductElement(U1, V0)) + HCurl(OuterProductElement(U0, V1))
    fspace = FunctionSpace(mesh, W1)
    f = project(Expression(("0.0", "0.8", "0.6")), fspace)
    assert abs(assemble(dot(f, f)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_b) - 1.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_tb) - 2.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_v) - 4.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('+'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('-'), f('-'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('-'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('+'))*dS_v) - (6.0 + 4*sqrt(2))) < 2e-7
    assert abs(assemble(dot(f('-'), f('-'))*dS_v) - (6.0 + 4*sqrt(2))) < 2e-7
    assert abs(assemble(dot(f('+'), f('-'))*dS_v) - (6.0 + 4*sqrt(2))) < 2e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_hdiv_area():
    mesh = extmesh(4, 4, 4)
    U1 = FiniteElement("RT", "triangle", 1)
    U2 = FiniteElement("DG", "triangle", 0)
    V0 = FiniteElement("CG", "interval", 1)
    V1 = FiniteElement("DG", "interval", 0)
    W2 = HDiv(OuterProductElement(U1, V1)) + HDiv(OuterProductElement(U2, V0))
    fspace = FunctionSpace(mesh, W2)
    f = project(Expression(("0.0", "0.8", "0.6")), fspace)
    assert abs(assemble(dot(f, f)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_b) - 1.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_tb) - 2.0) < 1e-7
    assert abs(assemble(dot(f, f)*ds_v) - 4.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('+'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('-'), f('-'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('-'))*dS_h) - 3.0) < 1e-7
    assert abs(assemble(dot(f('+'), f('+'))*dS_v) - (6.0 + 4*sqrt(2))) < 1e-7
    assert abs(assemble(dot(f('-'), f('-'))*dS_v) - (6.0 + 4*sqrt(2))) < 1e-7
    assert abs(assemble(dot(f('+'), f('-'))*dS_v) - (6.0 + 4*sqrt(2))) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_exterior_horizontal_normals():
    mesh = extmesh(4, 4, 4)
    n = FacetNormal(mesh)
    U1 = FiniteElement("RT", "triangle", 1)
    U2 = FiniteElement("DG", "triangle", 0)
    V0 = FiniteElement("CG", "interval", 1)
    V1 = FiniteElement("DG", "interval", 0)
    W2 = HDiv(OuterProductElement(U1, V1)) + HDiv(OuterProductElement(U2, V0))
    fspace = FunctionSpace(mesh, W2)
    f = project(Expression(("1.0", "0.0", "0.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_t) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - 0.0) < 1e-7
    f = project(Expression(("0.0", "0.0", "1.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - (-1.0)) < 1e-7


@pytest.mark.xfail(reason="waiting for extruded facet changes")
def test_exterior_vertical_normals():
    mesh = extmesh(4, 4, 4)
    n = FacetNormal(mesh)
    U1 = FiniteElement("RT", "triangle", 1)
    U2 = FiniteElement("DG", "triangle", 0)
    V0 = FiniteElement("CG", "interval", 1)
    V1 = FiniteElement("DG", "interval", 0)
    W2 = HDiv(OuterProductElement(U1, V1)) + HDiv(OuterProductElement(U2, V0))
    fspace = FunctionSpace(mesh, W2)
    f = project(Expression(("1.0", "0.0", "0.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_v(1)) - (-1.0)) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 1.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(3)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(4)) - 0.0) < 1e-7
    f = project(Expression(("0.0", "1.0", "0.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_v(1)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(3)) - (-1.0)) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(4)) - 1.0) < 1e-7
    f = project(Expression(("0.0", "0.0", "1.0")), fspace)
    assert abs(assemble(dot(f, n)*ds_v(1)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(3)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(4)) - 0.0) < 1e-7


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

"""Testing assembly of scalars on facets of extruded meshes in 2D"""
import pytest

from firedrake import *


@pytest.fixture(scope="module")
def mesh(extmesh_2D):
    return extmesh_2D(4, 4)


@pytest.fixture(scope="module", params=[1, 2])
def f(mesh, request):
    fspace = FunctionSpace(mesh, "CG", request.param)
    return Function(fspace)


@pytest.fixture(scope="module")
def RT2(mesh):
    U0 = FiniteElement("CG", "interval", 2)
    U1 = FiniteElement("DG", "interval", 1)
    W1 = HDiv(TensorProductElement(U1, U0)) + HDiv(TensorProductElement(U0, U1))
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
    xs = SpatialCoordinate(f.function_space().mesh())
    f.interpolate(xs[1])
    assert abs(assemble(f*ds_t) - 1.0) < 1e-7
    assert abs(assemble(f*ds_b) - 0.0) < 1e-7
    assert abs(assemble(f*ds_tb) - 1.0) < 1e-7
    assert abs(assemble(f*ds_v) - 1.0) < 1e-7
    assert abs(assemble(f('+')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_h) - 1.5) < 1e-7
    assert abs(assemble(f('+')*dS_v) - 1.5) < 1e-7
    assert abs(assemble(f('-')*dS_v) - 1.5) < 1e-7


def test_hdiv_area(RT2):
    f = project(as_vector([0.8, 0.6]), RT2)
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
    f = project(as_vector([1.0, 0.0]), RT2)
    assert abs(assemble(dot(f, n)*ds_t) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - 0.0) < 1e-7
    f = project(as_vector([0.0, 1.0]), RT2)
    assert abs(assemble(dot(f, n)*ds_t) - 1.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_b) - (-1.0)) < 1e-7


def test_exterior_vertical_normals(RT2):
    n = FacetNormal(RT2.mesh())
    f = project(as_vector([1.0, 0.0]), RT2)
    assert abs(assemble(dot(f, n)*ds_v(1)) - (-1.0)) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 1.0) < 1e-7
    f = project(as_vector([0.0, 1.0]), RT2)
    assert abs(assemble(dot(f, n)*ds_v(1)) - 0.0) < 1e-7
    assert abs(assemble(dot(f, n)*ds_v(2)) - 0.0) < 1e-7

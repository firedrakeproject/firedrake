import pytest
import numpy as np
from firedrake import *
from firedrake.__future__ import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return ExtrudedMesh(m, layers=4, layer_height=0.25)


def test_horiz_facet_interior_jump(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = project(as_vector([z, y, x]), DG)
    form = jump(f[2]*f[1]*conj(u), n=n)*dS_h

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_horiz_facet_interior_avg(mesh):
    DG = FunctionSpace(mesh, "DG", 1)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = assemble(interpolate(x + 2*y + 4*z, DG))
    form = avg(inner(f, u))*dS_h

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_vert_facet_interior_jump(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = project(as_vector([z, y, x]), DG)
    form = jump(f[0]*conj(u), n=n)*dS_v

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_vert_facet_interior_avg(mesh):
    DG = FunctionSpace(mesh, "DG", 1)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = assemble(interpolate(x + 2*y + 4*z, DG))
    form = avg(inner(f, u))*dS_v

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_top_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = project(as_vector([z, y, x]), DG)

    form = inner(n, f[2]*f[1]*u)*ds_t
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_bottom_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = project(as_vector([z, y, x]), DG)

    form = inner(n, f[2]*f[1]*u)*ds_b
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_vert_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = project(as_vector([z, y, x]), DG)

    form = inner(n, f[0]*u)*ds_v
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_total_interior_avg(mesh):
    DG = FunctionSpace(mesh, "DG", 1)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = assemble(interpolate(x + 2*y + 4*z, DG))
    form = avg(inner(f, u))*(dS_v + dS_h)

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_total_facet(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh)
    f = project(as_vector([z, y, x]), DG)

    top = inner(n, f[0]*f[1]*u)*ds_t
    bottom = inner(n, f[2]*f[1]*u)*ds_b
    horiz = jump(f[0]*conj(u), n=n)*dS_h
    vert = jump(f[2]*conj(u), n=n)*dS_v
    form = top + bottom + horiz + vert

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_no_horiz_jump():
    mesh = UnitTriangleMesh()
    mesh = ExtrudedMesh(mesh, 1)
    DG = FunctionSpace(mesh, "DG", 0)
    u = TestFunction(DG)

    _, _, z = SpatialCoordinate(mesh)
    form = jump(z*u)*dS_h

    assert np.allclose(assemble(Tensor(form)).dat.data, assemble(form).dat.data)

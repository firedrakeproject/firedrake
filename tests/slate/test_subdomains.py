import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh_2d(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return m


@pytest.fixture(scope='module')
def mesh_3d(request):
    m = UnitCubeMesh(2, 2, 2)
    return m


@pytest.fixture(scope='module', params=[False, True])
def mesh_extr(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return ExtrudedMesh(m, layers=4, layer_height=0.25)


@pytest.mark.parametrize("subdomain", (1, 2, 3, 4))
def test_2d_facet_subdomains(mesh_2d, subdomain):
    DG = VectorFunctionSpace(mesh_2d, "DG", 1)
    n = FacetNormal(mesh_2d)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh_2d)
    f = project(as_vector([x, y]), DG)

    form = inner(n, f[0]*f[1]*u)*ds(subdomain)

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


@pytest.mark.parametrize("subdomain", (1, 2, 3, 4, 5, 6))
def test_3d_facet_subdomains(mesh_3d, subdomain):
    DG = VectorFunctionSpace(mesh_3d, "DG", 1)
    n = FacetNormal(mesh_3d)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh_3d)
    f = project(as_vector([x, y, z]), DG)

    form = inner(n, f[0]*f[1]*f[2]*u)*ds(subdomain)

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


@pytest.mark.parametrize("subdomain", (1, 2, 3, 4))
def test_extr_vert_facet_subdomains(mesh_extr, subdomain):
    DG = VectorFunctionSpace(mesh_extr, "DG", 1)
    n = FacetNormal(mesh_extr)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh_extr)
    f = project(as_vector([z, y, x]), DG)

    form = inner(n, f[0]*f[1]*f[2]*u)*ds_v(subdomain)
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_multiple_subdomains_2d(mesh_2d):
    DG = VectorFunctionSpace(mesh_2d, "DG", 1)
    n = FacetNormal(mesh_2d)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh_2d)
    f = project(as_vector([x, y]), DG)

    ds_sd = ds(1) + ds(2) + ds(3) + ds(4)
    form = inner(n, f[0]*f[1]*u)*ds_sd

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_multiple_subdomains_3d(mesh_3d):
    DG = VectorFunctionSpace(mesh_3d, "DG", 1)
    n = FacetNormal(mesh_3d)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh_3d)
    f = project(as_vector([x, y, z]), DG)

    ds_sd = ds(1)
    for i in range(2, 7):
        ds_sd += ds(i)
    form = inner(n, f[0]*f[1]*f[2]*u)*ds_sd

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_multiple_subdomains_extr(mesh_extr):
    DG = VectorFunctionSpace(mesh_extr, "DG", 1)
    n = FacetNormal(mesh_extr)
    u = TestFunction(DG)

    x, y, z = SpatialCoordinate(mesh_extr)
    f = project(as_vector([z, y, x]), DG)

    ds_vsd = ds_v(1) + ds_v(2) + ds_v(3) + ds_v(4)
    form = inner(n, f[0]*f[1]*f[2]*u)*ds_vsd

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)

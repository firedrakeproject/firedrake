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


@pytest.mark.parametrize("subdomain", (1, 2, 3, 4))
def test_2d_facet_subdomains(mesh_2d, subdomain):
    DG = VectorFunctionSpace(mesh_2d, "DG", 1)
    n = FacetNormal(mesh_2d)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh_2d)
    f = project(as_vector([x, y]), DG)

    form = dot(f[0]*f[1]*u, n)*ds(subdomain)

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

    form = dot(f[0]*f[1]*f[2]*u, n)*ds(subdomain)

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)

import pytest
import numpy as np
from firedrake import *
from firedrake.__future__ import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return m


def test_facet_interior_jump(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh)
    f = project(as_vector([x, y]), DG)

    form = jump(f[0]*f[1]*conj(u), n=n)*dS

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_facet_interior_avg(mesh):
    DG = FunctionSpace(mesh, "DG", 1)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh)
    f = assemble(interpolate(x + y, DG))

    form = avg(inner(f, u))*dS

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)


def test_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    x, y = SpatialCoordinate(mesh)
    f = project(as_vector([x, y]), DG)

    form = inner(n, f[0]*f[1]*u)*ds

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-14)

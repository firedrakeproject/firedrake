from __future__ import absolute_import, print_function, division
import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return ExtrudedMesh(m, layers=4, layer_height=0.25)


def test_horiz_facet_interior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[2]", "x[1]", "x[0]")), DG)

    A = assemble(Tensor(dot(f[2]*f[1]*u, n)*dS_h)).dat.data
    ref = assemble(jump(f[2]*f[1]*u, n=n)*dS_h).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_vert_facet_interior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[2]", "x[1]", "x[0]")), DG)

    A = assemble(Tensor(dot(f[0]*u, n)*dS_v)).dat.data
    ref = assemble(jump(f[0]*u, n=n)*dS_v).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_top_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[2]", "x[1]", "x[0]")), DG)

    form = dot(f[2]*f[1]*u, n)*ds_t
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_bottom_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[2]", "x[1]", "x[0]")), DG)

    form = dot(f[2]*f[1]*u, n)*ds_b
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_vert_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[2]", "x[1]", "x[0]")), DG)

    form = dot(f[0]*u, n)*ds_v
    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

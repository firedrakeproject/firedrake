from __future__ import absolute_import, print_function, division
import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return m


def test_facet_interior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[0]", "x[1]")), DG)

    form = dot(f[0]*f[1]*u, n)*dS

    A = assemble(Tensor(form)).dat.data
    ref = assemble(jump(f[0]*f[1]*u, n=n)*dS).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


def test_facet_exterior(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)

    f = project(Expression(("x[0]", "x[1]")), DG)

    form = dot(f[0]*f[1]*u, n)*ds

    A = assemble(Tensor(form)).dat.data
    ref = assemble(form).dat.data

    assert np.allclose(A, ref, rtol=1e-8)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[False, True])
def mesh(request):
    m = UnitSquareMesh(2, 2, quadrilateral=request.param)
    return ExtrudedMesh(m, layers=4, layer_height=0.25)


def test_scalar_field_facet_extr(mesh):
    DG = VectorFunctionSpace(mesh, "DG", 1)
    n = FacetNormal(mesh)
    u = TestFunction(DG)
    A = assemble(Tensor(dot(u, n)*dS_h)).dat.data
    ref = assemble(jump(u, n=n)*dS_h + dot(u, n)*ds_t + dot(u, n)*ds_b).dat.data

    assert np.allclose(A, ref, rtol=1e-8)

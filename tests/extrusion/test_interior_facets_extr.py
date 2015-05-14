import numpy as np
import pytest

from firedrake import *


@pytest.mark.xfail
def test_interior_facet_vfs_extr_horiz_2d():
    m = UnitIntervalMesh(1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    w = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(w, n)*dS_h).dat.data

    assert np.all(temp[:, 0] == 0.0)
    assert not np.all(temp[:, 1] == 0.0)


@pytest.mark.xfail
def test_interior_facet_vfs_extr_horiz_3d():
    m = UnitSquareMesh(1, 1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    w = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(w, n)*dS_h).dat.data

    assert np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)
    assert not np.all(temp[:, 2] == 0.0)


def test_interior_facet_vfs_extr_vert():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, layers=1)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    w = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(w, n)*dS_v).dat.data

    assert not np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

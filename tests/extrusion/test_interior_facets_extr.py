import numpy as np
import pytest

from firedrake import *


def test_interior_facet_vfs_extr_horiz_2d():
    m = UnitIntervalMesh(1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(v, n)*dS_h).dat.data

    assert np.all(temp[:, 0] == 0.0)
    assert not np.all(temp[:, 1] == 0.0)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(dot(u, n)*dot(v, n))*dS_h)

    assert temp.M.values[0, 0] == 0.0
    assert temp.M.values[1, 1] != 0.0
    assert temp.M.values[2, 2] == 0.0
    assert temp.M.values[3, 3] != 0.0


def test_interior_facet_vfs_extr_horiz_3d():
    m = UnitSquareMesh(1, 1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(v, n)*dS_h).dat.data

    assert np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)
    assert not np.all(temp[:, 2] == 0.0)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(dot(u, n)*dot(v, n))*dS_h)

    assert temp.M.values[0, 0] == 0.0
    assert temp.M.values[1, 1] == 0.0
    assert temp.M.values[2, 2] != 0.0


def test_interior_facet_vfs_extr_vert():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, layers=1)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(v, n)*dS_v).dat.data

    assert not np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(dot(u, n)*dot(v, n))*dS_v)

    assert temp.M.values[0, 0] != 0.0
    assert temp.M.values[1, 1] == 0.0
    assert temp.M.values[2, 2] != 0.0
    assert temp.M.values[3, 3] == 0.0

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

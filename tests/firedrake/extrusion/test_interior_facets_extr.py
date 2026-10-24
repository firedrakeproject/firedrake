import numpy as np
import pytest

from firedrake import *


def test_interior_facet_vfs_extr_horiz_2d_rhs():
    m = UnitIntervalMesh(1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(conj(v), n)*dS_h).dat.data_ro

    assert np.all(temp[:, 0] == 0.0)
    assert not np.all(temp[:, 1] == 0.0)


def test_interior_facet_vfs_extr_horiz_2d_lhs():
    m = UnitIntervalMesh(1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(inner(dot(u, n), dot(v, n)))*dS_h)

    assert temp.M.values[0, 0] == 0.0
    assert temp.M.values[1, 1] != 0.0
    assert temp.M.values[2, 2] == 0.0
    assert temp.M.values[3, 3] != 0.0


def test_interior_facet_vfs_extr_horiz_2d_mixed():
    m = UnitIntervalMesh(1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'RTCF', 1)
    W = U*V

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    pp = assemble(inner(u2('+'), v1('+'))*dS_h)
    pm = assemble(inner(u2('+'), v1('-'))*dS_h)
    mp = assemble(inner(u2('-'), v1('+'))*dS_h)
    mm = assemble(inner(u2('-'), v1('-'))*dS_h)

    label0, label1 = W._labels
    assert not np.all(pp.M[label0, label1].values == pm.M[label0, label1].values)
    assert not np.all(pp.M[label0, label1].values == mp.M[label0, label1].values)
    assert not np.all(pp.M[label0, label1].values == mm.M[label0, label1].values)
    assert not np.all(pm.M[label0, label1].values == mp.M[label0, label1].values)
    assert not np.all(pm.M[label0, label1].values == mm.M[label0, label1].values)
    assert not np.all(mp.M[label0, label1].values == mm.M[label0, label1].values)


def test_interior_facet_vfs_extr_horiz_3d_rhs():
    m = UnitSquareMesh(1, 1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(conj(v), n)*dS_h).dat.data_ro

    assert np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)
    assert not np.all(temp[:, 2] == 0.0)


def test_interior_facet_vfs_extr_horiz_3d_lhs():
    m = UnitSquareMesh(1, 1)
    mesh = ExtrudedMesh(m, layers=2)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(inner(dot(u, n), dot(v, n)))*dS_h)

    assert temp.M.values[0, 0] == 0.0
    assert temp.M.values[1, 1] == 0.0
    assert temp.M.values[2, 2] != 0.0


def test_interior_facet_vfs_extr_vert_rhs():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, layers=1)

    U = VectorFunctionSpace(mesh, 'DG', 1)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(jump(conj(v), n)*dS_v).dat.data_ro

    assert not np.all(temp[:, 0] == 0.0)
    assert np.all(temp[:, 1] == 0.0)


def test_interior_facet_vfs_extr_vert_lhs():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, layers=1)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    u = TrialFunction(U)
    v = TestFunction(U)
    n = FacetNormal(mesh)

    temp = assemble(avg(inner(dot(u, n), dot(v, n)))*dS_v)

    assert temp.M.values[0, 0] != 0.0
    assert temp.M.values[1, 1] == 0.0
    assert temp.M.values[2, 2] != 0.0
    assert temp.M.values[3, 3] == 0.0


def test_interior_facet_vfs_extr_vert_mixed():
    m = UnitIntervalMesh(2)
    mesh = ExtrudedMesh(m, layers=1)

    U = VectorFunctionSpace(mesh, 'DG', 0)
    V = FunctionSpace(mesh, 'RTCF', 1)
    W = U*V

    u1, u2 = TrialFunctions(W)
    v1, v2 = TestFunctions(W)

    pp = assemble(inner(u2('+'), v1('+'))*dS_v)
    pm = assemble(inner(u2('+'), v1('-'))*dS_v)
    mp = assemble(inner(u2('-'), v1('+'))*dS_v)
    mm = assemble(inner(u2('-'), v1('-'))*dS_v)

    label0, label1 = W._labels
    assert not np.all(pp.M[label0, label1].values == pm.M[label0, label1].values)
    assert not np.all(pp.M[label0, label1].values == mp.M[label0, label1].values)
    assert not np.all(pp.M[label0, label1].values == mm.M[label0, label1].values)
    assert not np.all(pm.M[label0, label1].values == mp.M[label0, label1].values)
    assert not np.all(pm.M[label0, label1].values == mm.M[label0, label1].values)
    assert not np.all(mp.M[label0, label1].values == mm.M[label0, label1].values)


@pytest.mark.parallel(nprocs=[1, 3])
def test_interior_facet_dS_extr():
    # The unit cube with 2x2 columns and 2 layers has 4 interior vertical
    # facets of total area 2 and one interior horizontal plane of area 1.
    m = UnitSquareMesh(2, 2, quadrilateral=True)
    mesh = ExtrudedMesh(m, layers=2)

    area = assemble(Constant(1)*dS(domain=mesh))

    assert np.isclose(area, 3.0)


@pytest.mark.parallel(nprocs=[1, 3])
def test_interior_facet_maps_extr():
    # The mesh has 4 columns of width 1/4 and 2 layers of height 1/2, so it
    # has 6 interior vertical facets and 4 interior horizontal facets.
    m = UnitIntervalMesh(4)
    mesh = ExtrudedMesh(m, layers=2)
    V = FunctionSpace(mesh, "DG", 0)
    x, z = SpatialCoordinate(mesh)
    centres = np.column_stack([Function(V).interpolate(x).dat.data_ro_with_halos,
                               Function(V).interpolate(z).dat.data_ro_with_halos])

    nfacets = mesh.interior_facets.owned.local_size
    cells = V.interior_facet_node_list[:nfacets]
    local_facets = mesh.interior_facet_local_facet_indices.data_ro.reshape((-1, 2))[:nfacets]

    # The local facets 2*k and 2*k+1 are normal to the k-th axis of the
    # reference cell. The base cells can be reversed, but the extruded axis
    # always points upwards, so the local facet 3 is the top facet of a cell.
    steps = centres[cells[:, 1]] - centres[cells[:, 0]]
    axis = local_facets // 2
    horiz = axis[:, 0] == 1
    # The owned facets of all processes are the 10 distinct interior facets.
    midpoints = np.concatenate(mesh.comm.allgather((centres[cells[:, 0]] + centres[cells[:, 1]]) / 2))
    assert len(np.unique(midpoints.round(8), axis=0)) == len(midpoints) == 10
    assert np.array_equal(axis[:, 0], axis[:, 1])
    assert np.allclose(abs(steps), np.array([1/4, 1/2]) * np.eye(2)[axis[:, 0]])
    assert np.allclose(steps[horiz, 1], np.where(local_facets[horiz, 0] == 3, 1/2, -1/2))

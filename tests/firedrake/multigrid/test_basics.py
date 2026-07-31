from firedrake import *
import pytest


def test_refine_interval():
    m = UnitIntervalMesh(10)

    mh = MeshHierarchy(m, 1)

    assert mh[1].num_cells() == 2 * mh[0].num_cells()


@pytest.mark.parallel(nprocs=2)
def test_refine_interval_parallel():
    m = UnitIntervalMesh(10)

    mh = MeshHierarchy(m, 1)

    assert mh[1].num_cells() < 2 * mh[0].num_cells()


def test_refine_quad_mesh():
    m = UnitSquareMesh(2, 2, quadrilateral=True)

    mh = MeshHierarchy(m, 1)

    assert mh[1].num_cells() == 4 * mh[0].num_cells()


def test_refine_tet_mesh():
    m = UnitCubeMesh(2, 2, 2)

    mh = MeshHierarchy(m, 1)

    assert mh[1].num_cells() == 8 * mh[0].num_cells()


def test_refine_hex_mesh():
    m = UnitSquareMesh(2, 2, quadrilateral=True)
    mh = MeshHierarchy(m, 1)
    mh = ExtrudedMeshHierarchy(mh, layers=[2, 2], height=1)

    assert mh[1].num_cells() == 4 * mh[0].num_cells()


def test_refine_square_ncell():
    m = UnitSquareMesh(1, 1)

    mh = MeshHierarchy(m, 1)

    assert mh[1].num_cells() == 4 * mh[0].num_cells()


@pytest.mark.parallel(nprocs=2)
def test_refine_square_ncell_parallel():
    m = UnitSquareMesh(4, 4)

    mh = MeshHierarchy(m, 1)

    # Should be fewer than 4 times the number of coarse cells due to
    # halo shrinking.
    assert mh[1].num_cells() < 4 * mh[0].num_cells()


def test_mesh_sequence_hierarchy():
    """MeshHierarchy() dispatches across the distinct component meshes of a
    genuinely multi-subdomain MeshSequenceGeometry, building one sub-hierarchy
    per component, and the result supports adaptive growth via add_mesh.
    """
    from firedrake.mesh import MeshSequenceGeometry
    from firedrake.mg.utils import has_level, get_level

    meshA = UnitSquareMesh(2, 2)
    meshB = UnitSquareMesh(3, 3)
    seq = MeshSequenceGeometry([meshA, meshB], set_hierarchy=False)

    mh = MeshHierarchy(seq, 2)
    assert len(mh) == 3
    for level in range(3):
        a, b = mh[level]
        assert a.num_cells() == meshA.num_cells() * 4**level
        assert b.num_cells() == meshB.num_cells() * 4**level

    finest = mh[-1]
    assert has_level(finest[0]) and has_level(finest[1])
    assert get_level(finest[0])[1] == 2
    assert get_level(finest[1])[1] == 2

    # Adaptive growth: refine only the first component, leave the second
    # untouched.
    Za = FunctionSpace(finest[0], "DG", 0)
    Zb = FunctionSpace(finest[1], "DG", 0)
    Z = MixedFunctionSpace([Za, Zb])
    mark = Function(Z)
    mark.subfunctions[0].assign(1)
    mark.subfunctions[1].assign(0)

    refined = finest.refine_marked_elements(mark)
    grown = mh.add_mesh(refined)

    assert len(mh) == 4
    assert grown[0].num_cells() == 4 * finest[0].num_cells()
    assert grown[1].num_cells() == finest[1].num_cells()

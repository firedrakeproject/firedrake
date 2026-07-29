import numpy as np
import pytest

from firedrake import *

@pytest.fixture(params=[          
    # No interior facets
    pytest.param(lambda: UnitIntervalMesh(1), id="interval-1"),
    pytest.param(lambda: UnitSquareMesh(1, 1), id="tri-square-1x1"),
    pytest.param(lambda: UnitSquareMesh(1, 1, quadrilateral=True), id="quad-square-1x1"),
    pytest.param(lambda: UnitCubeMesh(1, 1, 1), id="tet-cube-1x1x1"),
    pytest.param(lambda: UnitCubeMesh(1, 1, 1, hexahedral=True), id="hex-cube-1x1x1"),

    # With interior facets
    pytest.param(lambda: UnitIntervalMesh(2), id="interval-2"),
    pytest.param(lambda: UnitSquareMesh(2, 1), id="tri-square-2x1"),
    pytest.param(lambda: UnitSquareMesh(2, 1, quadrilateral=True), id="quad-square-2x1"),
    pytest.param(lambda: UnitCubeMesh(2, 1, 1), id="tet-cube-2x1x1"),
    pytest.param(lambda: UnitCubeMesh(2, 1, 1, hexahedral=True), id="hex-cube-2x1x1"),                
])
def mesh(request):
    return request.param()


@pytest.mark.parallel([1, 3])
def test_cell_facet_neighbours_are_valid(mesh):
    topology = mesh.topology
    
    # Check that every cell has exactly one neighbour entry per local facet
    owned_neighbours = topology.cell_facet_neighbours.data_ro
    neighbours_with_halos = topology.cell_facet_neighbours.data_ro_with_halos

    assert owned_neighbours.shape == (
        topology.cell_set.size,
        mesh.ufl_cell().num_facets,
    )

    assert neighbours_with_halos.shape == (
        topology.cell_set.total_size,
        mesh.ufl_cell().num_facets,
    )

    for c, row in enumerate(owned_neighbours):
        for n in row:
            if n == -1:
                continue

            # Check that every non-boundary neighbour has a valid cell number
            assert 0 <= n < topology.cell_set.total_size

            # Check reciprocity: If c is the neighbour of n across a given facet, 
            # then n also appears as the neighbour of c
            assert c in neighbours_with_halos[n]


@pytest.mark.parallel([1, 3])
def test_cell_facet_neighbours_match_interior_facets(mesh):
    topology = mesh.topology

    neighbours = topology.cell_facet_neighbours.data_ro_with_halos

    # Test only with owned facets while allowing either adjacent cell to be halo
    nowned_facets = topology.interior_facets.set.size
    facet_cells = topology.interior_facets.facet_cell[:nowned_facets]
    local_facets = topology.interior_facets.local_facet_dat.data_ro[:nowned_facets]
    
    # Check that the neighbouring cell is attached to the right facet
    for (c0, c1), (lf0, lf1) in zip(facet_cells, local_facets):
        if c0 == -1 or c1 == -1:
            # Skip interior facets for which the rank doesn't know both adjacent cells
            continue

        assert 0 <= c0 < topology.cell_set.total_size
        assert 0 <= c1 < topology.cell_set.total_size

        assert neighbours[c0, lf0] == c1
        assert neighbours[c1, lf1] == c0

        f0 = topology._cell_facet_point(c0, lf0)
        f1 = topology._cell_facet_point(c1, lf1)

        assert f0 == f1


@pytest.mark.parallel([1, 3])
def test_cell_facet_exterior_mask(mesh):
    topology = mesh.topology

    mask = topology.cell_facet_exterior_mask

    expected = np.zeros(
        (topology.cell_set.total_size, mesh.ufl_cell().num_facets),
        dtype=bool,
    )

    facet_cells = np.asarray(topology.exterior_facets.facet_cell).reshape((-1, 1))
    local_facets = np.asarray(
        topology.exterior_facets.local_facet_dat.data_ro_with_halos
    ).reshape((-1, 1))

    for (c,), (lf,) in zip(facet_cells, local_facets):
        if c == -1:
            continue
        assert 0 <= c < topology.cell_set.total_size
        expected[c, lf] = True

    assert mask.shape == expected.shape
    assert np.array_equal(mask, expected)


@pytest.mark.parallel([1, 3])
def test_cell_facet_coord_transforms_are_inverse_on_interior_facets(mesh):
    """Checks that the forward and backward transforms are consistent for two adjacent cells over a fixed shared facet."""
    topology = mesh.topology

    A_dat, b_dat = topology.cell_facet_coord_transforms
    A = A_dat.data_ro_with_halos
    b = b_dat.data_ro_with_halos
    
    embed_A, embed_b, _ = topology._get_facet_embedding_maps()

    # Restrict to owned interior facets
    nowned_facets = topology.interior_facets.set.size
    facet_cells = topology.interior_facets.facet_cell[:nowned_facets]
    local_facets = topology.interior_facets.local_facet_dat.data_ro[:nowned_facets]

    for (c0, c1), (lf0, lf1) in zip(facet_cells, local_facets):
        if c0 == -1 or c1 == -1:
            continue

        assert 0 <= c0 < topology.cell_set.total_size
        assert 0 <= c1 < topology.cell_set.total_size

        assert np.isfinite(A[c0, lf0]).all()
        assert np.isfinite(b[c0, lf0]).all()
        assert np.isfinite(A[c1, lf1]).all()
        assert np.isfinite(b[c1, lf1]).all()

        facet_dim = embed_A[lf0].shape[1]
        points = [np.zeros(facet_dim)]
        points += [np.eye(facet_dim)[i] for i in range(facet_dim)]

        for Xf in points:
            x = embed_A[lf0] @ Xf + embed_b[lf0]
            y = A[c0, lf0] @ x + b[c0, lf0]
            x_back = A[c1, lf1] @ y + b[c1, lf1]

            assert np.allclose(x_back, x)


@pytest.mark.parallel([1, 3])
def test_cell_facet_coord_transforms_map_to_neighbour_facet(mesh):
    """Checks geometric consistency, that is the linear transform for a cell c on local facet lf 
    sends points on that facet to points on the corresponding facet of the neighbouring cell."""
    topology = mesh.topology
    
    A_dat, b_dat = topology.cell_facet_coord_transforms
    A = A_dat.data_ro_with_halos
    b = b_dat.data_ro_with_halos

    embed_A, embed_b, _ = topology._get_facet_embedding_maps()

    # Restrict to owned interior facets
    nowned_facets = topology.interior_facets.set.size
    facet_cells = topology.interior_facets.facet_cell[:nowned_facets]
    local_facets = topology.interior_facets.local_facet_dat.data_ro[:nowned_facets]

    for (c0, c1), (lf0, lf1) in zip(facet_cells, local_facets):
        if c0 == -1 or c1 == -1:
            continue

        assert 0 <= c0 < topology.cell_set.total_size
        assert 0 <= c1 < topology.cell_set.total_size

        facet_dim = embed_A[lf0].shape[1]
        points = [np.zeros(facet_dim)]
        points += [np.eye(facet_dim)[i] for i in range(facet_dim)]

        for Xf in points:
            X = embed_A[lf0] @ Xf + embed_b[lf0]
            Y = A[c0, lf0] @ X + b[c0, lf0]

            # Y should lie on neighbour local facet lf1.
            Yf = np.linalg.pinv(embed_A[lf1]) @ (Y - embed_b[lf1])
            assert np.allclose(embed_A[lf1] @ Yf + embed_b[lf1], Y)
        
        facet_dim = embed_A[lf1].shape[1]
        points = [np.zeros(facet_dim)]
        points += [np.eye(facet_dim)[i] for i in range(facet_dim)]

        for Yf in points:
            Y = embed_A[lf1] @ Yf + embed_b[lf1]
            X = A[c1, lf1] @ Y + b[c1, lf1]

            Xf = np.linalg.pinv(embed_A[lf0]) @ (X - embed_b[lf0])
            assert np.allclose(embed_A[lf0] @ Xf + embed_b[lf0], X)


@pytest.mark.parallel([1, 3])
def test_cell_facet_topology_parallel_smoke():
    mesh = UnitSquareMesh(4, 4)
    topology = mesh.topology

    topology.cell_facet_neighbours.data_ro_with_halos

    A, b = topology.cell_facet_coord_transforms
    A.data_ro_with_halos
    b.data_ro_with_halos

    topology.cell_facet_exterior_mask
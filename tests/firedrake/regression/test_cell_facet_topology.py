import numpy as np
import pytest
from collections import defaultdict

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
    pytest.param(lambda: UnitSquareMesh(8, 8), id="tri-square-8x8"),
    pytest.param(lambda: UnitCubeMesh(3, 3, 3, hexahedral=True), id="hex-cube-3x3x3"),
])
def mesh(request):
    return request.param()


@pytest.mark.parallel([1, 3])
def test_cell_facet_neighbours_match_shared_closure_facets(mesh):
    """Checks that two cells are neighbours across a given facet if and only if
    that facet appears in the closure of both cells."""
    topology = mesh.topology
    num_facets = mesh.ufl_cell().num_facets
    neighbour_cells, neighbour_facets = topology._cell_facet_adjacency

    # Get the facet points of each cell, in local facet order.
    fstart, fend = topology.topology_dm.getHeightStratum(1)
    closure = topology.cell_closure
    facet_points = closure[(fstart <= closure) & (closure < fend)].reshape(-1, num_facets)

    # Record the position of each (cell, local facet) at which each facet point occurs.
    positions = defaultdict(list)
    for cell, points in enumerate(facet_points):
        for local_facet, point in enumerate(points):
            positions[point].append((cell, local_facet))

    expected_cells = np.full_like(neighbour_cells, -1)
    expected_facets = np.full_like(neighbour_facets, -1)
    for shared in positions.values():
        assert len(shared) <= 2
        if len(shared) == 2:
            (c0, lf0), (c1, lf1) = shared
            expected_cells[c0, lf0], expected_facets[c0, lf0] = c1, lf1
            expected_cells[c1, lf1], expected_facets[c1, lf1] = c0, lf0

    assert neighbour_cells.shape == (topology.cell_set.total_size, num_facets)
    assert np.array_equal(neighbour_cells, expected_cells)
    assert np.array_equal(neighbour_facets, expected_facets)

    # The Dat in mesh.py must hold the same values as the Cython output.
    assert topology.cell_facet_neighbours.data_ro.shape == (topology.cell_set.size, num_facets)
    assert np.array_equal(topology.cell_facet_neighbours.data_ro_with_halos, neighbour_cells)


@pytest.mark.parallel([1, 3])
def test_cell_facet_neighbours_match_interior_facets(mesh):
    topology = mesh.topology

    neighbours = topology.cell_facet_neighbours.data_ro_with_halos
    neighbour_facets = topology._cell_facet_adjacency[1]

    # Consider only facets owned by this rank
    nowned_facets = topology.interior_facets.set.size
    facet_cells = topology.interior_facets.facet_cell[:nowned_facets]
    local_facets = topology.interior_facets.local_facet_dat.data_ro[:nowned_facets]

    # Check that the neighbouring cell is attached to the right facet
    for (c0, c1), (lf0, lf1) in zip(facet_cells, local_facets):
        if c0 == -1 or c1 == -1:
            # Skip interior facets whose either adjacent cell isn't on this rank
            continue

        assert 0 <= c0 < topology.cell_set.total_size
        assert 0 <= c1 < topology.cell_set.total_size

        assert neighbours[c0, lf0] == c1
        assert neighbours[c1, lf1] == c0
        assert neighbour_facets[c0, lf0] == lf1
        assert neighbour_facets[c1, lf1] == lf0


@pytest.mark.parallel([1, 3])
def test_cell_facet_coord_transforms_map_facet_vertices_to_shared_vertices(mesh):
    """Checks that each vertex of a facet maps to the reference vertex of the
    neighbouring cell that is the same mesh vertex."""
    import FIAT
    topology = mesh.topology
    fiat_cell = FIAT.ufc_cell(mesh.ufl_cell())
    vertices = np.asarray(fiat_cell.get_vertices(), dtype=float)
    facet_vertices = fiat_cell.get_topology()[topology.cell_dimension() - 1]
    # cell_closure contains the mesh vertices of each cell, in the order of the vertices of the FIAT reference cell.
    cell_vertices = topology.cell_closure[:, :len(vertices)]
    neighbour_cells, _ = topology._cell_facet_adjacency
    A_dat, b_dat = topology.cell_facet_coord_transforms
    A, b = A_dat.data_ro_with_halos, b_dat.data_ro_with_halos

    for c, lf in zip(*np.nonzero(neighbour_cells >= 0)):
        n = neighbour_cells[c, lf]
        for k in facet_vertices[lf]:
            # Check that the neighbour's reference vertex that is the same mesh vertex as vertex k of c.
            k_n, = np.flatnonzero(cell_vertices[n] == cell_vertices[c, k])
            assert np.allclose(A[c, lf] @ vertices[k] + b[c, lf], vertices[k_n])

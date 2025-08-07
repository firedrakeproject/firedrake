import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope="module", params=[False, True])
def meshdata(request):
    extruded = request.param
    if extruded:
        m = ExtrudedMesh(UnitIntervalMesh(3), 3)
    else:
        m = UnitSquareMesh(3, 3, quadrilateral=True)
    V = FunctionSpace(m, 'DG', 0)
    f = Function(V)
    x = SpatialCoordinate(m)
    f.interpolate(3*x[0] + 9*x[1] - 1)
    return m, f


@pytest.mark.parametrize(('point', 'value'),
                         [((0.2, 0.1), 1),
                          ((0.5, 0.2), 2),
                          ((0.7, 0.1), 3),
                          ((0.2, 0.4), 4),
                          ((0.4, 0.4), 5),
                          ((0.8, 0.5), 6),
                          ((0.1, 0.7), 7),
                          ((0.5, 0.9), 8),
                          ((0.9, 0.8), 9)])
def test_locate_cell(meshdata, point, value):
    m, f = meshdata

    def value_at(p, cell_ignore=None):
        cell = m.locate_cell(p, cell_ignore=cell_ignore)
        return f.dat.data[cell]

    def value_at_and_dist(p, cell_ignore=None):
        if cell_ignore is not None:
            cell_ignore = [[cell_ignore]]
        cells, _, l1_dists = m.locate_cells_ref_coords_and_dists([p], cells_ignore=cell_ignore)
        return f.dat.data[cells[0]], l1_dists[0]

    assert np.allclose(value, value_at(point))
    cell = m.locate_cell(point)
    assert not np.allclose(value, value_at(point, cell_ignore=cell))
    value_at, l1_dist = value_at_and_dist(point)
    assert np.allclose(value, value_at)
    assert np.isclose(l1_dist, 0.0)
    value_at, l1_dist = value_at_and_dist(point, cell_ignore=cell)
    assert not np.allclose(value, value_at)
    assert l1_dist > 0.0


def test_locate_cell_not_found(meshdata):
    m, f = meshdata

    assert m.locate_cell((0.2, -0.4)) is None


def test_locate_cells_ref_coords_and_dists(meshdata):
    m, f = meshdata

    points = [(0.2, 0.1), (0.5, 0.2), (0.7, 0.1), (0.2, 0.4), (0.4, 0.4), (0.8, 0.5), (0.1, 0.7), (0.5, 0.9), (0.9, 0.8)]
    cells, ref_coords, l1_dists = m.locate_cells_ref_coords_and_dists(points)
    assert np.allclose(f.dat.data[cells], [1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.allclose(l1_dists, 0.0)
    fcells, ref_coords, l1_dists = m.locate_cells_ref_coords_and_dists(points[:2], cells_ignore=np.array([cells[:1], cells[1:2]]))
    assert fcells[0] == -1 or fcells[0] in cells[1:]
    assert fcells[1] == -1 or fcells[1] in cells[2:] or fcells[1] in cells[:1]
    fcells, ref_coords, l1_dists = m.locate_cells_ref_coords_and_dists(points[:2], cells_ignore=np.array([cells[:2], cells[1:3]]))
    assert fcells[0] == -1 or fcells[0] in cells[2:]
    assert fcells[1] == -1 or fcells[1] in cells[3:] or fcells[1] in cells[:1]
    fcells, ref_coords, l1_dists = m.locate_cells_ref_coords_and_dists(points[:2], cells_ignore=np.array([cells[:3], cells[1:4]]))
    assert fcells[0] == -1 or fcells[0] in cells[3:]
    assert fcells[1] == -1 or fcells[1] in cells[4:] or fcells[1] in cells[:1]
    fcells, ref_coords, l1_dists = m.locate_cells_ref_coords_and_dists(points[:2], cells_ignore=np.array([cells[:4], cells[1:5]]))
    assert fcells[0] == -1 or fcells[0] in cells[4:]
    assert fcells[1] == -1 or fcells[1] in cells[5:] or fcells[1] in cells[:1]


def test_high_order_location():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 3, variant="equispaced")
    f = Function(V)
    f.interpolate(mesh.coordinates)

    def warp(x, p):
        return p * x * (2 * x - 1)

    warp_indices = np.where((f.dat.data[:, 0] > 0.0) & (f.dat.data[:, 0] < 0.5) & (f.dat.data[:, 1] == 0.0))[0]
    f.dat.data[warp_indices, 1] = warp(f.dat.data[warp_indices, 0], 5.0)

    mesh = Mesh(f)

    # The point (0.25, -0.6) *is* in the mesh, but falls outside the Lagrange bounding box
    # The below used to return (None, None), but projecting to Bernstein coordinates
    # allows us to locate the cell.
    assert mesh.locate_cell([0.25, -0.6], tolerance=0.001) is not None
    # The point (0.25, -0.7) is outside the mesh, but inside the Bernstein bounding box.
    # This should return (None, None).
    assert mesh.locate_cell([0.25, -0.7], tolerance=0.001) is None

    # Change mesh coordinates to check that the bounding box is recalculated
    mesh.coordinates.dat.data_wo[warp_indices, 1] = warp(mesh.coordinates.dat.data_ro[warp_indices, 0], 8.0)
    assert mesh.locate_cell([0.25, -0.6], tolerance=0.0001) is not None
    assert mesh.locate_cell([0.25, -0.7], tolerance=0.0001) is not None
    assert mesh.locate_cell([0.25, -0.95], tolerance=0.0001) is not None
    assert mesh.locate_cell([0.25, -1.05], tolerance=0.0001) is None


def test_high_order_location_internal():
    mesh = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(mesh, "CG", 3, variant="equispaced")
    f = Function(V)
    f.interpolate(mesh.coordinates)

    warp_indices = np.where((f.dat.data[:, 0] > 0.0) & (f.dat.data[:, 0] < 0.5) & np.isclose(f.dat.data[:, 1], 0.5))[0]
    f.dat.data[warp_indices, 1] += 0.1
    mesh = Mesh(f)

    assert mesh.locate_cell([0.25, 0.605], tolerance=0.0001) == 1
    assert mesh.locate_cell([0.25, 0.62], tolerance=0.0001) == 3
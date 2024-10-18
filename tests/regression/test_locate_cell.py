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

    def value_at(p):
        cell = m.locate_cell(p)
        return f.dat.data[cell]

    assert np.allclose(value, value_at(point))


def test_locate_cell_not_found(meshdata):
    m, f = meshdata

    assert m.locate_cell((0.2, -0.4)) is None


@pytest.mark.parallel(2)
def test_owned_cells_preferred_over_ghost_cells():
    """Test that points are preferentially allocated into owned cells.

    Specifically this test takes the partitioned global mesh

           p0   |   p1
        o-----o | -----o
                |

    and tries to locate a point (x) onto it

           p0   |   p1
        o-----o | -----o
                |  x

    Since points should be allocated preferentially to owned cells, both
    ranks (p0 and p1) should claim that x lives in their owned cell, which
    is always numbered 0.

    This behaviour is important to ensure that points do not get lost between
    processes, where p0 would believe p1 to own the point and p1 believe that
    it is owned by p0 (e.g. https://github.com/firedrakeproject/firedrake/issues/3151).

    """
    mesh = UnitIntervalMesh(2)
    mesh.init()

    # each process owns one cell and sees both
    assert mesh.cell_set.size == 1
    assert mesh.cell_set.total_size == 2

    # points near the boundary should be claimed by both cells
    coords = [(0.4,), (0.6,)]
    for coord in coords:
        cell = mesh.locate_cell(coord, tolerance=0.5)
        assert cell == 0

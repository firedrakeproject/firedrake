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


@pytest.mark.parametrize(('points', 'values'),
                         [([(0.2, 0.1), (0.5, 0.2), (0.7, 0.1), (0.2, 0.4), (0.4, 0.4),
                            (0.8, 0.5), (0.1, 0.7), (0.5, 0.9), (0.9, 0.8)],
                          [1, 2, 3, 4, 5, 6, 7, 8, 9])])
def test_locate_cells(meshdata, points, values):
    m, f = meshdata

    point_responses = [f.dat.data[cell] for cell in m.locate_cells(points)]

    assert np.allclose(values, point_responses)


def test_locate_cell_not_found(meshdata):
    m, f = meshdata
    assert m.locate_cells(np.array([[0.2, -0.4]]))[0] == -1

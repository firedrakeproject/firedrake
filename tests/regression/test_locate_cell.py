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

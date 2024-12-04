import pytest
import numpy as np

from FIAT.reference_element import UFCInterval, UFCTriangle, UFCTetrahedron
from FIAT.reference_element import UFCQuadrilateral, TensorProductCell

from tsfc.fem import make_cell_facet_jacobian

interval = UFCInterval()
triangle = UFCTriangle()
quadrilateral = UFCQuadrilateral()
tetrahedron = UFCTetrahedron()
interval_x_interval = TensorProductCell(interval, interval)
triangle_x_interval = TensorProductCell(triangle, interval)
quadrilateral_x_interval = TensorProductCell(quadrilateral, interval)


@pytest.mark.parametrize(('cell', 'cell_facet_jacobian'),
                         [(interval, [[],
                                      []]),
                          (triangle, [[-1, 1],
                                      [0, 1],
                                      [1, 0]]),
                          (quadrilateral, [[0, 1],
                                           [0, 1],
                                           [1, 0],
                                           [1, 0]]),
                          (tetrahedron, [[-1, -1, 1, 0, 0, 1],
                                         [0, 0, 1, 0, 0, 1],
                                         [1, 0, 0, 0, 0, 1],
                                         [1, 0, 0, 1, 0, 0]])])
def test_cell_facet_jacobian(cell, cell_facet_jacobian):
    facet_dim = cell.get_spatial_dimension() - 1
    for facet_number in range(len(cell.get_topology()[facet_dim])):
        actual = make_cell_facet_jacobian(cell, facet_dim, facet_number)
        expected = np.reshape(cell_facet_jacobian[facet_number], actual.shape)
        assert np.allclose(expected, actual)


@pytest.mark.parametrize(('cell', 'cell_facet_jacobian'),
                         [(interval_x_interval, [1, 0]),
                          (triangle_x_interval, [1, 0, 0, 1, 0, 0]),
                          (quadrilateral_x_interval, [[1, 0, 0, 1, 0, 0]])])
def test_cell_facet_jacobian_horiz(cell, cell_facet_jacobian):
    dim = cell.get_spatial_dimension()

    actual = make_cell_facet_jacobian(cell, (dim - 1, 0), 0)  # bottom facet
    assert np.allclose(np.reshape(cell_facet_jacobian, actual.shape), actual)

    actual = make_cell_facet_jacobian(cell, (dim - 1, 0), 1)  # top facet
    assert np.allclose(np.reshape(cell_facet_jacobian, actual.shape), actual)


@pytest.mark.parametrize(('cell', 'cell_facet_jacobian'),
                         [(interval_x_interval, [[0, 1],
                                                 [0, 1]]),
                          (triangle_x_interval, [[-1, 0, 1, 0, 0, 1],
                                                 [0, 0, 1, 0, 0, 1],
                                                 [1, 0, 0, 0, 0, 1]]),
                          (quadrilateral_x_interval, [[0, 0, 1, 0, 0, 1],
                                                      [0, 0, 1, 0, 0, 1],
                                                      [1, 0, 0, 0, 0, 1],
                                                      [1, 0, 0, 0, 0, 1]])])
def test_cell_facet_jacobian_vert(cell, cell_facet_jacobian):
    dim = cell.get_spatial_dimension()
    vert_dim = (dim - 2, 1)
    for facet_number in range(len(cell.get_topology()[vert_dim])):
        actual = make_cell_facet_jacobian(cell, vert_dim, facet_number)
        expected = np.reshape(cell_facet_jacobian[facet_number], actual.shape)
        assert np.allclose(expected, actual)


if __name__ == "__main__":
    import os
    import sys
    pytest.main(args=[os.path.abspath(__file__)] + sys.argv[1:])

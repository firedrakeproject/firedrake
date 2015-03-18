import pytest

from firedrake import *


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_1d(n):
    assert abs(assemble(CellSize(UnitIntervalMesh(n))*dx) - 1.0/n) < 1e-14


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_2d(n):
    assert abs(assemble(CellSize(UnitSquareMesh(n, n))*dx) - 1.0/n) < 1e-14


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_3d(n):
    assert abs(assemble(CellSize(UnitCubeMesh(n, n, n))*dx) - 1.0/n) < 5e-13

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

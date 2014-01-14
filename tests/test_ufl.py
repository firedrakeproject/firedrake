import pytest

from firedrake import *


def test_cellsize_1d():
    assert abs(assemble(CellSize(UnitIntervalMesh(1))*dx) - 1.0) < 1e-14


def test_cellsize_2d():
    assert abs(assemble(CellSize(UnitSquareMesh(1, 1))*dx) - sqrt(2)) < 1e-14


def test_cellsize_3d():
    assert abs(assemble(CellSize(UnitCubeMesh(1, 1, 1))*dx) - sqrt(3)) < 1e-14

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

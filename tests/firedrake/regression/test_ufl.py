import pytest

from firedrake import *
from firedrake.utils import single_mode


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_1d(n):
    assert abs(assemble(CellSize(UnitIntervalMesh(n))*dx) - 1.0/n) < (1e-7 if single_mode else 1e-14)


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_2d(n):
    assert abs(assemble(CellSize(UnitSquareMesh(n, n))*dx) - sqrt(2)/n) < (1e-6 if single_mode else 1e-14)


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_3d(n):
    assert abs(assemble(CellSize(UnitCubeMesh(n, n, n))*dx) - sqrt(3)/n) < (1e-4 if single_mode else 5e-12)

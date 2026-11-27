import pytest

from firedrake import *


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_1d(n):
    assert abs(assemble(CellSize(UnitIntervalMesh(n))*dx) - 1.0/n) < 1e-14


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_2d(n):
    assert abs(assemble(CellSize(UnitSquareMesh(n, n))*dx) - sqrt(2)/n) < 1e-14


@pytest.mark.parametrize('n', [1, 3, 16])
def test_cellsize_3d(n):
    assert abs(assemble(CellSize(UnitCubeMesh(n, n, n))*dx) - sqrt(3)/n) < 5e-12


def test_reject_noncontiguous_form_arguments():
    mesh = UnitSquareMesh(5, 5)
    V = FunctionSpace(mesh, "CG", 1)
    form = Argument(V, 0) * Argument(V, 2) * dx

    with pytest.raises(ValueError, match="contiguous"):
        assemble(form)

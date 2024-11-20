"""This tests that exchanging the coordinate field for one of a different dimension does the right thing."""

import pytest

from firedrake import *


@pytest.mark.parametrize("dim", [2, 3])
def test_immerse_1d(dim):
    m = UnitIntervalMesh(5)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, dim)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension() == dim


def test_immerse_2d():
    m = UnitSquareMesh(2, 2)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, 3)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension() == 3


def test_project_2d():
    m = CircleManifoldMesh(5)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, 1)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension() == 1


def test_immerse_extruded():
    m1 = UnitIntervalMesh(5)
    m = ExtrudedMesh(m1, 10)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, 3)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension() == 3

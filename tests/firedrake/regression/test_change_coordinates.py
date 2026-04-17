"""This tests that exchanging the coordinate field for one of a different dimension does the right thing."""

import pytest

from firedrake import *


@pytest.mark.parametrize("dim", [2, 3])
def test_immerse_1d(dim):
    m = UnitIntervalMesh(5)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, dim)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension == dim


def test_immerse_2d():
    m = UnitSquareMesh(2, 2)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, 3)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension == 3


def test_project_2d():
    m = CircleManifoldMesh(5)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, 1)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension == 1


def test_immerse_extruded():
    m1 = UnitIntervalMesh(5)
    m = ExtrudedMesh(m1, 10)
    cfs = VectorFunctionSpace(m, "Lagrange", 1, 3)
    new_coords = Function(cfs)

    m = Mesh(new_coords)

    assert m.geometric_dimension == 3


def test_relabeled_mesh_preserves_coord_changes():
    orig_mesh = UnitSquareMesh(3, 3)

    high_order_space = VectorFunctionSpace(orig_mesh, "CG", 3)
    high_order_coords = Function(high_order_space).interpolate(orig_mesh.coordinates)
    high_order_mesh = Mesh(high_order_coords)

    x, _ = SpatialCoordinate(high_order_mesh)
    marker_space = FunctionSpace(high_order_mesh, "DG", 0)
    marker = Function(marker_space).interpolate(conditional(x > 0.5, 1., 0.))
    relabeled_mesh = RelabeledMesh(high_order_mesh, [marker], [666])

    expected = high_order_mesh.coordinates.dat.data_ro
    actual = relabeled_mesh.coordinates.dat.data_ro
    assert actual.shape == expected.shape
    assert (actual == expected).all()

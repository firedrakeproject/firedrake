import numpy
import pytest

from firedrake import *


def test_facet_normal_unit_interval():
    """Compute facet normals on the sides of the unit square."""

    m = UnitIntervalMesh(2)
    V = VectorFunctionSpace(m, 'CG', 1)
    x_hat = Function(V).interpolate(Constant((1.0,)))
    n = FacetNormal(m)

    assert assemble(dot(x_hat, n)*ds(1)) == -1.0   # x = 0
    assert assemble(dot(x_hat, n)*ds(2)) == 1.0  # x = 1


@pytest.mark.parametrize('quadrilateral', [False, True])
def test_facet_normal_unit_square(quadrilateral):
    """Compute facet normals on the sides of the unit square."""

    m = UnitSquareMesh(2, 2, quadrilateral=quadrilateral)
    V = VectorFunctionSpace(m, 'CG', 1)
    x_hat = Function(V).interpolate(Constant((1, 0)))
    y_hat = Function(V).interpolate(Constant((0, 1)))
    n = FacetNormal(m)

    assert numpy.allclose(assemble(dot(x_hat, n)*ds(1)), -1.0)  # x = 0
    assert numpy.allclose(assemble(dot(x_hat, n)*ds(2)), 1.0)   # x = 1
    assert numpy.allclose(assemble(dot(x_hat, n)*ds(3)), 0.0)   # y = 0
    assert numpy.allclose(assemble(dot(x_hat, n)*ds(4)), 0.0)   # y = 1

    assert numpy.allclose(assemble(dot(y_hat, n)*ds(1)), 0.0)   # x = 0
    assert numpy.allclose(assemble(dot(y_hat, n)*ds(2)), 0.0)   # x = 1
    assert numpy.allclose(assemble(dot(y_hat, n)*ds(3)), -1.0)  # y = 0
    assert numpy.allclose(assemble(dot(y_hat, n)*ds(4)), 1.0)   # y = 1


def test_facet_normal_unit_cube():
    """Compute facet normals on the sides of the unit cube."""

    m = UnitCubeMesh(1, 1, 1)
    V = VectorFunctionSpace(m, 'CG', 1)
    x_hat = Function(V).interpolate(Constant((1, 0, 0)))
    y_hat = Function(V).interpolate(Constant((0, 1, 0)))
    z_hat = Function(V).interpolate(Constant((0, 0, 1)))
    n = FacetNormal(m)

    assert abs(assemble(dot(x_hat, n)*ds(1)) + 1.0) < 1e-14  # x = 0
    assert abs(assemble(dot(x_hat, n)*ds(2)) - 1.0) < 1e-14  # x = 1
    assert abs(assemble(dot(x_hat, n)*ds(3)) - 0.0) < 1e-14  # y = 0
    assert abs(assemble(dot(x_hat, n)*ds(4)) - 0.0) < 1e-14  # y = 1
    assert abs(assemble(dot(x_hat, n)*ds(5)) - 0.0) < 1e-14  # z = 0
    assert abs(assemble(dot(x_hat, n)*ds(6)) - 0.0) < 1e-14  # z = 1

    assert abs(assemble(dot(y_hat, n)*ds(1)) - 0.0) < 1e-14  # x = 0
    assert abs(assemble(dot(y_hat, n)*ds(2)) - 0.0) < 1e-14  # x = 1
    assert abs(assemble(dot(y_hat, n)*ds(3)) + 1.0) < 1e-14  # y = 0
    assert abs(assemble(dot(y_hat, n)*ds(4)) - 1.0) < 1e-14  # y = 1
    assert abs(assemble(dot(y_hat, n)*ds(5)) - 0.0) < 1e-14  # z = 0
    assert abs(assemble(dot(y_hat, n)*ds(6)) - 0.0) < 1e-14  # z = 1

    assert abs(assemble(dot(z_hat, n)*ds(1)) - 0.0) < 1e-14  # x = 0
    assert abs(assemble(dot(z_hat, n)*ds(2)) - 0.0) < 1e-14  # x = 1
    assert abs(assemble(dot(z_hat, n)*ds(3)) - 0.0) < 1e-14  # y = 0
    assert abs(assemble(dot(z_hat, n)*ds(4)) - 0.0) < 1e-14  # y = 1
    assert abs(assemble(dot(z_hat, n)*ds(5)) + 1.0) < 1e-14  # z = 0
    assert abs(assemble(dot(z_hat, n)*ds(6)) - 1.0) < 1e-14  # z = 1

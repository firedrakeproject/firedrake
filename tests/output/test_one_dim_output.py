from __future__ import absolute_import, print_function, division
from firedrake import *
from firedrake.plot import calculate_one_dim_points
import numpy as np


def test_plot_constant():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(as_ufl(1.0))
    x_vals, y_vals = calculate_one_dim_points(f, 10)
    for y in y_vals:
        assert np.allclose(1.0, y)


def test_plot_linear():
    mesh = IntervalMesh(10, 20)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] * 2.0)
    x_vals, y_vals = calculate_one_dim_points(f, 10)
    points = np.array([x_vals, y_vals]).T.reshape(2, -1)
    for point in points:
        assert np.allclose(2.0 * point[0], point[1])


def test_plot_quadratic():
    mesh = IntervalMesh(10, 20)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] ** 2 - x[0] + 3)
    x_vals, y_vals = calculate_one_dim_points(f, 10)
    points = np.array([x_vals, y_vals]).T.reshape(2, -1)
    for point in points:
        assert np.allclose(point[0] ** 2 - point[0] + 3, point[1])

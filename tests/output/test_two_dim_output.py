from __future__ import absolute_import, print_function, division
from firedrake import *
from firedrake.plot import _calculate_points
import numpy as np


def test_constant():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.interpolate(as_ufl(1.0))
    coord_vals, f_vals = _calculate_points(f, 10, 2)
    for f in f_vals:
        assert np.allclose(1.0, f)


def test_linear():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] + x[1])
    coords_vals, f_vals = _calculate_points(f, 10, 2)
    coords_vals = coords_vals.reshape(-1, 2)
    f_vals = f_vals.reshape(-1, 1)
    for i in range(f_vals.size):
        assert np.allclose(coords_vals[i][0] + coords_vals[i][1], f_vals[i])


def test_quadratic():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] ** 2 + x[1] ** 2)
    coords_vals, f_vals = _calculate_points(f, 10, 2)
    coords_vals = coords_vals.reshape(-1, 2)
    f_vals = f_vals.reshape(-1, 1)
    for i in range(f_vals.size):
        assert np.allclose(coords_vals[i][0] ** 2 + coords_vals[i][1] ** 2,
                           f_vals[i])


def test_linear_and_quadratic():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] + x[1] ** 2)
    coords_vals, f_vals = _calculate_points(f, 10, 2)
    coords_vals = coords_vals.reshape(-1, 2)
    f_vals = f_vals.reshape(-1, 1)
    for i in range(f_vals.size):
        assert np.allclose(coords_vals[i][0] + coords_vals[i][1] ** 2,
                           f_vals[i])

import pytest
import numpy as np
from firedrake import *

try:
    from firedrake.pyplot import FunctionPlotter
except ImportError:
    # Matplotlib is not installed
    pytest.skip("Matplotlib not installed", allow_module_level=True)


def test_1d_constant():
    mesh = UnitIntervalMesh(1)
    V = FunctionSpace(mesh, "CG", 1)
    fn_plotter = FunctionPlotter(mesh, 10)
    f = Function(V)
    f.interpolate(as_ufl(1.0))
    f_vals = fn_plotter(f)
    assert np.allclose(1.0, f_vals)


def test_1d_linear():
    mesh = IntervalMesh(10, 20)
    V = FunctionSpace(mesh, "CG", 1)
    fn_plotter = FunctionPlotter(mesh, 10)
    f = Function(V)
    x, = SpatialCoordinate(mesh)
    f.interpolate(2 * x)
    x_vals = fn_plotter(mesh.coordinates)
    f_vals = fn_plotter(f)
    assert np.allclose(2.0 * x_vals, f_vals)


def test_1d_quadratic():
    mesh = IntervalMesh(10, 20)
    V = FunctionSpace(mesh, "CG", 2)
    fn_plotter = FunctionPlotter(mesh, 10)
    f = Function(V)
    x, = SpatialCoordinate(mesh)
    f.interpolate(x**2 - x + 3)
    x_vals = fn_plotter(mesh.coordinates)
    f_vals = fn_plotter(f)
    assert np.allclose(x_vals**2 - x_vals + 3, f_vals)


def test_2d_constant():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    fn_plotter = FunctionPlotter(mesh, 10)
    f = Function(V)
    f.interpolate(as_ufl(1.0))
    f_vals = fn_plotter(f)
    assert np.allclose(1.0, f_vals)


def test_2d_linear():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    fn_plotter = FunctionPlotter(mesh, 10)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] + x[1])
    coord_vals = fn_plotter(mesh.coordinates).reshape(-1, 2)
    f_vals = fn_plotter(f)
    assert np.allclose(coord_vals[:, 0] + coord_vals[:, 1], f_vals)


def test_2d_quadratic():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    fn_plotter = FunctionPlotter(mesh, 10)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] + x[1] ** 2)
    coord_vals = fn_plotter(mesh.coordinates).reshape(-1, 2)
    f_vals = fn_plotter(f)
    assert np.allclose(coord_vals[:, 0] + coord_vals[:, 1]**2, f_vals)

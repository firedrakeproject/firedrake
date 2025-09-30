from os.path import abspath, dirname, join
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


@pytest.fixture(params=[False, True])
def mesh1d(request):
    periodic = request.param
    if periodic:
        return PeriodicUnitIntervalMesh(16)
    else:
        return UnitIntervalMesh(16)


@pytest.fixture(params=[('cg', False),
                        ('cg', True),
                        ('dg', False),
                        ('dg', True),
                        # TODO: generate mesh from .geo file
                        ('file', 't11_tria.msh'),
                        ('file', 't11_quad.msh')])
def mesh2d(request):
    if request.param[0] == 'cg':
        return UnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'dg':
        return PeriodicUnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'file':
        meshfile = join(cwd, '..', 'meshes', request.param[1])
        return Mesh(meshfile)


@pytest.fixture
def mesh3d():
    return UnitCubeMesh(8, 8, 8)


@pytest.fixture
def circle_mesh():
    return CircleManifoldMesh(16)


@pytest.fixture(params=['triangle', 'quadrilateral'])
def sphere_mesh(request):
    if request.param == 'triangle':
        return UnitIcosahedralSphereMesh(5)
    elif request.param == 'quadrilateral':
        return UnitCubedSphereMesh(4)


@pytest.fixture
def func1d(mesh1d):
    V = FunctionSpace(mesh1d, "CG", 2)
    x = SpatialCoordinate(mesh1d)
    f = Function(V).interpolate(x[0]*(1.0 - x[0]))
    return f


@pytest.fixture
def func2d(mesh2d):
    V = FunctionSpace(mesh2d, "CG", 2)
    x = SpatialCoordinate(mesh2d)
    f = Function(V).interpolate(cos(pi*(x[1] - 0.5))*sin(pi*x[0]))
    return f


@pytest.fixture
def func3d(mesh3d):
    V = FunctionSpace(mesh3d, "CG", 2)
    x = SpatialCoordinate(mesh3d)
    f = Function(V).interpolate(x[2]*(x[0] - 0.5*x[1]))
    return f


def test_1d(func1d):
    expr = lambda x: x*(1 - x)
    assert np.allclose(expr(0.1), func1d([0.1]))
    assert np.allclose(expr(0.5), func1d([0.5]))
    assert np.allclose(expr(0.6), func1d([0.6]))


def test_2d(func2d):
    expr = lambda x, y: cos(pi*(y - 0.5))*sin(pi*x)
    assert np.allclose(expr(0.10, 0.20), func2d([0.10, 0.20]), rtol=1e-2)
    assert np.allclose(expr(0.98, 0.94), func2d([0.98, 0.94]), rtol=1e-2)
    assert np.allclose(expr(0.72, 0.88), func2d([0.72, 0.88]), rtol=1e-2)


def test_3d(func3d):
    expr = lambda x, y, z: z*(x - 0.5*y)
    assert np.allclose(expr(0.10, 0.20, 0.00), func3d([0.10, 0.20, 0.00]))
    assert np.allclose(expr(0.96, 0.02, 0.51), func3d([0.96, 0.02, 0.51]))
    assert np.allclose(expr(0.39, 0.57, 0.49), func3d([0.39, 0.57, 0.49]))


def test_circle(circle_mesh):
    f = circle_mesh.coordinates
    point = [0.70710678118, +0.70710678118]
    assert np.allclose(f(point), point)


def test_sphere(sphere_mesh):
    f = sphere_mesh.coordinates
    point = [+0.57735026919, 0.57735026919, +0.57735026919]
    assert np.allclose(f(point), point, rtol=1.e-3)

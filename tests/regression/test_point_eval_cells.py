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
                        pytest.mark.xfail(('dg', False)),
                        ('dg', True),
                        # TODO: generate mesh from .geo file
                        ('file', '../t11_tria.msh'),
                        ('file', '../t11_quad.msh')])
def mesh2d(request):
    if request.param[0] == 'cg':
        return UnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'dg':
        return PeriodicUnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'file':
        meshfile = join(cwd, request.param[1])
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
        return UnitIcosahedralSphereMesh(3)
    elif request.param == 'quadrilateral':
        return UnitCubedSphereMesh(4)


@pytest.fixture
def func1d(mesh1d):
    V = FunctionSpace(mesh1d, "CG", 2)
    f = Function(V).interpolate(Expression("x[0]*(1.0 - x[0])"))
    return f


@pytest.fixture
def func2d(mesh2d):
    V = FunctionSpace(mesh2d, "CG", 2)
    f = Function(V).interpolate(Expression("x[1]*(x[0] - 0.5*x[1])"))
    return f


@pytest.fixture
def func3d(mesh3d):
    V = FunctionSpace(mesh3d, "CG", 2)
    f = Function(V).interpolate(Expression("x[2]*(x[0] - 0.5*x[1])"))
    return f


def test_1d(func1d):
    assert np.allclose(0.09, func1d([0.1]))
    assert np.allclose(0.25, func1d([0.5]))
    assert np.allclose(0.24, func1d([0.6]))


def test_2d(func2d):
    assert np.allclose(0.0000, func2d([0.10, 0.20]))
    assert np.allclose(0.4794, func2d([0.98, 0.94]))
    assert np.allclose(0.2464, func2d([0.72, 0.88]))


def test_3d(func3d):
    assert np.allclose(0.00000, func3d([0.10, 0.20, 0.00]))
    assert np.allclose(0.48450, func3d([0.96, 0.02, 0.51]))
    assert np.allclose(0.05145, func3d([0.39, 0.57, 0.49]))


@pytest.mark.xfail(run=False)
def test_circle(circle_mesh):
    f = func2d(circle_mesh)
    assert np.allclose(+0.2500, f([0.70710678118, +0.70710678118]))
    assert np.allclose(-0.5000, f([0.00000000000, -1.00000000000]))
    assert np.allclose(-0.4352, f([0.36000000000, -0.64000000000]))


@pytest.mark.xfail(run=False)
def test_sphere(sphere_mesh):
    f = func3d(sphere_mesh)
    assert np.allclose(+0.1666666666, f([+0.57735026919, 0.57735026919, +0.57735026919]))
    assert np.allclose(-0.4870627924, f([-0.40000000000, 0.59160797831, +0.70000000000]))
    assert np.allclose(+0.2304000000, f([-0.36000000000, 0.00000000000, -0.64000000000]))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

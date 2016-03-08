from os.path import abspath, dirname, join
import numpy as np
import pytest

from firedrake import *
from tests.common import longtest

cwd = abspath(dirname(__file__))


@pytest.fixture(params=[False, True])
def mesh2d(request):
    periodic = request.param
    if periodic:
        m = PeriodicUnitIntervalMesh(16)
    else:
        m = UnitIntervalMesh(16)
    return ExtrudedMesh(m, 12)


@pytest.fixture(params=[('cg', False),
                        ('cg', True),
                        ('dg', False),
                        pytest.mark.xfail(('dg', True)),
                        # TODO: generate mesh from .geo file
                        longtest(('file', '../t11_tria.msh')),
                        longtest(('file', '../t11_quad.msh'))])
def mesh3d(request):
    if request.param[0] == 'cg':
        m = UnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'dg':
        m = PeriodicUnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'file':
        meshfile = join(cwd, request.param[1])
        m = Mesh(meshfile)
    return ExtrudedMesh(m, 12)


@pytest.fixture
def cylinder_mesh():
    m = CircleManifoldMesh(16)
    return ExtrudedMesh(m, 12)


@pytest.fixture(params=['triangle', 'quadrilateral'])
def spherical_shell_mesh(request):
    if request.param == 'triangle':
        m = UnitIcosahedralSphereMesh(3)
    elif request.param == 'quadrilateral':
        m = UnitCubedSphereMesh(4)
    return ExtrudedMesh(m, 12, layer_height=1.0/12, extrusion_type='radial')


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


def test_2d(func2d):
    assert np.allclose(0.0000, func2d([0.10, 0.20]))
    assert np.allclose(0.4794, func2d([0.98, 0.94]))
    assert np.allclose(0.2464, func2d([0.72, 0.88]))


def test_3d(func3d):
    assert np.allclose(0.00000, func3d([0.10, 0.20, 0.00]))
    assert np.allclose(0.48450, func3d([0.96, 0.02, 0.51]))
    assert np.allclose(0.05145, func3d([0.39, 0.57, 0.49]))


@pytest.mark.xfail(run=False)
def test_cylinder(cylinder_mesh):
    f = func3d(cylinder_mesh)
    assert np.allclose(0.00, f([0.70710678118, +0.70710678118, 0.0]))
    assert np.allclose(0.25, f([0.00000000000, -1.00000000000, 0.5]))
    assert np.allclose(0.68, f([0.36000000000, -0.64000000000, 1.0]))


def test_spherical_shell(spherical_shell_mesh):
    f = func3d(spherical_shell_mesh)
    assert np.allclose(+0.2400000000, f([+0.69282032302, 0.69282032302, +0.69282032302]))
    assert np.allclose(-1.5780834474, f([-0.72000000000, 1.06489436096, +1.26000000000]))
    assert np.allclose(+0.5184000000, f([-0.54000000000, 0.00000000000, -0.96000000000]))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

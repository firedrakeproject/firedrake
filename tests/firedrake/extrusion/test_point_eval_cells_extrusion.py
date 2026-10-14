from os.path import abspath, dirname, join
import numpy as np
import pytest

from firedrake import *

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
                        ('dg', True),
                        # TODO: generate mesh from .geo file
                        ('file', 't11_tria.msh'),
                        ('file', 't11_quad.msh')])
def mesh3d(request):
    if request.param[0] == 'cg':
        m = UnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'dg':
        m = PeriodicUnitSquareMesh(12, 12, quadrilateral=request.param[1])
    elif request.param[0] == 'file':
        meshfile = join(cwd, '..', 'meshes', request.param[1])
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
    x, y = SpatialCoordinate(mesh2d)
    f = Function(V).interpolate(y*(x - 0.5*y))
    return f


@pytest.fixture
def func3d(mesh3d):
    V = FunctionSpace(mesh3d, "CG", 2)
    x, y, z = SpatialCoordinate(mesh3d)
    f = Function(V).interpolate(sin(pi*z)*(cos(pi*(x-0.5)) - sin(pi*y)))
    return f


def test_2d(func2d):
    assert np.allclose(0.0000, func2d([0.10, 0.20]))
    assert np.allclose(0.4794, func2d([0.98, 0.94]))
    assert np.allclose(0.2464, func2d([0.72, 0.88]))


def test_3d(func3d):
    expr = lambda x, y, z: sin(pi*z)*(cos(pi*(x - 0.5)) - sin(pi*y))
    assert np.allclose(expr(0.10, 0.20, 0.00), func3d([0.10, 0.20, 0.00]), rtol=1e-2)
    assert np.allclose(expr(0.96, 0.02, 0.51), func3d([0.96, 0.02, 0.51]), rtol=1e-2)
    assert np.allclose(expr(0.39, 0.57, 0.49), func3d([0.39, 0.57, 0.49]), rtol=1e-2)


def test_cylinder(cylinder_mesh):
    f = cylinder_mesh.coordinates
    point = [0.70710678118, +0.70710678118, 0.0]
    assert np.allclose(f(point), point)


def test_spherical_shell(spherical_shell_mesh):
    V = FunctionSpace(spherical_shell_mesh, "CG", 2)
    x, y, z = SpatialCoordinate(spherical_shell_mesh)
    f = Function(V).interpolate(sin(pi*z)*(cos(pi*(x-0.5)) - sin(pi*y)))
    expr = lambda x, y, z: sin(pi*z)*(cos(pi*(x - 0.5)) - sin(pi*y))
    for pt in [[+0.69282032302, 0.69282032302, +0.69282032302],
               [-0.72000000000, 1.06489436096, +1.26000000000],
               [-0.54000000000, 0.00000000000, -0.96000000000]]:
        assert np.allclose(expr(*pt), f(pt), rtol=5e-2)

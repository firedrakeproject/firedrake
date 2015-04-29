import pytest
from decorator import decorator

from firedrake import *


longtest = pytest.mark.skipif("config.option.short")


@decorator
def disable_cache_lazy(func, *args, **kwargs):
    val = parameters["assembly_cache"]["enabled"]
    parameters["assembly_cache"]["enabled"] = False
    lazy_val = parameters["pyop2_options"]["lazy_evaluation"]
    parameters["pyop2_options"]["lazy_evaluation"] = False
    try:
        func(*args, **kwargs)
    finally:
        parameters["assembly_cache"]["enabled"] = val
        parameters["pyop2_options"]["lazy_evaluation"] = lazy_val


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def extmesh(nx, ny, nz, quadrilateral=False):
    return ExtrudedMesh(UnitSquareMesh(nx, ny, quadrilateral=quadrilateral), nz, layer_height=1.0/nz)


def extmesh_2D(nx, ny):
    return ExtrudedMesh(UnitIntervalMesh(nx), ny, layer_height=1.0/ny)


@pytest.fixture(scope='module')
def cg1(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module')
def cg2(mesh):
    return FunctionSpace(mesh, "CG", 2)


@pytest.fixture(scope='module')
def dg0(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture(scope='module')
def dg1(mesh):
    return FunctionSpace(mesh, "DG", 1)


@pytest.fixture(scope='module')
def vcg1(mesh):
    return VectorFunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module')
def cg1cg1(cg1):
    return cg1 * cg1


@pytest.fixture(scope='module')
def cg1dg0(cg1, dg0):
    return cg1 * dg0


@pytest.fixture(scope='module')
def cg2dg1(cg2, dg1):
    return cg2 * dg1


@pytest.fixture(scope='module')
def cg1vcg1(mesh, cg1, vcg1):
    return cg1 * vcg1

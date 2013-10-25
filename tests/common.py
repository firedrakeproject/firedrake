import pytest

from firedrake import *


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module')
def cg1(mesh):
    return FunctionSpace(mesh, "Lagrange", 1)


@pytest.fixture(scope='module')
def cg1cg1(mesh):
    CG1 = FunctionSpace(mesh, "CG", 1)
    return CG1 * CG1


@pytest.fixture(scope='module')
def cg1dg0(mesh):
    CG1 = FunctionSpace(mesh, "CG", 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    return CG1 * DG0


@pytest.fixture(scope='module')
def cg2dg1(mesh):
    CG2 = FunctionSpace(mesh, "CG", 2)
    DG1 = FunctionSpace(mesh, "DG", 1)
    return CG2 * DG1

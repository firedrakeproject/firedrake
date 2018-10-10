import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module')
def mesh(request):
    return UnitSquareMesh(2, 2)


@pytest.fixture(scope='module', params=['M2', 'H3', 'B5', 'A5'])
def function_space(request, mesh):
    """Generates zany function spaces for testing SLATE tensor assembly."""
    A5 = FunctionSpace(mesh, "Argyris", 5)
    B5 = FunctionSpace(mesh, "Bell", 5)
    H3 = FunctionSpace(mesh, "Hermite", 3)
    M2 = FunctionSpace(mesh, "Morley", 2)
    return {'A5': A5,
            'B5': B5,
            'H3': H3,
            'M2': M2}[request.param]


@pytest.fixture
def mass(function_space):
    """Generate a generic zany mass form."""
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return inner(u, v) * dx


@pytest.fixture
def mass_matrix(mass):
    return Tensor(mass)


def test_assemble_zany_tensor(mass_matrix):
    M = assemble(mass_matrix)
    assert np.allclose(M.M.values, assemble(mass_matrix.form).M.values, rtol=1e-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

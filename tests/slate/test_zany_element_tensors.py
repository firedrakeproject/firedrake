import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module')
def mesh(request):
    return UnitSquareMesh(2, 2)


@pytest.fixture(scope='module',
                params=[("Morley", 2),
                        ("Hermite", 3),
                        ("Bell", 5),
                        ("Argyris", 5)],
                ids=['M2', 'H3', 'B5', 'A5'])
def function_space(request, mesh):
    """Generates zany function spaces for testing SLATE tensor assembly."""
    return FunctionSpace(mesh, *request.param)


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

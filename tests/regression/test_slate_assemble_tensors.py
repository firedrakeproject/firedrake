import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[interval, triangle, tetrahedron, quadrilateral])
def mesh(request):
    """Generate a mesh according to the cell provided."""
    cell = request.param
    if cell == interval:
        return UnitIntervalMesh(1)
    elif cell == triangle:
        return UnitSquareMesh(1, 1)
    elif cell == tetrahedron:
        return UnitCubeMesh(1, 1, 1)
    elif cell == quadrilateral:
        return UnitSquareMesh(1, 1, quadrilateral=True)
    else:
        raise ValueError("%s cell not recognized" % cell)


@pytest.fixture(scope='module', params=['cg1', 'cg2', 'dg0', 'dg1'])
def function_space(request, mesh):
    """Generates function spaces for testing SLATE tensor assembly."""
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    return {'cg1': cg1,
            'cg2': cg2,
            'dg0': dg0,
            'dg1': dg1}[request.param]


@pytest.fixture
def f(function_space):
    """Generate a Firedrake function given a particular function space."""
    f = Function(function_space)
    f.interpolate(Expression("x[0]"))
    return f


@pytest.fixture
def mass(function_space):
    """Generate a generic mass form."""
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return inner(u, v) * dx


@pytest.fixture
def rank_one_tensor(mass, f):
    return Tensor(action(mass, f))


@pytest.fixture
def rank_two_tensor(mass):
    return Tensor(mass)


def test_tensor_action(mass, f):
    V = assemble(Tensor(mass) * f)
    ref = assemble(action(mass, f))
    assert np.allclose(V.dat.data, ref.dat.data, rtol=1e-14)


def test_assemble_vector(rank_one_tensor):
    V = assemble(rank_one_tensor)
    assert isinstance(V, Function)
    assert np.allclose(V.dat.data, assemble(rank_one_tensor.form).dat.data, rtol=1e-14)


def test_assemble_matrix(rank_two_tensor):
    M = assemble(rank_two_tensor)
    assert np.allclose(M.M.values, assemble(rank_two_tensor.form).M.values, rtol=1e-14)


def test_assemble_vector_into_tensor(mesh):
    V = FunctionSpace(mesh, "DG", 1)
    v = TestFunction(V)
    f = Function(V)
    # Assemble a SLATE tensor into f
    f = assemble(Tensor(v * dx), f)
    # Assemble a different tensor into f
    f = assemble(Tensor(Constant(2) * v * dx), f)
    assert np.allclose(f.dat.data, 2*assemble(Tensor(v * dx)).dat.data, rtol=1e-14)


def test_assemble_matrix_into_tensor(mesh):
    V = FunctionSpace(mesh, "DG", 0)
    u = TestFunction(V)
    v = TrialFunction(V)
    M = assemble(Tensor(u * v * dx))
    # Assemble a different SLATE tensor into M
    M = assemble(Tensor(Constant(2) * u * v * dx), M)
    assert np.allclose(M.M.values, 2*assemble(Tensor(u * v * dx)).M.values, rtol=1e-14)


@pytest.mark.parametrize("fe_family", ("RT",
                                       "BDM",
                                       "N1curl",
                                       "N2curl"))
def test_vector_family_mass(fe_family):
    """Assemble a mass matrix of a vector-valued element
    family defined on simplices. Compare Firedrake assembled
    mass with SLATE assembled mass.
    """
    V = FunctionSpace(UnitSquareMesh(1, 1), fe_family, 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    mass = dot(u, v)*dx

    A = assemble(Tensor(mass))
    ref = assemble(mass)

    assert np.allclose(A.M.values, ref.M.values)


def test_poisson_operator(mesh):
    """Assemble the Poisson operator in SLATE and
    compare with Firedrake.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = inner(grad(u), grad(v))*dx

    P = assemble(Tensor(form))
    ref = assemble(form)

    assert np.allclose(P.M.values, ref.M.values)


def test_helmholtz_operator(mesh):
    """Assemble the (nice) Helmholtz operator in SLATE and
    compare with Firedrake.
    """
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    form = (inner(grad(u), grad(v)) + u*v)*dx

    H = assemble(Tensor(form))
    ref = assemble(form)

    assert np.allclose(H.M.values, ref.M.values)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

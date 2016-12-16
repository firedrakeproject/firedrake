import pytest
import numpy as np
from firedrake import *


@pytest.fixture(scope='module', params=[triangle, tetrahedron, quadrilateral])
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


@pytest.fixture(scope='module', params=['cg1', 'cg2', 'dg0', 'dg1',
                                        'vcg1', 'vcg2', 'tcg1', 'tcg2'])
def function_space(request, mesh):
    """Generates function spaces for testing SLATE tensor assembly."""
    cg1 = FunctionSpace(mesh, "CG", 1)
    cg2 = FunctionSpace(mesh, "CG", 2)
    dg0 = FunctionSpace(mesh, "DG", 0)
    dg1 = FunctionSpace(mesh, "DG", 1)
    vcg1 = VectorFunctionSpace(mesh, "CG", 1)
    vcg2 = VectorFunctionSpace(mesh, "CG", 2)
    tcg1 = TensorFunctionSpace(mesh, "CG", 1)
    tcg2 = TensorFunctionSpace(mesh, "CG", 2)
    return {'cg1': cg1,
            'cg2': cg2,
            'dg0': dg0,
            'dg1': dg1,
            'vcg1': vcg1,
            'vcg2': vcg2,
            'tcg1': tcg1,
            'tcg2': tcg2}[request.param]


@pytest.fixture
def f(function_space):
    """Generate a Firedrake function given a particular function space."""
    f = Function(function_space)
    if function_space.rank >= 1:
        f.interpolate(Expression(("x[0]",) * function_space.dim))
    else:
        f.interpolate(Expression("x[0]"))
    return f


@pytest.fixture
def mass(function_space):
    """Generate a generic mass form."""
    u = TrialFunction(function_space)
    v = TestFunction(function_space)
    return inner(u, v) * dx


def test_tensor_action(mass, f):
    V = assemble(Tensor(mass) * f)
    ref = assemble(action(mass, f))
    assert np.allclose(V.dat.data, ref.dat.data, rtol=1e-14)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

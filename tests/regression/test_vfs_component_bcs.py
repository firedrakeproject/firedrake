import pytest
from firedrake import *
import numpy as np


@pytest.fixture
def m():
    return UnitSquareMesh(4, 4)


@pytest.fixture
def V(m):
    return VectorFunctionSpace(m, 'CG', 1)


@pytest.fixture(params=[0, 1])
def idx(request):
    return request.param


def test_assign_component(V):
    f = Function(V)

    f.assign(Constant((1, 2)))

    assert np.allclose(f.dat.data, [1, 2])

    g = f.sub(0)

    g.assign(10)

    assert np.allclose(g.dat.data, 10)

    assert np.allclose(f.dat.data, [10, 2])

    g = f.sub(1)

    g.assign(3)

    assert np.allclose(f.dat.data, [10, 3])

    assert np.allclose(g.dat.data, 3)


def test_apply_bc_component(V, idx):
    f = Function(V)

    bc = DirichletBC(V.sub(idx), Constant(10), (1, 3))

    bc.apply(f)

    nodes = bc.nodes

    assert np.allclose(f.dat.data[nodes, idx], 10)

    assert np.allclose(f.dat.data[nodes, 1 - idx], 0)


def test_poisson_in_components(V):
    # Solve vector laplacian with different boundary conditions on the
    # x and y components, giving effectively two decoupled Poisson
    # problems in the two components
    g = Function(V)

    f = Constant((0, 0))

    bcs = [DirichletBC(V.sub(0), 0, 1),
           DirichletBC(V.sub(0), 42, 2),
           DirichletBC(V.sub(1), 10, 3),
           DirichletBC(V.sub(1), 15, 4)]

    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx

    L = dot(f, v)*dx

    solve(a == L, g, bcs=bcs)

    expect = Function(V)

    expect.interpolate(Expression(("42*x[0]", "5*x[1] + 10")))
    assert np.allclose(g.dat.data, expect.dat.data)


@pytest.mark.parametrize("nested",
                         [False, True])
def test_poisson_in_mixed_plus_vfs_components(V, nested):
    # Solve five decoupled poisson problems with different boundary
    # conditions in a mixed space composed of two VectorFunctionSpaces
    # and one scalar FunctionSpace.
    # Tests application of boundary conditions to components in mixed
    # spaces.
    Q = FunctionSpace(V.mesh(), "CG", 2)
    W = V*Q*V

    g = Function(W)

    bcs = [DirichletBC(W.sub(0).sub(0), 0, 1),
           DirichletBC(W.sub(0).sub(0), 42, 2),
           DirichletBC(W.sub(0).sub(1), 10, 3),
           DirichletBC(W.sub(0).sub(1), 15, 4),

           DirichletBC(W.sub(1), 4, 1),
           DirichletBC(W.sub(1), 10, 2),

           DirichletBC(W.sub(2).sub(0), -10, 1),
           DirichletBC(W.sub(2).sub(0), 10, 2),
           DirichletBC(W.sub(2).sub(1), 15, 3),
           DirichletBC(W.sub(2).sub(1), 5, 4)]

    u, p, r = TrialFunctions(W)
    v, q, s = TestFunctions(W)

    a = inner(grad(u), grad(v))*dx + \
        inner(grad(r), grad(s))*dx + \
        dot(grad(p), grad(q))*dx

    L = dot(Constant((0, 0)), v)*dx + \
        Constant(0)*q*dx + \
        dot(Constant((0, 0)), s)*dx

    solve(a == L, g, bcs=bcs, nest=nested)

    expected = Function(W)

    u, p, r = expected.split()

    u.interpolate(Expression(("42*x[0]", "5*x[1] + 10")))
    p.interpolate(Expression("6*x[0] + 4"))
    r.interpolate(Expression(("20*x[0] - 10", "-10*x[1] + 15")))

    for actual, expect in zip(g.dat.data, expected.dat.data):
        assert np.allclose(actual, expect)


def test_cant_integrate_subscripted_VFS(V):
    f = Function(V)
    with pytest.raises(NotImplementedError):
        assemble(f.sub(0)*dx)


@pytest.mark.parametrize("cmpt",
                         [-1, 2])
def test_cant_subscript_outside_components(V, cmpt):
    with pytest.raises(AssertionError):
        return V.sub(cmpt)


def test_cant_subscript_3_cmpt(m):
    V = VectorFunctionSpace(m, 'CG', 1, dim=4)
    with pytest.raises(NotImplementedError):
        V.sub(3)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

import pytest
import numpy
from functools import reduce
from operator import add
from firedrake import *


@pytest.fixture
def mesh():
    return UnitSquareMesh(2, 2)


@pytest.fixture
def Vc(mesh):
    return VectorFunctionSpace(mesh, "CG", 1)


@pytest.fixture
def Vd(mesh):
    return VectorFunctionSpace(mesh, "DG", 1)


@pytest.fixture
def Q(mesh):
    return FunctionSpace(mesh, "DG", 0)


@pytest.fixture
def Wd(Vd, Q):
    return Vd*Q


@pytest.fixture
def Wc(Vc, Q):
    return Vc*Q


@pytest.fixture(params=["aij", "nest"])
def mat_type(request):
    return request.param


def test_slate_mixed_vector(Wd):
    u, p = TrialFunctions(Wd)
    v, q = TestFunctions(Wd)

    a = inner(u, v)*dx + inner(p, q)*dx

    A = Tensor(a)

    f = Function(Wd)
    f.sub(0).assign(2)
    f.sub(1).assign(1)
    B = assemble(A.inv * A * AssembledVector(f))

    assert numpy.allclose(B.sub(0).dat.data_ro, 2)
    assert numpy.allclose(B.sub(1).dat.data_ro, 1)

    C = assemble(A * AssembledVector(f))

    expect = assemble(action(a, f))

    for c, e in zip(C.subfunctions, expect.subfunctions):
        assert numpy.allclose(c.dat.data_ro, e.dat.data_ro)


def test_slate_mixed_matrix(Wd, mat_type):
    W2 = Wd*Wd
    a = reduce(add, (inner(a, b)*dx
                     for a, b in zip(TrialFunctions(W2), TestFunctions(W2))))

    A = Tensor(a)

    B = assemble(A.inv * A, mat_type=mat_type)

    for i, j in numpy.ndindex(B.block_shape):
        if i == j:
            assert numpy.allclose(B.M[i, j].values, numpy.eye(W2.sub(i).dim()))
        else:
            assert numpy.allclose(B.M[i, j].values, 0)


@pytest.mark.parametrize("bc_type", ["component", "full"])
def test_slate_mixed_matrix_stokes(Wc, mat_type, bc_type):
    u, p = TrialFunctions(Wc)
    v, q = TestFunctions(Wc)

    a = (inner(sym(grad(u)), sym(grad(v)))
         - inner(p, div(v))
         - inner(div(u), q))*dx
    A = Tensor(a)

    if bc_type == "component":
        bcs = [DirichletBC(Wc.sub(0).sub(1), 0, (1, 2)),
               DirichletBC(Wc.sub(0).sub(0), 0, (3, 4))]
    else:
        bcs = [DirichletBC(Wc.sub(0), 0, (1, 2)),
               DirichletBC(Wc.sub(0), 0, (3, 4))]

    expect = assemble(a, bcs=bcs, mat_type=mat_type)
    actual = assemble(A, bcs=bcs, mat_type=mat_type)

    for i, j in numpy.ndindex(expect.block_shape):
        assert numpy.allclose(expect.M[i, j].values, actual.M[i, j].values)

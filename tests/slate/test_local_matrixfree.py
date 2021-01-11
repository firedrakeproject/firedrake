import pytest
import numpy as np
from firedrake import *


@pytest.fixture
def mymesh():
    return UnitSquareMesh(6, 6)


@pytest.fixture
def V(mymesh):
    dimension = 3
    return FunctionSpace(mymesh, "CG", dimension) 


@pytest.fixture
def A(V): 
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(v), grad(u)) + v * u) * dx
    return Tensor(a)


@pytest.fixture
def A2(V):
    u = TrialFunction(V)
    v = TestFunction(V)
    a = (dot(grad(v), grad(u))) * dx
    return Tensor(a)


@pytest.fixture
def f(V, mymesh):
    f = Function(V)
    x, y= SpatialCoordinate(mymesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(f)


@pytest.fixture(params=["A+A",
                        "A-A",
                        "A+A+A2",
                        "A+A2+A",
                        "A+A2-A",
                        "A-A*A.inv*A"])
def expr(request, A, A2, f):
    if request.param == "A+A":
        return A+A
    elif request.param == "A-A":
        return A-A
    elif request.param == "A+A+A2":
        return A+A+A2
    elif request.param == "A+A2+A":
        return A+A2+A
    elif request.param == "A+A2-A":
        return A+A2-A
    elif request.param == "A-A+A2":
        return A-A+A2
    elif request.param == "A-A*A.inv*A":
        return A-A*A.inv*A


def test_new_slateoptpass(expr, f):
        print("Test is running for expresion " + str(expr))
        tmp = assemble(expr*f, form_compiler_parameters={"optimise_slate": False, "replace_mul_with_action": False})
        tmp_opt = assemble(expr*f, form_compiler_parameters={"optimise_slate": True, "replace_mul_with_action": True})
        assert np.allclose(tmp.dat.data, tmp_opt.dat.data)


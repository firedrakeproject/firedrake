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
def p2():
    return VectorElement("CG", triangle, 2)


@pytest.fixture
def Velo(mymesh, p2):
    return FunctionSpace(mymesh, p2)


@pytest.fixture
def p1():
    return FiniteElement("CG", triangle, 1)


@pytest.fixture
def Pres(mymesh, p1):
    return FunctionSpace(mymesh, p1)


@pytest.fixture
def Mixed(mymesh, p2, p1):
    p2p1 = MixedElement([p2, p1])
    return FunctionSpace(mymesh, p2p1)


@pytest.fixture
def dg(mymesh):
    return FunctionSpace(mymesh, "DG", 1)


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
def A3(mymesh, Mixed, Velo, Pres, dg):
    w = Function(Mixed)
    velocity = as_vector((10, 10))
    velo = Function(Velo).assign(velocity)
    w.sub(0).assign(velo)
    pres = Function(Pres).assign(1)
    w.sub(1).assign(pres)

    T = TrialFunction(dg)
    v = TestFunction(dg)

    h = 2*Circumradius(mymesh)
    n = FacetNormal(mymesh)

    u = split(w)[0]
    un = abs(dot(u('+'), n('+')))
    jump_v = v('+')*n('+') + v('-')*n('-')
    jump_T = T('+')*n('+') + T('-')*n('-')

    return Tensor(-dot(u*T, grad(v))*dx + (dot(u('+'), jump_v)*avg(T))*dS + dot(v, dot(u, n)*T)*ds + 0.5*un*dot(jump_T, jump_v)*dS)


@pytest.fixture
def f(V, mymesh):
    f = Function(V)
    x, y= SpatialCoordinate(mymesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(f)


@pytest.fixture
def f2(dg, mymesh):
    x, y = SpatialCoordinate(mymesh)
    T = Function(dg).interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    return AssembledVector(T)


@pytest.fixture(params=["A+A",
                        "A-A",
                        "A+A+A2",
                        "A+A2+A",
                        "A+A2-A",
                        "A*A.inv",
                        "A.inv*A",
                        "A2*A.inv",
                        "A2.inv*A",
                        "A2*A.inv*A",
                        "A+A-A2*A.inv*A",
                        "advection"])
def expr(request, A, A2, A3, f, f2):
    if request.param == "A+A":
        return (A+A)*f
    elif request.param == "A-A":
        return (A-A)*f
    elif request.param == "A+A+A2":
        return (A+A+A2)*f
    elif request.param == "A+A2+A":
        return (A+A2+A)*f
    elif request.param == "A+A2-A":
        return (A+A2-A)*f
    elif request.param == "A-A+A2":
        return (A-A+A2)*f
    elif request.param == "A*A.inv":
        return (A*A.inv)*f
    elif request.param == "A.inv*A":
        return (A.inv*A)*f
    elif request.param == "A2*A.inv":
        return (A2*A.inv)*f
    elif request.param == "A2.inv*A":
        return (A2.inv*A)*f
    elif request.param == "A2*A.inv*A":
        return (A2*A.inv*A)*f
    elif request.param == "A-A.inv*A":
        return (A-A.inv*A)*f
    elif request.param == "A+A2*A.inv*A":
        return (A+A-A2*A.inv*A)*f
    elif request.param == "advection":
        return A3*f2


def test_new_slateoptpass(expr):
        print("Test is running for expresion " + str(expr))
        tmp = assemble(expr, form_compiler_parameters={"optimise_slate": False, "replace_mul_with_action": False, "visual_debug": False})
        tmp_opt = assemble(expr, form_compiler_parameters={"optimise_slate": True, "replace_mul_with_action": True, "visual_debug": False})
        assert np.allclose(tmp.dat.data, tmp_opt.dat.data, atol=0.0001)


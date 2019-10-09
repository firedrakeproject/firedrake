from firedrake import *
import pytest


@pytest.fixture(params=["scalar", "vector", "tensor"])
def shapify(request):
    if request.param == "scalar":
        return lambda x: x
    elif request.param == "vector":
        return VectorElement
    elif request.param == "tensor":
        return TensorElement
    else:
        raise RuntimeError


@pytest.fixture(params=["x", "both"])
def direction(request):
    return request.param


@pytest.fixture(params=[("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1), ("DG", 2)])
def space(request):
    return request.param


def get_func(fs, d):
    x, y = SpatialCoordinate(fs.mesh())
    if isinstance(fs.ufl_element(), VectorElement):
        return as_vector([x*(1-x), y]) if d == 'x' else as_vector([x*(1-x), y*(1-y)])
    if isinstance(fs.ufl_element(), TensorElement):
        if d == 'x':
            return as_matrix([[x*(1-x), y], [-y, -x*(1-x)]])
        else:
            return as_matrix([[x*(1-x), y*(1-y)], [-y*(1-y), -x*(1-x)]])
    else:
        return x*(1-x) if d == 'x' else x*(1-x)*y*(1-y)


def test_periodic(shapify, direction, space):
    mesh = UnitSquareMesh(3, 4)
    mesh_p = PeriodicUnitSquareMesh(3, 4, direction=direction)
    ele = shapify(FiniteElement(space[0], mesh.ufl_cell(), space[1]))
    V = FunctionSpace(mesh, ele)
    V_p = FunctionSpace(mesh_p, ele)

    f = Function(V)
    f.interpolate(get_func(V, direction))

    f_p = Function(V_p)
    f_p.project(f)

    g_p = Function(V_p)
    g_p.interpolate(get_func(V_p, direction))

    assert errornorm(f_p, g_p) < 1e-8

    g = Function(V)
    g.project(g_p)

    assert errornorm(f, g) < 1e-8

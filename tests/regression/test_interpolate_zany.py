from firedrake import *
import numpy
import pytest


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture(params=[("Argyris", 5), ("Bell", 5), ("HCT", 3),
                        ("Morley", 2), ("Hermite", 3)],
                scope="module",
                ids=lambda x: x[0])
def element(request):
    return request.param


@pytest.fixture(scope="module")
def V(element, mesh):
    return FunctionSpace(mesh, *element)


@pytest.fixture(params=["coefficient", "grad"])
def which(request):
    return request.param


@pytest.fixture
def tolerance(element, which):
    name, _ = element
    if name == "Bell":
        # Not sure why this is worse
        if which == "coefficient":
            return 1e-6
        elif which == "grad":
            return 1e-4
    else:
        return 1e-6


@pytest.fixture
def expect(V, which):
    x, y = SpatialCoordinate(V.mesh())
    expr = (x + y)**(V.ufl_element().degree())
    if which == "coefficient":
        return expr
    elif which == "grad":
        a, b = grad(expr)
        return a + b


def test_interpolate(V, mesh, which, expect, tolerance):
    degree = V.ufl_element().degree()
    Vcg = FunctionSpace(mesh, "P", degree)

    x, y = SpatialCoordinate(mesh)

    f = Function(V)
    g = Function(Vcg)

    expr = (x + y)**degree
    f.project(expr, solver_parameters={"ksp_type": "preonly",
                                       "pc_type": "lu"})

    if which == "coefficient":
        g.interpolate(f)
    elif which == "grad":
        a, b = grad(f)
        g.interpolate(a + b)

    assert numpy.allclose(norm(g - expect), 0, atol=tolerance)

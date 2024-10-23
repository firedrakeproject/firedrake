import pytest
import numpy as np
from firedrake import *
from firedrake.__future__ import *
import finat


@pytest.fixture(params=["square", "cube"])
def mesh_type(request):
    return request.param


@pytest.fixture
def mesh(mesh_type):
    return {
        "square": lambda n: UnitSquareMesh(2 ** n, 2 ** n),
        "cube": lambda n: UnitCubeMesh(2 ** n, 2 ** n, 2 ** n),
    }[mesh_type]


@pytest.fixture
def max_degree(mesh_type):
    return {"square": 6, "cube": 4}[mesh_type]


@pytest.fixture
def interpolation_expr(mesh_type):
    return {
        "square": lambda x, y: (1 + 8 * pi * pi) * cos(x * pi * 2) * cos(y * pi * 2),
        "cube": lambda x, y, z: (1 + 8 * pi * pi)
        * cos(x * pi * 2)
        * cos(y * pi * 2)
        * cos(z * pi * 2),
    }[mesh_type]


def run_interpolation(mesh, expr, p):
    expr = expr(*SpatialCoordinate(mesh))
    V = FunctionSpace(mesh, "KMV", p)
    return errornorm(expr, assemble(interpolate(expr, V)))


def test_interpolation_KMV(mesh, max_degree, interpolation_expr):
    for p in range(1, max_degree):
        errors = [
            run_interpolation(mesh(r), interpolation_expr, p) for r in range(3, 6)
        ]
        errors = np.asarray(errors)
        l2conv = np.log2(errors[:-1] / errors[1:])
        assert (l2conv > p + 0.7).all()


def run_projection(mesh, expr, p):
    V = FunctionSpace(mesh, "KMV", p)
    T = V.finat_element.cell
    u, v = TrialFunction(V), TestFunction(V)
    qr = finat.quadrature.make_quadrature(T, p, "KMV")
    r = Function(V)
    f = assemble(interpolate(expr(*SpatialCoordinate(mesh)), V))
    solve(
        inner(u, v) * dx(scheme=qr) == inner(f, v) * dx(scheme=qr),
        r,
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        },
    )
    return norm(r - f)


def test_projection_KMV(mesh, max_degree, interpolation_expr):
    for p in range(1, max_degree):
        error = run_projection(mesh(1), interpolation_expr, p)
        assert np.abs(error) < 2e-14

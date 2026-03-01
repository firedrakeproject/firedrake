"""Tests for periodic meshes loaded from Gmsh files.

Covers single-direction (x) and doubly-periodic (x+y) meshes in 2D,
and single-direction (x) periodicity in 3D.
"""

from os.path import abspath, dirname, join

import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


@pytest.fixture(params=["p2d", "p2d_xy"])
def periodic_2d_mesh(request):
    return Mesh(join(cwd, "geom", f"{request.param}.msh"))


@pytest.fixture
def periodic_3d_mesh():
    return Mesh(join(cwd, "geom", "p3d.msh"))


def test_periodic_2d_coordinates(periodic_2d_mesh):
    """Mesh uses a DG coordinate element after loading."""
    mesh = periodic_2d_mesh
    elem = mesh.ufl_coordinate_element()
    assert "DG" in str(elem)


def test_periodic_2d_x_solve():
    """Poisson solve on x-periodic rectangle with Dirichlet BCs on top/bottom."""
    mesh = Mesh(join(cwd, "geom", "p2d.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v)) * dx - inner(Constant(1e-1), v) * dx
    bc = DirichletBC(V, Constant(0), [1, 3])
    solve(F == 0, u, bc)

    assert u.dat.data.max() > 0


def test_periodic_2d_xy_solve():
    """Poisson solve on doubly-periodic rectangle with null-space handling."""
    mesh = Mesh(join(cwd, "geom", "p2d_xy.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = sin(2 * pi * x / 0.6) * cos(2 * pi * y / 0.5)

    F = inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    solve(
        F == 0,
        u,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        solver_parameters={"ksp_type": "cg", "pc_type": "gamg"},
    )

    assert u.dat.data.max() > 0


def test_periodic_2d_xy_periodicity():
    """Solution values match across both periodic boundaries."""
    mesh = Mesh(join(cwd, "geom", "p2d_xy.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = sin(2 * pi * x / 0.6) * cos(2 * pi * y / 0.5)

    F = inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    solve(
        F == 0,
        u,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        solver_parameters={"ksp_type": "cg", "pc_type": "gamg"},
    )

    probe_points = np.array(
        [
            [0.0, 0.25],
            [0.6, 0.25],  # left-right
            [0.3, 0.0],
            [0.3, 0.5],  # bottom-top
            [0.0, 0.0],
            [0.6, 0.0],  # corners x
            [0.0, 0.0],
            [0.0, 0.5],  # corners y
        ]
    )
    evaluator = PointEvaluator(mesh, probe_points)
    vals = evaluator.evaluate(u)

    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for ia, ib in pairs:
        assert abs(vals[ia] - vals[ib]) < 1e-10, (
            f"u{tuple(probe_points[ia])}={vals[ia]}  !=  u{tuple(probe_points[ib])}={vals[ib]}"
        )


def test_periodic_3d_solve(periodic_3d_mesh):
    """Poisson solve on x-periodic box with Dirichlet BCs on other faces."""
    mesh = periodic_3d_mesh
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v)) * dx - inner(Constant(1e-1), v) * dx
    bc = DirichletBC(V, Constant(0), [3, 4, 5, 6])
    solve(F == 0, u, bc)

    assert u.dat.data.max() > 0

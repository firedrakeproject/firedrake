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


def test_periodic_2d_coordinates(periodic_2d_mesh):
    """Mesh uses a DG coordinate element after loading."""
    elem = periodic_2d_mesh.ufl_coordinate_element()
    assert "DG" in str(elem)


def _run_periodic_2d_x():
    mesh = Mesh(join(cwd, "geom", "p2d.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v)) * dx - inner(Constant(1e-1), v) * dx
    bc = DirichletBC(V, Constant(0), [1, 3])
    solve(F == 0, u, bc)

    probe_points = np.array([
        [0.0, 0.1], [0.6, 0.1],
        [0.0, 0.25], [0.6, 0.25],
        [0.0, 0.4], [0.6, 0.4],
    ])
    evaluator = PointEvaluator(mesh, probe_points)
    vals = evaluator.evaluate(u)

    for ia, ib in [(0, 1), (2, 3), (4, 5)]:
        assert abs(vals[ia] - vals[ib]) < 1e-10, (
            f"u{tuple(probe_points[ia])}={vals[ia]}  !=  "
            f"u{tuple(probe_points[ib])}={vals[ib]}"
        )


def test_periodic_2d_x_solve():
    """Poisson on x-periodic rectangle; solution matches across x boundary."""
    _run_periodic_2d_x()


@pytest.mark.parallel(nprocs=2)
def test_periodic_2d_x_solve_parallel():
    _run_periodic_2d_x()


def _run_periodic_2d_xy():
    mesh = Mesh(join(cwd, "geom", "p2d_xy.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    x, y = SpatialCoordinate(mesh)
    f = sin(2 * pi * x / 0.6) * cos(2 * pi * y / 0.5)

    F = inner(grad(u), grad(v)) * dx - inner(f, v) * dx
    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    solve(
        F == 0, u,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        solver_parameters={"ksp_type": "cg", "pc_type": "gamg"},
    )

    probe_points = np.array([
        [0.0, 0.25], [0.6, 0.25],
        [0.3, 0.0], [0.3, 0.5],
        [0.0, 0.0], [0.6, 0.0],
        [0.0, 0.0], [0.0, 0.5],
    ])
    evaluator = PointEvaluator(mesh, probe_points)
    vals = evaluator.evaluate(u)

    for ia, ib in [(0, 1), (2, 3), (4, 5), (6, 7)]:
        assert abs(vals[ia] - vals[ib]) < 1e-10, (
            f"u{tuple(probe_points[ia])}={vals[ia]}  !=  "
            f"u{tuple(probe_points[ib])}={vals[ib]}"
        )


def test_periodic_2d_xy_solve():
    """Poisson on doubly-periodic rectangle; solution matches across both boundaries."""
    _run_periodic_2d_xy()


@pytest.mark.parallel(nprocs=2)
def test_periodic_2d_xy_solve_parallel():
    _run_periodic_2d_xy()


def _run_periodic_3d():
    mesh = Mesh(join(cwd, "geom", "p3d.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    v = TestFunction(V)

    F = inner(grad(u), grad(v)) * dx - inner(Constant(1e-1), v) * dx
    bc = DirichletBC(V, Constant(0), [3, 4, 5, 6])
    solve(F == 0, u, bc)

    probe_points = np.array([
        [0.0, 0.5, 0.5], [1.0, 0.5, 0.5],
        [0.0, 0.3, 0.7], [1.0, 0.3, 0.7],
        [0.0, 0.7, 0.3], [1.0, 0.7, 0.3],
    ])
    evaluator = PointEvaluator(mesh, probe_points)
    vals = evaluator.evaluate(u)

    for ia, ib in [(0, 1), (2, 3), (4, 5)]:
        assert abs(vals[ia] - vals[ib]) < 1e-10, (
            f"u{tuple(probe_points[ia])}={vals[ia]}  !=  "
            f"u{tuple(probe_points[ib])}={vals[ib]}"
        )


def test_periodic_3d_solve():
    """Poisson on x-periodic box; solution matches across x boundary."""
    _run_periodic_3d()


@pytest.mark.parallel(nprocs=2)
def test_periodic_3d_solve_parallel():
    _run_periodic_3d()

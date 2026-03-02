"""Tests for periodic meshes loaded from Gmsh files.

Solves the Helmholtz equation -div(grad(u)) + u = f with manufactured
solutions on periodic meshes.  All solutions use cosine factors so
that the boundary data on the periodic boundaries is non-constant,
genuinely exercising the periodic identification.  Covers 2D and 3D
cases.
"""

from math import pi
from os.path import abspath, dirname, join

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


def _run_periodic_helmholtz_2d_x():
    """Helmholtz on x-periodic rectangle [0,0.6]x[0,0.5].

    Manufactured solution u_exact = cos(2*pi*x/0.6) * y*(0.5 - y).
    Periodic in x with non-constant boundary data, zero on y boundaries.
    """
    mesh = Mesh(join(cwd, "geom", "p2d.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    Lx = 0.6
    u_exact_expr = cos(2 * pi * x[0] / Lx) * x[1] * (0.5 - x[1])

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = a(v, u_exact_expr)

    uh = Function(V)
    bc = DirichletBC(V, Constant(0), [1, 3])
    solve(a == L, uh, bcs=bc, solver_parameters={"ksp_type": "cg"})

    assert errornorm(u_exact_expr, uh, "L2") < 0.005


def test_periodic_2d_x_solve():
    _run_periodic_helmholtz_2d_x()


@pytest.mark.parallel(nprocs=2)
def test_periodic_2d_x_solve_parallel():
    _run_periodic_helmholtz_2d_x()


def _run_periodic_helmholtz_2d_xy():
    """Helmholtz on doubly-periodic rectangle [0,0.6]x[0,0.5].

    Trigonometric manufactured solution
    u_exact = cos(2*pi*x/0.6) * cos(2*pi*y/0.5), periodic in both
    x and y with non-constant boundary data.  No boundary conditions
    needed.

    Uses a wider tolerance than the other tests because the
    trigonometric solution requires fine resolution per wavelength.
    """
    mesh = Mesh(join(cwd, "geom", "p2d_xy.msh"))
    V = FunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)

    Lx, Ly = 0.6, 0.5
    u_exact = Function(V)
    u_exact.interpolate(cos(2 * pi * x[0] / Lx) * cos(2 * pi * x[1] / Ly))

    f_coeff = (2 * pi / Lx) ** 2 + (2 * pi / Ly) ** 2 + 1.0
    f = Function(V).assign(f_coeff * u_exact)

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = inner(f, v) * dx

    uh = Function(V)
    solve(a == L, uh, solver_parameters={"ksp_type": "cg"})

    l2err = sqrt(assemble(inner(uh - u_exact, uh - u_exact) * dx))
    l2norm = sqrt(assemble(inner(u_exact, u_exact) * dx))
    assert l2err / l2norm < 0.15


def test_periodic_2d_xy_solve():
    _run_periodic_helmholtz_2d_xy()


@pytest.mark.parallel(nprocs=2)
def test_periodic_2d_xy_solve_parallel():
    _run_periodic_helmholtz_2d_xy()


def _run_periodic_helmholtz_3d():
    """Helmholtz on x-periodic box [0,1]^3.

    Manufactured solution u_exact = cos(2*pi*x) * y*(1-y)*z*(1-z).
    Periodic in x with non-constant boundary data, zero on y/z
    boundaries.
    """
    mesh = Mesh(join(cwd, "geom", "p3d.msh"))
    V = FunctionSpace(mesh, "CG", 4)
    x = SpatialCoordinate(mesh)

    u_exact_expr = cos(2 * pi * x[0]) * x[1] * (1 - x[1]) * x[2] * (1 - x[2])

    u = TrialFunction(V)
    v = TestFunction(V)
    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    L = a(v, u_exact_expr)

    uh = Function(V)
    bc = DirichletBC(V, Constant(0), [3, 4, 5, 6])
    solve(a == L, uh, bcs=bc, solver_parameters={"ksp_type": "cg"})

    assert errornorm(u_exact_expr, uh, "L2") < 1e-4


def test_periodic_3d_solve():
    _run_periodic_helmholtz_3d()


@pytest.mark.parallel(nprocs=2)
def test_periodic_3d_solve_parallel():
    _run_periodic_helmholtz_3d()

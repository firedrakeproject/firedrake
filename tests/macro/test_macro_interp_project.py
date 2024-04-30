import pytest
from firedrake import *


def interp(u, f):
    u.interpolate(f)


def proj(u, f):
    u.project(f)


def proj_bc(u, f):
    u.project(f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


def h1_proj(u, f, bcs=None):
    # compute h1 projection of f into u's function
    # space, store the result in u.
    v = TestFunction(u.function_space())
    F = (inner(grad(u-f), grad(v)) * dx
         + inner(u-f, v) * dx)
    solve(F == 0, u,
          bcs=bcs,
          solver_parameters={"snes_type": "ksponly",
                             "ksp_type": "preonly",
                             "pc_type": "cholesky"})


def h1_proj_bc(u, f):
    h1_proj(u, f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


@pytest.fixture(params=("square", "cube"))
def mesh(request):
    if request.param == "square":
        return UnitSquareMesh(8, 8)
    elif request.param == "cube":
        return UnitCubeMesh(4, 4, 4)


@pytest.mark.parametrize(('variant', 'degree'),
                         [(None, 2),
                          ("integral", 'd'),
                          ('alfeld', 1),
                          ('alfeld', 'd'),
                          ('iso(2)', 2)])
@pytest.mark.parametrize('op', (interp, proj, proj_bc, h1_proj, h1_proj_bc))
def test_projection_scalar_monomial(op, mesh, degree, variant):
    if degree == 'd':
        degree = mesh.geometric_dimension()
    V = FunctionSpace(mesh, "CG", degree=degree, variant=variant)
    u = Function(V)
    x = SpatialCoordinate(mesh)
    f = sum(x) ** degree
    op(u, f)
    error = sqrt(assemble(inner(u - f, u - f) * dx))
    assert error < 1E-7


@pytest.mark.parametrize(('el', 'accorder'),
                         [('HCT', 3),
                          ('HCT-red', 2)])
@pytest.mark.parametrize('op', (proj, h1_proj))
def test_projection_hct(el, accorder, op):
    msh = UnitSquareMesh(1, 1)
    V = FunctionSpace(msh, el, 3)
    u = Function(V)
    x = SpatialCoordinate(msh)
    f = sum(x) ** accorder
    op(u, f)
    error = sqrt(assemble(inner(u - f, u - f) * dx))
    assert error < 1E-9

import pytest
import numpy
from firedrake import *


def interp(u, f):
    u.interpolate(f)
    return assemble(inner(u-f, u-f)*dx)**0.5


def proj(u, f, bcs=None):
    u.project(f, bcs=bcs)
    return assemble(inner(u-f, u-f)*dx)**0.5


def proj_bc(u, f):
    return proj(u, f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


def h1_proj(u, f, bcs=None):
    # compute h1 projection of f into u's function
    # space, store the result in u.
    v = TestFunction(u.function_space())

    d = {H2: grad, H1: grad, HCurl: curl, HDiv: div, HDivDiv: div}[u.ufl_element().sobolev_space]
    F = (inner(d(u-f), d(v)) * dx
         + inner(u-f, v) * dx)
    fcp = {"mode": "vanilla"}
    solve(F == 0, u,
          bcs=bcs,
          solver_parameters={"snes_type": "ksponly",
                             "ksp_type": "preonly",
                             "pc_type": "cholesky"},
          form_compiler_parameters=fcp)
    return assemble(F(u-f), form_compiler_parameters=fcp)**0.5


def h1_proj_bc(u, f):
    return h1_proj(u, f, bcs=DirichletBC(u.function_space(), f, "on_boundary"))


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
    error = op(u, f)
    assert error < 1E-7


@pytest.fixture
def hierarchy(request):
    refine = 1
    mh2 = MeshHierarchy(UnitSquareMesh(5, 5), refine)
    mh3 = MeshHierarchy(UnitCubeMesh(3, 3, 3), refine)
    return {2: mh2, 3: mh3}


def mesh_sizes(mh):
    mesh_size = []
    for msh in mh:
        DG0 = FunctionSpace(msh, "DG", 0)
        h = Function(DG0).interpolate(CellDiameter(msh))
        with h.dat.vec as hvec:
            _, maxh = hvec.max()
        mesh_size.append(maxh)
    return mesh_size


def conv_rates(x, h):
    x = numpy.asarray(x)
    h = numpy.asarray(h)
    return numpy.log2(x[:-1] / x[1:]) / numpy.log2(h[:-1] / h[1:])


def run_convergence(mh, el, deg, convrate, op):
    errors = []
    for msh in mh:
        V = FunctionSpace(msh, el, deg)
        u = Function(V)
        x = SpatialCoordinate(msh)
        f = sum(x) ** (deg+1)
        if u.ufl_shape != ():
            f = f * Constant(numpy.ones(u.ufl_shape))
        errors.append(op(u, f))

    conv = conv_rates(errors, mesh_sizes(mh))
    assert numpy.all(conv > convrate - 0.25)


# Test L2/H1 convergence on C1 elements
@pytest.mark.parametrize(('dim', 'el', 'deg', 'convrate'),
                         [(2, 'PS6', 2, 2),
                          (2, 'PS12', 2, 2),
                          (2, 'HCT', 3, 3),
                          (2, 'HCT-red', 3, 2),
                          ])
@pytest.mark.parametrize('op', (proj, h1_proj))
def test_scalar_convergence(hierarchy, dim, el, deg, convrate, op):
    if op == proj:
        convrate += 1
    run_convergence(hierarchy[dim], el, deg, convrate, op)


# Test L2/H1 convergence on Stokes elements
@pytest.mark.parametrize(('dim', 'el', 'deg', 'convrate'),
                         [(2, 'Alfeld-Sorokina', 2, 2),
                          (3, 'Alfeld-Sorokina', 2, 2),
                          (2, 'Reduced-Arnold-Qin', 2, 1),
                          (3, 'Christiansen-Hu', 1, 1),
                          (2, 'Bernardi-Raugel', 1, 1),
                          (3, 'Bernardi-Raugel', 1, 1),
                          (2, 'Johnson-Mercier', 1, 1),
                          (3, 'Johnson-Mercier', 1, 1),
                          (3, 'Guzman-Neilan 1st kind H1', 1, 1),
                          (3, 'Guzman-Neilan H1(div)', 3, 2),
                          ])
@pytest.mark.parametrize('op', (proj, h1_proj))
def test_piola_convergence(hierarchy, dim, el, deg, convrate, op):
    if op == proj:
        convrate += 1
    run_convergence(hierarchy[dim], el, deg, convrate, op)

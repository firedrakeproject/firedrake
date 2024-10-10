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
                          (3, 'Guzman-Neilan', 3, 1),
                          (3, 'Christiansen-Hu', 1, 1),
                          (2, 'Bernardi-Raugel', 2, 1),
                          (3, 'Bernardi-Raugel', 3, 1),
                          (2, 'Johnson-Mercier', 1, 1),
                          (3, 'Johnson-Mercier', 1, 1),
                          ])
@pytest.mark.parametrize('op', (proj, h1_proj))
def test_piola_convergence(hierarchy, dim, el, deg, convrate, op):
    if op == proj:
        convrate += 1
    run_convergence(hierarchy[dim], el, deg, convrate, op)


# Test that DirichletBC does not set derivative nodes of supersmooth H1 functions
def test_supersmooth_bcs(mesh):
    tdim = mesh.topological_dimension()
    if tdim == 3:
        V = FunctionSpace(mesh, "GNH1div", 3)
    else:
        V = FunctionSpace(mesh, "Alfeld-Sorokina", 2)

    # check that V in H1
    assert V.ufl_element().sobolev_space == H1

    # check that V is supersmooth
    nodes = V.finat_element.fiat_equivalent.dual.nodes
    deriv_nodes = [i for i, node in enumerate(nodes) if len(node.deriv_dict)]
    assert len(deriv_nodes) == tdim + 1

    deriv_ids = V.cell_node_list[:, deriv_nodes]
    u = Function(V)

    CG = FunctionSpace(mesh, "Lagrange", 2)
    RT = FunctionSpace(mesh, "RT", 1)
    for sub in [1, (1, 2), "on_boundary"]:
        bc = DirichletBC(V, 0, sub)

        # check that we have the expected number of bc nodes
        nnodes = len(bc.nodes)
        expected = tdim * len(DirichletBC(CG, 0, sub).nodes)
        if tdim == 3:
            expected += len(DirichletBC(RT, 0, sub).nodes)
        assert nnodes == expected

        # check that the bc does not set the derivative nodes
        u.assign(111)
        u.dat.data_wo[deriv_ids] = 42
        bc.zero(u)
        assert numpy.allclose(u.dat.data_ro[deriv_ids], 42)

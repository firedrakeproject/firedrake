import pytest
from firedrake import *
from os.path import abspath, dirname, join


cwd = abspath(dirname(__file__))


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('mesh_from_file', [False, True])
@pytest.mark.parametrize('family', ["Q", "DQ"])
def test_integral_hex_exterior_facet(mesh_from_file, family):
    if mesh_from_file:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
    else:
        mesh = UnitCubeMesh(2, 3, 5, hexahedral=True)
    V = FunctionSpace(mesh, family, 3)
    x, y, z = SpatialCoordinate(mesh)
    f = Function(V).interpolate(2 * x + 3 * y * y + 4 * z * z * z)
    assert abs(assemble(f * ds) - (2 + 4 + 2 + 5 + 2 + 6)) < 1.e-10


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('mesh_from_file', [False, True])
@pytest.mark.parametrize('family', ["Q", "DQ"])
def test_integral_hex_interior_facet(mesh_from_file, family):
    if mesh_from_file:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
    else:
        mesh = UnitCubeMesh(2, 3, 5, hexahedral=True)
    V = FunctionSpace(mesh, family, 3)
    x, y, z = SpatialCoordinate(mesh)
    f = Function(V).interpolate(2 * x + 3 * y * y + 4 * z * z * z)
    assert assemble((f('+') - f('-'))**2 * dS)**0.5 < 1.e-14


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('mesh_from_file', [False, True])
def test_integral_hex_interior_facet_solve(mesh_from_file):
    if mesh_from_file:
        mesh = Mesh(join(cwd, "..", "meshes", "cube_hex.msh"))
    else:
        mesh = UnitCubeMesh(2, 3, 5, hexahedral=True)
    V = FunctionSpace(mesh, "Q", 1)
    x, y, z = SpatialCoordinate(mesh)
    f = Function(V).interpolate(2 * x + 3 * y + 5 * z)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(u('+'), v('+')) * dS(degree=3)
    L = inner(f('+'), v('-')) * dS(degree=3)
    bc = DirichletBC(V, f, "on_boundary")
    sol = Function(V)
    solve(a == L, sol, bcs=[bc])
    err = assemble((sol - f)**2 * dx)**0.5
    assert err < 1.e-14


def make_nonuniform_box_mesh():
    mesh = BoxMesh(2, 1, 1, 2., 1., 1., hexahedral=True)
    coordV = mesh.coordinates.function_space()
    coords = Function(coordV).assign(mesh.coordinates)
    bc = DirichletBC(coordV.sub(0), 3., 2)
    bc.apply(coords)
    return Mesh(coords)


@pytest.mark.parametrize('GQ_expected', [(CellSize, sqrt(6.)),
                                         (CellVolume, 2.),
                                         (FacetArea, 1.)])
def test_integral_hex_interior_facet_geometric_quantities(GQ_expected):
    GQ, expected = GQ_expected
    mesh = make_nonuniform_box_mesh()
    x, y, z = SpatialCoordinate(mesh)
    e = y('+') * z('-')**2
    E = assemble(e * dS)
    assert abs(E - 1. / 6.) < 1.e-14
    a = GQ(mesh)('-')
    A = assemble(a * dS)
    assert abs(A - expected) < 1.e-14
    EA = assemble(e * a * dS)
    assert abs(EA - E * A) < 1.e-14


def test_integral_hex_interior_facet_facet_avg():
    mesh = make_nonuniform_box_mesh()
    x, y, z = SpatialCoordinate(mesh)
    e = y('+') * z('-')**2
    E = assemble(e * dS)
    assert abs(E - 1. / 6.) < 1.e-14
    a = facet_avg(y('-')**3 * z('+')**4)
    A = assemble(a * dS)
    assert abs(A - 1. / 20.) < 1.e-14
    EA = assemble(e * a * dS)
    assert abs(EA - E * A) < 1.e-14

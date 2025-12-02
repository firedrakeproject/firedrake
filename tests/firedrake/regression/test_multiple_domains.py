from firedrake import *
import pytest
import numpy as np
from functools import partial
from ufl.algorithms.ad import expand_derivatives


@pytest.fixture(params=["interval", "tri", "quad", "tet"])
def typ(request):
    return {"interval": partial(UnitIntervalMesh, 1),
            "tri": UnitTriangleMesh,
            "quad": partial(UnitSquareMesh, 1, 1, quadrilateral=True),
            "tet": UnitTetrahedronMesh}[request.param]


@pytest.fixture
def mesh1(typ):
    return typ()


@pytest.fixture
def mesh2(mesh1):
    new_coords = Function(mesh1.coordinates)
    new_coords *= 0.5
    return Mesh(new_coords)


@pytest.fixture
def mesh3(typ):
    return typ()


def test_mismatching_meshes_indexed_function(mesh1, mesh3):
    V1 = VectorFunctionSpace(mesh1, "CG", 1)
    V2 = FunctionSpace(mesh3, "CG", 1)

    donor = Function(V1)
    target = Function(V2)

    d1, *_ = split(donor)

    with pytest.raises(NotImplementedError):
        project(d1, target)

    with pytest.raises(NotImplementedError):
        assemble(inner(d1, TestFunction(V2))*dx(domain=mesh3))

    with pytest.raises(NotImplementedError):
        assemble(inner(d1, TestFunction(V2))*dx(domain=mesh1))


def test_mismatching_meshes_real_space(mesh1, mesh3):
    V2 = FunctionSpace(mesh3, "CG", 1)

    donor = Function(FunctionSpace(mesh1, "R")).assign(1)
    target = Function(V2)

    with pytest.raises(NotImplementedError):
        project(donor, target)


def test_functional(mesh1, mesh2):
    c = Constant(1)

    val = assemble(c*dx(domain=mesh1))

    cell_volume = mesh1.coordinates.function_space().finat_element.cell.volume()
    assert np.allclose(val, cell_volume)

    val = assemble(c*dx(domain=mesh2))

    assert np.allclose(val, cell_volume * (0.5**mesh1.topological_dimension))

    val = assemble(c*dx(domain=mesh1) + c*dx(domain=mesh2))

    assert np.allclose(val, cell_volume * (1 + 0.5**mesh1.topological_dimension))


def cell_measure(primal, secondary):
    return Measure("dx", primal, intersect_measures=(Measure("dx", secondary),))


@pytest.mark.parametrize("form,expect", [
    (lambda v, mesh1, mesh2: conj(v)*dx(domain=mesh1), lambda vol, dim: vol),
    (lambda v, mesh1, mesh2: conj(v)*cell_measure(mesh2, mesh1), lambda vol, dim: vol*(0.5**dim)),
    (lambda v, mesh1, mesh2: conj(v)*dx(domain=mesh1) + conj(v)*cell_measure(mesh2, mesh1), lambda vol, dim: vol*(1 + 0.5**dim))
], ids=["conj(v)*dx(mesh1)", "conj(v)*dx(mesh2)", "conj(v)*(dx(mesh1) + dx(mesh2)"])
def test_one_form(mesh1, mesh2, form, expect):
    V = FunctionSpace(mesh1, "DG", 0)

    v = TestFunction(V)

    cell_volume = mesh1.coordinates.function_space().finat_element.cell.volume()
    dim = mesh1.topological_dimension

    form = form(v, mesh1, mesh2)
    expect = expect(cell_volume, dim)
    val = assemble(form).dat.data_ro

    assert np.allclose(val, expect)


@pytest.mark.parametrize("form,expect", [
    (lambda u, v, mesh1, mesh2: inner(u, v)*dx(domain=mesh1), lambda vol, dim: vol),
    (lambda u, v, mesh1, mesh2: inner(u, v)*cell_measure(mesh2, mesh1), lambda vol, dim: vol*(0.5**dim)),
    (lambda u, v, mesh1, mesh2: inner(u, v)*dx(domain=mesh1) + inner(u, v)*cell_measure(mesh2, mesh1), lambda vol, dim: vol*(1 + 0.5**dim))
], ids=["inner(u, v)*dx(mesh1)", "inner(u, v)*dx(mesh2)", "inner(u, v)*(dx(mesh1) + dx(mesh2)"])
def test_two_form(mesh1, mesh2, form, expect):
    V = FunctionSpace(mesh1, "DG", 0)

    v = TestFunction(V)
    u = TrialFunction(V)

    cell_volume = mesh1.coordinates.function_space().finat_element.cell.volume()
    dim = mesh1.topological_dimension

    form = form(u, v, mesh1, mesh2)
    expect = expect(cell_volume, dim)
    val = assemble(form).M.values

    assert np.allclose(val, expect)


def test_multi_domain_assemble():
    mesh1 = UnitSquareMesh(7, 7, quadrilateral=True)
    x1, y1 = SpatialCoordinate(mesh1)
    mesh2 = UnitSquareMesh(8, 8)
    x2, y2 = SpatialCoordinate(mesh2)
    V1 = FunctionSpace(mesh1, "Q", 3)
    V2 = FunctionSpace(mesh2, "CG", 2)
    V = V1 * V2

    u1, u2 = TrialFunctions(V)
    v1, v2 = TestFunctions(V)

    a = (
        inner(grad(u1), grad(v1))*dx(domain=mesh1)
        + inner(grad(u2), grad(v2))*dx(domain=mesh2)
    )

    u_exact_expr1 = sin(pi * x1) * sin(pi * y1)
    u_exact_expr2 = x2 * y2 * (1 - x2) * (1 - y2)
    f1 = expand_derivatives(-div(grad(u_exact_expr1)))
    f2 = expand_derivatives(-div(grad(u_exact_expr2)))

    L = (
        inner(f1, v1)*dx(domain=mesh1)
        + inner(f2, v2)*dx(domain=mesh2)
    )

    bc1 = DirichletBC(V.sub(0), 0, "on_boundary")
    bc2 = DirichletBC(V.sub(1), 0, "on_boundary")
    u_sol = Function(V)
    solve(a == L, u_sol, bcs=[bc1, bc2])
    u1_sol, u2_sol = u_sol.subfunctions

    u_exact = Function(V)
    u1_exact, u2_exact = u_exact.subfunctions
    u1_exact.interpolate(u_exact_expr1)
    u2_exact.interpolate(u_exact_expr2)

    err1 = errornorm(u1_exact, u1_sol)
    assert err1 < 1e-5
    err2 = errornorm(u2_exact, u2_sol)
    assert err2 < 1e-5

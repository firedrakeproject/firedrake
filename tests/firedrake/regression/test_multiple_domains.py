from firedrake import *
import pytest
import numpy as np
from functools import partial


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


def test_mismatching_meshes_constant(mesh1, mesh3):
    V2 = FunctionSpace(mesh3, "CG", 1)

    donor = Constant(1, domain=mesh1)
    target = Function(V2)

    with pytest.raises(NotImplementedError):
        project(donor, target)


def test_mismatching_topologies(mesh1, mesh3):
    with pytest.raises(NotImplementedError):
        assemble(1*dx(domain=mesh1) + 2*dx(domain=mesh3))


def test_functional(mesh1, mesh2):
    c = Constant(1)

    val = assemble(c*dx(domain=mesh1))

    cell_volume = mesh1.coordinates.function_space().finat_element.cell.volume()
    assert np.allclose(val, cell_volume)

    val = assemble(c*dx(domain=mesh2))

    assert np.allclose(val, cell_volume * (0.5**mesh1.topological_dimension()))

    val = assemble(c*dx(domain=mesh1) + c*dx(domain=mesh2))

    assert np.allclose(val, cell_volume * (1 + 0.5**mesh1.topological_dimension()))


@pytest.mark.parametrize("form,expect", [
    (lambda v, mesh1, mesh2: conj(v)*dx(domain=mesh1), lambda vol, dim: vol),
    (lambda v, mesh1, mesh2: conj(v)*dx(domain=mesh2), lambda vol, dim: vol*(0.5**dim)),
    (lambda v, mesh1, mesh2: conj(v)*dx(domain=mesh1) + conj(v)*dx(domain=mesh2), lambda vol, dim: vol*(1 + 0.5**dim))
], ids=["conj(v)*dx(mesh1)", "conj(v)*dx(mesh2)", "conj(v)*(dx(mesh1) + dx(mesh2)"])
def test_one_form(mesh1, mesh2, form, expect):
    V = FunctionSpace(mesh1, "DG", 0)

    v = TestFunction(V)

    cell_volume = mesh1.coordinates.function_space().finat_element.cell.volume()
    dim = mesh1.topological_dimension()

    form = form(v, mesh1, mesh2)
    expect = expect(cell_volume, dim)
    val = assemble(form).dat.data_ro

    assert np.allclose(val, expect)


@pytest.mark.parametrize("form,expect", [
    (lambda u, v, mesh1, mesh2: inner(u, v)*dx(domain=mesh1), lambda vol, dim: vol),
    (lambda u, v, mesh1, mesh2: inner(u, v)*dx(domain=mesh2), lambda vol, dim: vol*(0.5**dim)),
    (lambda u, v, mesh1, mesh2: inner(u, v)*dx(domain=mesh1) + inner(u, v)*dx(domain=mesh2), lambda vol, dim: vol*(1 + 0.5**dim))
], ids=["inner(u, v)*dx(mesh1)", "inner(u, v)*dx(mesh2)", "inner(u, v)*(dx(mesh1) + dx(mesh2)"])
def test_two_form(mesh1, mesh2, form, expect):
    V = FunctionSpace(mesh1, "DG", 0)

    v = TestFunction(V)
    u = TrialFunction(V)

    cell_volume = mesh1.coordinates.function_space().finat_element.cell.volume()
    dim = mesh1.topological_dimension()

    form = form(u, v, mesh1, mesh2)
    expect = expect(cell_volume, dim)
    val = assemble(form).M.values

    assert np.allclose(val, expect)

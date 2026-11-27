from firedrake import *
import pytest
import numpy as np
import ufl
from functools import reduce


def test_overlap_subdomain_facets():
    m = UnitSquareMesh(10, 10)

    c = Constant(1)

    f = assemble(c * (ds(1, domain=m) + ds(domain=m)))

    assert np.allclose(f, 5.0)


@pytest.fixture
def mesh():
    from os.path import abspath, dirname, join
    return Mesh(join(abspath(dirname(__file__)), "..",
                "meshes", "cell-sets.msh"))


@pytest.fixture
def V(mesh):
    return FunctionSpace(mesh, "DG", 1)


@pytest.fixture
def u(V):
    return Function(V)


@pytest.fixture(params=["inner(u, v) * dx + inner(u, v) * dx(2) - conj(v) * dx",
                        "inner(u, v) * dx(1) + inner(u, v) * dx(2) + inner(u, v) * dx(2) - conj(v) * dx",
                        "inner(u, v) * dx + inner(u, v) * dx(2) - conj(v) * dx(1) - conj(v) * dx(2)",
                        "inner(u, v) * dx(1) + inner(u, v) * dx(2) + inner(u, v) * dx(2) - conj(v) * dx(1) - conj(v) * dx(2)"])
def form(request, u):
    v = TestFunction(u.function_space())  # noqa
    return eval(request.param)


def test_solve_cell_subdomains(form, u):
    solve(form == 0, u)
    expect = Function(u.function_space())

    expect.interpolate(Constant(1.0), subdomain_id=1)
    expect.interpolate(Constant(0.5), subdomain_id=2)

    assert np.allclose(expect.dat.data_ro, u.dat.data_ro)


@pytest.fixture
def square():
    from os.path import abspath, dirname, join
    return Mesh(join(abspath(dirname(__file__)), "..",
                     "meshes", "square.msh"))


@pytest.fixture(params=[("inner(u, v) * dx", "inner(u, v) * ds(2)"),
                        ("inner(u, v) * dx(1)", "inner(u, v) * ds(2)", "inner(u, v) * dx(1)"),
                        ("inner(u, v) * dx", "inner(u, v) * ds(1)")],
                ids=lambda x: " + ".join(x))
def forms(request):
    return request.param


def test_cell_facet_subdomains(square, forms):
    from operator import add
    V = FunctionSpace(square, "CG", 1)
    v = TestFunction(V)         # noqa
    u = TrialFunction(V)        # noqa
    forms = list(map(eval, forms))
    full = reduce(add, forms)
    full_mat = assemble(full).M.values
    part_mat = reduce(add, map(lambda x: assemble(x).M.values, forms))
    assert np.allclose(part_mat, full_mat)


def test_rename_subdomain_appends_and_parse():
    mesh = UnitSquareMesh(1, 1)
    mesh.rename_subdomain(1, 1, "walls")
    mesh.rename_subdomain(1, [2, 3], "walls")

    assert mesh.topology.subdomain_names[1]["walls"] == [1, 2, 3]
    assert mesh.parse_subdomain_id(1, "walls") == (1, 2, 3)
    assert mesh.parse_subdomain_id(1, ["walls", 4]) == (1, 2, 3, 4)
    assert mesh.parse_subdomain_id(1, "on_boundary") == "on_boundary"

    with pytest.raises(ValueError, match="Subdomain 'missing'"):
        mesh.parse_subdomain_id(1, "missing")


def test_measure_accepts_subdomain_name_with_domain():
    mesh = UnitSquareMesh(2, 2)
    mesh.rename_subdomain(1, [1, 2], "vertical")

    named = assemble(Constant(1) * ds(domain=mesh, subdomain_id="vertical"))
    ufl_named = assemble(Constant(1) * ufl.Measure("ds", domain=mesh, subdomain_id="vertical"))
    target = assemble(Constant(1) * ds(1, domain=mesh)) + assemble(Constant(1) * ds(2, domain=mesh))

    assert named == pytest.approx(target)
    assert ufl_named == pytest.approx(target)


def test_measure_accepts_subdomain_name_with_inferred_domain():
    mesh = UnitSquareMesh(2, 2)
    mesh.rename_subdomain(1, [1, 2], "vertical")
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    f.assign(1)

    named = assemble(f * ds("vertical"))
    target = assemble(f * ds(1)) + assemble(f * ds(2))

    assert named == pytest.approx(target)

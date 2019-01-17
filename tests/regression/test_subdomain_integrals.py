from firedrake import *
import pytest
import numpy as np
from functools import reduce


def test_overlap_subdomain_facets():
    m = UnitSquareMesh(10, 10)

    c = Constant(1, domain=m)

    f = assemble(c*(ds(1) + ds))

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


@pytest.fixture(params=["v*u*dx + v*u*dx(2) - v*dx",
                        "v*u*dx(1) + v*u*dx(2) + v*u*dx(2) - v*dx",
                        "v*u*dx + v*u*dx(2) - v*dx(1) - v*dx(2)",
                        "v*u*dx(1) + v*u*dx(2) + v*u*dx(2) -v*dx(1) - v*dx(2)"])
def form(request, u):
    v = TestFunction(u.function_space())  # noqa
    return eval(request.param)


def test_solve_cell_subdomains(form, u):
    solve(form == 0, u)
    expect = Function(u.function_space())

    mesh = u.ufl_domain()
    expect.interpolate(Constant(1.0), subset=mesh.cell_subset(1))
    expect.interpolate(Constant(0.5), subset=mesh.cell_subset(2))

    assert np.allclose(expect.dat.data_ro, u.dat.data_ro)


@pytest.fixture
def square():
    from os.path import abspath, dirname, join
    return Mesh(join(abspath(dirname(__file__)), "..",
                     "meshes", "square.msh"))


@pytest.fixture(params=[("v*u*dx", "v*u*ds(2)"),
                        ("v*u*dx(1)", "v*u*ds(2)", "v*u*dx(1)"),
                        ("v*u*dx", "v*u*ds(1)")],
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

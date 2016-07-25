from firedrake import *
import pytest
import numpy as np


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


if __name__ == "__main__":
    import os
    pytest.main(os.path.abspath(__file__))

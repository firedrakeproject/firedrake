from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *
import numpy


@pytest.fixture(params=[1, 2, 3])
def degree(request):
    return request.param


@pytest.fixture
def convergence(degree):
    return {1: 1.7,
            2: 2.7,
            3: 3.7}[degree]


@pytest.fixture(params=["north", "south"])
def hemisphere(request):
    return request.param


def run_test(degree, refinements, hemisphere):
    mesh = UnitOctahedralSphereMesh(refinements,
                                    degree=degree,
                                    hemisphere=hemisphere,
                                    smoothing_iterations=50,
                                    z_min=0.9)
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    x, y, z = SpatialCoordinate(mesh)

    L = -(x*y*z)*v*dx

    exact = -(x*y*z)/12.0

    bc = DirichletBC(V, exact, "on_boundary")

    u = Function(V)
    solve(a == L, u, bcs=bc,
          solver_parameters={"ksp_type": "preonly",
                             "pc_type": "lu"})

    uerr = Function(V).interpolate(u - exact)
    f = File('octa'+str(refinements)+'.pvd')
    f.write(uerr)
    uerr.assign(u)
    f.write(uerr)

    return errornorm(u, interpolate(exact, V))


def test_octahedral_hemisphere(degree, hemisphere, convergence):
    errs = numpy.asarray([run_test(degree, r, hemisphere) for r in range(3, 7)])
    l2conv = numpy.log2(errs[:-1] / errs[1:])
    assert (l2conv > convergence).all()

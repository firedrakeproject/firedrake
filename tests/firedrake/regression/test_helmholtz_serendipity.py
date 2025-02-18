"""This demo program solves Helmholtz's equation

  - div grad u(x, y) + u(x,y) = f(x, y)

on the unit square with source f given by

  f(x, y) = (1.0 + 8.0*pi**2)*cos(x[0]*2*pi)*cos(x[1]*2*pi)

and the analytical solution

  u(x, y) = cos(x[0]*2*pi)*cos(x[1]*2*pi)
"""

from os.path import abspath, dirname, join
import numpy as np
import pytest

from firedrake import *

cwd = abspath(dirname(__file__))


def helmholtz(r, quadrilateral=True, degree=2, mesh=None):
    # Create mesh and define function space
    if mesh is None:
        mesh = UnitSquareMesh(2 ** r, 2 ** r, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "S", degree)

    x, y = SpatialCoordinate(mesh)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    uex = cos(x*pi*2)*cos(y*pi*2)
    f = -div(grad(uex)) + uex

    a = (inner(grad(u), grad(v)) + inner(u, v))*dx
    L = inner(f, v)*dx(degree=12)

    params = {"snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    # Compute solution
    sol = Function(V)
    solve(a == L, sol, solver_parameters=params)
    # Error norm
    return sqrt(assemble(dot(sol - uex, sol - uex) * dx)), sol, uex


def run_firedrake_helmholtz():
    diff = np.array([helmholtz(i)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()


def test_firedrake_helmholtz_serial():
    run_firedrake_helmholtz()


@pytest.mark.parallel
def test_firedrake_helmholtz_parallel():
    run_firedrake_helmholtz()


@pytest.mark.parametrize(('testcase', 'convrate'),
                         [((1, (4, 6)), 1.9),
                          ((2, (3, 6)), 2.9),
                          ((3, (2, 4)), 3.9),
                          ((4, (2, 4)), 4.7),
                          ((5, (2, 4)), 5.7)])
def test_firedrake_helmholtz_scalar_convergence_on_quadrilaterals_s(testcase, convrate):
    degree, (start, end) = testcase
    l2err = np.zeros(end - start)
    for ii in [i + start for i in range(len(l2err))]:
        l2err[ii - start] = helmholtz(ii, quadrilateral=True, degree=degree)[0]
    assert (np.array([np.log2(l2err[i]/l2err[i+1]) for i in range(len(l2err)-1)]) > convrate).all()


def run_firedrake_helmholtz_on_quadrilateral_mesh_from_file():
    meshfile = join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh")
    assert helmholtz(None, mesh=Mesh(meshfile))[0] <= 0.01


def test_firedrake_helmholtz_on_quadrilateral_mesh_from_file_serial():
    run_firedrake_helmholtz_on_quadrilateral_mesh_from_file()


@pytest.mark.parallel
def test_firedrake_helmholtz_on_quadrilateral_mesh_from_file_parallel():
    run_firedrake_helmholtz_on_quadrilateral_mesh_from_file()

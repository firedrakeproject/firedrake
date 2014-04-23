from firedrake import *
import pytest
from os.path import abspath, dirname, join


cwd = abspath(dirname(__file__))


# This test solves a mixed formulation of the Poisson equation with
# inhomogeneous Neumann boundary conditions such that the exact
# solution is p(x, y) = x - 0.5.  First on a 2D mesh, and then again
# on a 2D mesh embedded in 3D.
def run_no_manifold():
    mesh = UnitSquareMesh(1, 1)

    V0 = FunctionSpace(mesh, "RT", 2)
    V1 = FunctionSpace(mesh, "DG", 1)

    V = V0 * V1

    bc = DirichletBC(V.sub(0), (-1, 0), (1, 2, 3, 4))

    u, p = TrialFunctions(V)
    v, q = TestFunctions(V)

    a = (dot(u, v) - p*div(v) - div(u)*q)*dx

    f = Function(V1)
    f.assign(0)
    L = -f*q*dx

    up = Function(V)

    null_vec = Function(V)
    null_vec.dat[1].data[:] = 1/sqrt(V1.dof_count)
    nullspace = VectorSpaceBasis(vecs=[null_vec])
    solve(a == L, up, bcs=bc, nullspace=nullspace)
    exact = Function(V1).interpolate(Expression('x[0] - 0.5'))

    u, p = up.split()
    assert errornorm(exact, p, degree_rise=0) < 1e-8


def run_manifold():
    mesh = Mesh(join(cwd, "unitsquare_in_3d.node"), dim=3)

    mesh.init_cell_orientations(Expression(('0', '0', '1')))
    V0 = FunctionSpace(mesh, "RT", 2)
    V1 = FunctionSpace(mesh, "DG", 1)

    V = V0 * V1

    bc = DirichletBC(V.sub(0), (-1, 0, 0), (1, 2, 3, 4))

    u, p = TrialFunctions(V)
    v, q = TestFunctions(V)

    a = (dot(u, v) - p*div(v) - div(u)*q)*dx

    f = Function(V1)
    f.assign(0)
    L = -f*q*dx

    up = Function(V)

    null_vec = Function(V)
    null_vec.dat[1].data[:] = 1/sqrt(V1.dof_count)
    nullspace = VectorSpaceBasis(vecs=[null_vec])
    solve(a == L, up, bcs=bc, nullspace=nullspace)
    exact = Function(V1).interpolate(Expression('x[0] - 0.5'))

    u, p = up.split()
    assert errornorm(exact, p, degree_rise=0) < 1e-8


def test_no_manifold_serial():
    run_no_manifold()


def test_manifold_serial():
    run_manifold()


@pytest.mark.parallel(nprocs=2)
def test_no_manifold_parallel():
    run_no_manifold()


@pytest.mark.parallel(nprocs=2)
def test_manifold_parallel():
    run_manifold()


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

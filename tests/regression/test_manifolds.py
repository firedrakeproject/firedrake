from firedrake import *
import pytest
import numpy as np
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


@pytest.mark.parametrize('space', ["RT", "BDM", "RTCF"])
def test_contravariant_piola_facet_integral(space):
    if space == "RTCF":
        mesh = UnitCubedSphereMesh(refinement_level=2)
    else:
        mesh = UnitIcosahedralSphereMesh(refinement_level=2)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)
    if space == "RTCF":
        C_elt = FiniteElement("CG", "interval", 1)
        D_elt = FiniteElement("DG", "interval", 0)
        V_elt = HDiv(OuterProductElement(C_elt, D_elt)) + HDiv(OuterProductElement(D_elt, C_elt))
        V = FunctionSpace(mesh, V_elt)
    else:
        V = FunctionSpace(mesh, space, 1)
    # Some non-zero function
    u = project(Expression(('x[0]', '-x[1]', '0')), V)
    n = FacetNormal(mesh)

    pos = inner(u('+'), n('+'))*dS
    neg = inner(u('-'), n('-'))*dS

    assert np.allclose(assemble(pos) + assemble(neg), 0)
    assert np.allclose(assemble(pos + neg), 0)


@pytest.mark.parametrize('space', ["N1curl", "N2curl", "RTCE"])
def test_covariant_piola_facet_integral(space):
    if space == "RTCE":
        mesh = UnitCubedSphereMesh(refinement_level=2)
    else:
        mesh = UnitIcosahedralSphereMesh(refinement_level=2)
    global_normal = Expression(("x[0]", "x[1]", "x[2]"))
    mesh.init_cell_orientations(global_normal)
    if space == "RTCE":
        C_elt = FiniteElement("CG", "interval", 1)
        D_elt = FiniteElement("DG", "interval", 0)
        V_elt = HCurl(OuterProductElement(C_elt, D_elt)) + HCurl(OuterProductElement(D_elt, C_elt))
        V = FunctionSpace(mesh, V_elt)
    else:
        V = FunctionSpace(mesh, space, 1)
    # Some non-zero function
    u = project(Expression(('x[0]', '-x[1]', '0')), V)
    n = FacetNormal(mesh)

    pos = inner(u('+'), n('+'))*dS
    neg = inner(u('-'), n('-'))*dS

    assert np.allclose(assemble(pos) + assemble(neg), 0, atol=1e-7)
    assert np.allclose(assemble(pos + neg), 0, atol=1e-7)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

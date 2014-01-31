"""
This demo verifies that the various FEEC operators can reproduce the
Betti numbers of the 2D annulus.
"""
import numpy.linalg as linalg
import numpy
from firedrake import *


def test_betti0():
    mesh = Mesh("annulus.msh")
    V0 = FunctionSpace(mesh, "CG", 1)

#V0 Hodge Laplacian
    u = TrialFunction(V0)
    v = TestFunction(V0)

    L = assemble(inner(nabla_grad(u), nabla_grad(v))*dx)

    u, s, v = linalg.svd(L.M.values)
    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 1)


def test_betti1():
    mesh = Mesh("annulus.msh")
    V0 = FunctionSpace(mesh, "CG", 1)
    V1 = FunctionSpace(mesh, "RT", 1)

    W = V0*V1
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    L = assemble((sigma*tau - inner(rot(tau), u) + inner(rot(sigma), v) +
                  div(u)*div(v))*dx)

    dV0 = V0.dof_count
    dV1 = V1.dof_count

    A = numpy.zeros((dV0+dV1, dV0+dV1))
    A[:dV0, :dV0] = L.M[0, 0].values
    A[:dV0, dV0:dV0+dV1] = L.M[0, 1].values
    A[dV0:dV0+dV1, :dV0] = L.M[1, 0].values
    A[dV0:dV0+dV1, dV0:dV0+dV1] = L.M[1, 1].values

    u, s, v = linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 1)


def test_betti2():
    mesh = Mesh("annulus.msh")
    V1 = FunctionSpace(mesh, "RT", 1)
    V2 = FunctionSpace(mesh, "DG", 0)

    W = V1*V2

    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    L = assemble((inner(sigma, tau) - div(tau)*u + div(sigma)*v)*dx)

    dV1 = V1.dof_count
    dV2 = V2.dof_count

    A = numpy.zeros((dV1+dV2, dV1+dV2))
    A[:dV1, :dV1] = L.M[0, 0].values
    A[:dV1, dV1:dV1+dV2] = L.M[0, 1].values
    A[dV1:dV1+dV2, :dV1] = L.M[1, 0].values
    A[dV1:dV1+dV2, dV1:dV1+dV2] = L.M[1, 1].values

    u, s, v = linalg.svd(A)

    nharmonic = sum(s < 1.0e-5)
    assert(nharmonic == 0)

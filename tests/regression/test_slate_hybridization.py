"""Solve a mixed Helmholtz problem

sigma + grad(u) = 0,
u + div(sigma) = f,

using hybridisation with SLATE performing the forward elimination.
The corresponding finite element variational problem is:

dot(sigma, tau)*dx - u*div(tau)*dx + lambdar*dot(tau, n)*dS = 0
div(sigma)*v*dx + u*v*dx = f*v*dx
gammar*dot(sigma, n)*dS = 0

for all tau, v, and gammar.

This is solved using broken Raviart-Thomas elements of degree k for
(sigma, tau), discontinuous Galerkin elements of degree k - 1
for (u, v), and HDiv-Trace elements of degree k - 1 for (lambdar, gammar).

The forcing function is chosen as:

(1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2),

which reproduces the known analytical solution:

sin(x[0]*pi*2)*sin(x[1]*pi*2)
"""


import pytest
from firedrake import *


@pytest.mark.parametrize("degree", range(1, 3))
def test_slate_hybridization(degree):
    # Create a mesh
    mesh = UnitSquareMesh(8, 8)

    # Create mesh normal
    n = FacetNormal(mesh)

    # Define relevant function spaces
    RT = FiniteElement("RT", triangle, degree)
    BRT = FunctionSpace(mesh, BrokenElement(RT))
    DG = FunctionSpace(mesh, "DG", degree - 1)
    T = FunctionSpace(mesh, "HDiv Trace", degree - 1)

    W = BRT * DG

    # Define the trial and test functions
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    gammar = TestFunction(T)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define finite element variational forms
    Mass_v = dot(sigma, tau) * dx
    Mass_p = u * v * dx
    Div = div(sigma) * v * dx
    Div_adj = div(tau) * u * dx
    local_trace = gammar('+') * dot(sigma, n) * dS
    L = f * v * dx

    # Trace variables are 0 on the boundary of the domain
    # so we remove their contribution on all exterior edges
    bcs = DirichletBC(T, Constant(0.0), (1, 2, 3, 4))

    # Perform the Schur-complement with SLATE expressions
    A = slate.Tensor(Mass_v + Mass_p + Div - Div_adj)
    K = slate.Tensor(local_trace)
    Schur = -K * A.inv * K.T

    F = slate.Tensor(L)
    RHS = -K * A.inv * F

    S = assemble(Schur, bcs=bcs)
    E = assemble(RHS)

    # Solve the reduced system for the Lagrange multipliers
    lambda_sol = Function(T)
    solve(S, lambda_sol, E, solver_parameters={'pc_type': 'lu',
                                               'ksp_type': 'cg'})

    # Currently, SLATE can only assemble one expression at a time.
    # However, we may still write out the pressure and velocity reconstructions
    # in SLATE and obtain our solutions by assembling the SLATE tensor expressions.
    # NOTE: SLATE cannot assemble expressions that result in a tensor with arguments
    # in a mixed function space (yet). Therefore we have to separate the arguments
    # from the mixed space:
    sigma = TrialFunction(BRT)
    tau = TestFunction(BRT)
    u = TrialFunction(DG)
    v = TestFunction(DG)

    A_v = Tensor(dot(sigma, tau) * dx)
    A_p = Tensor(u * v * dx)
    B = Tensor(div(sigma) * v * dx)
    K = Tensor(dot(sigma, n) * gammar('+') * dS)
    F = Tensor(f * v * dx)

    # SLATE expression for pressure recovery:
    u_sol = (B * A_v.inv * B.T + A_p).inv * (F + B * A_v.inv * K.T * lambda_sol)
    u_h = assemble(u_sol)

    # SLATE expression for velocity recovery
    sigma_sol = A_v.inv * (B.T * u_h - K.T * lambda_sol)
    sigma_h = assemble(sigma_sol)

    # Now we compare the hybridized solution to the non-hybridized computation
    V = FunctionSpace(mesh, "RT", degree)
    W2 = V * DG
    sigma, u = TrialFunctions(W2)
    tau, v = TestFunctions(W2)
    ref = Function(W2)
    a = dot(sigma, tau) * dx - div(tau) * u * dx + u * v * dx + v * div(sigma) * dx
    L = f * v * dx
    # Need to slam it with preconditioning due to the system's indefiniteness
    solve(a == L, ref, solver_parameters={'pc_type': 'fieldsplit',
                                          'pc_fieldsplit_type': 'schur',
                                          'ksp_type': 'cg',
                                          'ksp_rtol': 1e-14,
                                          'pc_fieldsplit_schur_fact_type': 'FULL',
                                          'fieldsplit_V_ksp_type': 'cg',
                                          'fieldsplit_P_ksp_type': 'cg'})
    nh_sigma, nh_u = ref.split()

    # Return the L2 error (should be comparable with numerical tolerance
    sigma_err = sqrt(assemble(dot(sigma_h - nh_sigma, sigma_h - nh_sigma) * dx))
    u_err = sqrt(assemble((u_h - nh_u) * (u_h - nh_u) * dx))

    assert sigma_err < 1e-11
    assert u_err < 1e-11


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

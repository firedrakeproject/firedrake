"""Solve a mixed Helmholtz problem

sigma + grad(u) = 0,
u + div(sigma) = f,

using hybridisation with SLATE performing the forward elimination and
backwards reconstructions. The corresponding finite element variational
problem is:

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

from __future__ import absolute_import, print_function, division

import pytest
from firedrake import *


@pytest.mark.parametrize("degree", range(1, 3))
def test_slate_hybridization(degree):
    # Create a mesh
    mesh = UnitSquareMesh(8, 8)
    RT = FunctionSpace(mesh, "RT", degree)
    DG = FunctionSpace(mesh, "DG", degree - 1)
    W = RT * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define the variational forms
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner, pc_type slate.HybridizationPC
    w = Function(W)
    solve(a == L, w,
          solver_parameters={'mat_type': 'matfree',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'trace_ksp_rtol': 1e-8,
                             'trace_pc_type': 'lu',
                             'trace_ksp_type': 'preonly'})
    sigma_h, u_h = w.split()

    # (Non-hybrid) Need to slam it with preconditioning due to the
    # system's indefiniteness
    w2 = Function(W)
    solve(a == L, w2,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'schur',
                             'ksp_type': 'cg',
                             'ksp_rtol': 1e-14,
                             'pc_fieldsplit_schur_fact_type': 'FULL',
                             'fieldsplit_V_ksp_type': 'cg',
                             'fieldsplit_P_ksp_type': 'cg'})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error (should be comparable with numerical tolerance
    sigma_err = sqrt(assemble(dot(sigma_h - nh_sigma,
                                  sigma_h - nh_sigma) * dx))
    u_err = sqrt(assemble((u_h - nh_u) * (u_h - nh_u) * dx))

    assert sigma_err < 1e-11
    assert u_err < 1e-11


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

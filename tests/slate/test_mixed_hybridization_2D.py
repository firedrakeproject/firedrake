"""Solve a mixed Helmholtz problem

sigma + grad(u) = 0,
u + div(sigma) = f,
u = 42 on the exterior boundary,

using hybridisation with SLATE performing the forward elimination and
backwards reconstructions. The finite element variational problem is:

(sigma, tau) - (u, div(tau)) = -<42*tau, n>
(div(sigma), v) + (u, v) = (f, v)

for all tau and v. The forcing function is chosen as:

(1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2),

and the strong boundary condition along the boundary is:

u = 42

which appears in the variational form as the term: -<42*tau, n>
"""
import pytest
from firedrake import *


@pytest.mark.parametrize(("degrees", "hdiv_family", "quadrilateral"),
                         [((1, 0), "RT", False), ((1, 0), "RTCF", True),
                          ((2, 1), "RT", False), ((2, 1), "RTCF", True),
                          ((1, 0), "BDM", False), ((2, 1), "BDM", False),
                          ((2, 1), "BDFM", False)])
def test_mixed_hybridization(degrees, hdiv_family, quadrilateral):
    # Create a mesh
    d_hdiv, d_dg = degrees
    mesh = UnitSquareMesh(4, 4, quadrilateral=quadrilateral)
    HDiv = FunctionSpace(mesh, hdiv_family, d_hdiv)
    DG = FunctionSpace(mesh, "DG", d_dg)
    W = HDiv * DG
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

    # Define the source function
    f = Function(DG)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define the variational forms
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx - 42 * dot(tau, n)*ds

    # Compare hybridized solution with non-hybridized
    # (Hybrid) Python preconditioner
    w = Function(W)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.HybridizationPC',
              'hybridization': {'ksp_type': 'preonly',
                                'pc_type': 'lu',
                                'hdiv_residual': {'ksp_type': 'cg',
                                                  'ksp_rtol': 1e-14},
                                'hdiv_projection': {'ksp_type': 'cg',
                                                    'ksp_rtol': 1e-14}}}
    solve(a == L, w, solver_parameters=params)
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
                             'fieldsplit_0_ksp_type': 'cg',
                             'fieldsplit_1_ksp_type': 'cg'})
    nh_sigma, nh_u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, nh_sigma)
    u_err = errornorm(u_h, nh_u)

    assert sigma_err < 1e-11
    assert u_err < 1e-11


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

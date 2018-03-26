from firedrake import *
import pytest


def test_matrix_free_hybridization():
    # Create a mesh
    mesh = UnitSquareMesh(6, 6)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)

    # Define the source function
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*sin(x*pi*2)*sin(y*pi*2))

    # Define the variational forms
    a = (dot(sigma, tau) - div(tau) * u + u * v + v * div(sigma)) * dx
    L = f * v * dx

    # Compare hybridized solution with non-hybridized
    w = Function(W)

    matfree_params = {'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'ksp_rtol': 1e-8,
                                        'S_mat_type': 'matfree'}}
    solve(a == L, w, solver_parameters=matfree_params)
    sigma_h, u_h = w.split()

    w2 = Function(W)
    aij_params = {'mat_type': 'matfree',
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'cg',
                                    'ksp_rtol': 1e-8,
                                    'S_mat_type': 'aij'}}
    solve(a == L, w2, solver_parameters=aij_params)
    _sigma, _u = w2.split()

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

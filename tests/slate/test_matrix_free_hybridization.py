from firedrake import *


def test_matrix_free_hybridization():
    # Create a mesh
    mesh = UnitSquareMesh(6, 6)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = U * V
    sigma, u = TrialFunctions(W)
    tau, v = TestFunctions(W)
    n = FacetNormal(mesh)

    # Define the source function
    x, y = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate(10*exp(-(pow(x - 0.5, 2) + pow(y - 0.5, 2)) / 0.02))

    # Define the variational forms
    a = (inner(sigma, tau) + inner(u, div(tau)) + inner(div(sigma), v)) * dx
    L = -inner(f, v) * dx + Constant(0.0) * dot(conj(tau), n) * (ds(3) + ds(4))

    # Compare hybridized solution with non-hybridized
    w = Function(W)
    bc1 = DirichletBC(W[0], as_vector([0.0, -sin(5*x)]), 1)
    bc2 = DirichletBC(W[0], as_vector([0.0, sin(5*y)]), 2)
    bcs = [bc1, bc2]

    matfree_params = {'mat_type': 'matfree',
                      'ksp_type': 'preonly',
                      'pc_type': 'python',
                      'pc_python_type': 'firedrake.HybridizationPC',
                      'hybridization': {'ksp_type': 'cg',
                                        'pc_type': 'none',
                                        'ksp_rtol': 1e-8,
                                        'mat_type': 'matfree'}}
    solve(a == L, w, bcs=bcs, solver_parameters=matfree_params)
    sigma_h, u_h = w.subfunctions

    w2 = Function(W)
    aij_params = {'mat_type': 'matfree',
                  'ksp_type': 'preonly',
                  'pc_type': 'python',
                  'pc_python_type': 'firedrake.HybridizationPC',
                  'hybridization': {'ksp_type': 'cg',
                                    'pc_type': 'none',
                                    'ksp_rtol': 1e-8,
                                    'mat_type': 'aij'}}
    solve(a == L, w2, bcs=bcs, solver_parameters=aij_params)
    _sigma, _u = w2.subfunctions

    # Return the L2 error
    sigma_err = errornorm(sigma_h, _sigma)
    u_err = errornorm(u_h, _u)

    assert sigma_err < 1e-8
    assert u_err < 1e-8

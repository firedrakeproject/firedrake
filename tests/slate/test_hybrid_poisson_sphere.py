from __future__ import absolute_import, print_function, division
import pytest
from firedrake import *
import numpy as np


def run_hybrid_poisson_sphere(MeshClass, refinement, hdiv_space):
    """Test hybridizing lowest order mixed methods on a sphere."""
    mesh = MeshClass(refinement_level=refinement)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, hdiv_space, 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = U * V

    f = Function(U)
    f.interpolate(x*y*z)

    u_exact = Function(U).interpolate(x*y*z/12.0)

    u, sigma = TrialFunctions(W)
    v, tau = TestFunctions(W)

    a = (dot(sigma, tau) - div(tau)*u + v*div(sigma))*dx
    L = f*v*dx
    w = Function(W)

    nullsp = MixedVectorSpaceBasis(W, [VectorSpaceBasis(constant=True), W[1]])
    solve(a == L, w,
          nullspace=nullsp,
          solver_parameters={'ksp_type': 'preonly',
                             'mat_type': 'matfree',
                             'pc_type': 'python',
                             'pc_python_type': 'firedrake.HybridizationPC',
                             'hybridization_pc_type': 'lu',
                             'hybridization_ksp_type': 'preonly',
                             'hybridization_projector_tolerance': 1e-14})
    u_h, _ = w.split()
    error = errornorm(u_exact, u_h)
    return error


@pytest.mark.parallel
@pytest.mark.parametrize(('MeshClass', 'hdiv_family'),
                         [(UnitIcosahedralSphereMesh, 'BDM'),
                          (UnitCubedSphereMesh, 'RTCF')])
def test_hybrid_conv_parallel(MeshClass, hdiv_family):
    """Should expect approximately quadratic convergence for lowest order
    mixed method.
    """
    errors = [run_hybrid_poisson_sphere(MeshClass, r, hdiv_family)
              for r in range(2, 5)]
    errors = np.asarray(errors)
    l2conv = np.log2(errors[:-1] / errors[1:])[-1]
    assert l2conv > 1.8


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

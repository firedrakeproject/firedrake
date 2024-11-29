import pytest
from firedrake import *
from firedrake.petsc import DEFAULT_DIRECT_SOLVER_PARAMETERS
import numpy as np


def run_hybrid_poisson_sphere(MeshClass, refinement, hdiv_space):
    """Test hybridizing lowest order mixed methods on a sphere."""
    mesh = MeshClass(refinement_level=refinement)
    mesh.init_cell_orientations(SpatialCoordinate(mesh))
    x, y, z = SpatialCoordinate(mesh)

    V = FunctionSpace(mesh, hdiv_space, 1)
    U = FunctionSpace(mesh, "DG", 0)
    W = U * V

    f = Function(U)
    f.interpolate(x*y*z)

    u_exact = Function(U).interpolate(x*y*z/12.0)

    u, sigma = TrialFunctions(W)
    v, tau = TestFunctions(W)

    a = (inner(sigma, tau) - inner(u, div(tau)) + inner(div(sigma), v))*dx
    L = inner(f, v)*dx
    w = Function(W)

    params = {
        'mat_type': 'matfree',
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': 'firedrake.HybridizationPC',
        'hybridization': {
            'ksp_type': 'preonly',
            'pc_type': 'redundant',
            'redundant_pc_type': 'lu',
            'redundant_pc_factor': DEFAULT_DIRECT_SOLVER_PARAMETERS
        }
    }

    # Provide a callback to construct the trace nullspace
    def nullspace_basis(T):
        return VectorSpaceBasis(constant=True)

    appctx = {'trace_nullspace': nullspace_basis}
    solve(a == L, w, solver_parameters=params, appctx=appctx)
    u_h, _ = w.subfunctions
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

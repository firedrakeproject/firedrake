import pytest
from firedrake import *


@pytest.mark.parametrize('degree', [4, 5])
def test_static_condensation_3d_helmholtz(degree):
    mesh = UnitCubeMesh(4, 4, 4)
    V = FunctionSpace(mesh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    x = SpatialCoordinate(mesh)
    f = Function(V)
    f.interpolate((1+12*pi*pi)*cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2]))

    a = (dot(grad(v), grad(u)) + v*u) * dx
    L = f * v * dx

    # Solve using static condensation
    u_hsc = Function(V)
    params = {'mat_type': 'matfree',
              'ksp_type': 'preonly',
              'pc_type': 'python',
              'pc_python_type': 'firedrake.StaticCondensationPC',
              'static_condensation': {'ksp_type': 'preonly',
                                      'pc_type': 'lu',
                                      'pc_factor_mat_solver_package': 'mumps'}}
    solve(a == L, u_hsc, solver_parameters=params)

    # Solve without static condensation
    u_h = Function(V)
    solve(a == L, u_h, solver_parameters={'ksp_type': 'preonly',
                                          'pc_type': 'lu',
                                          'pc_factor_mat_solver_package': 'mumps'})

    assert errornorm(u_h, u_hsc) < 1.0e-10


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

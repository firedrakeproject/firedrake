import pytest

from firedrake import *


def run_test(x, degree=1, family='CG'):
    m = UnitSquareMesh(2 ** x, 2 ** x)
    V = FunctionSpace(m, family, degree)
    e = Expression('cos(x[0]*pi*2)*sin(x[1]*pi*2)')
    exact = Function(FunctionSpace(m, 'CG', 5))
    exact.interpolate(e)

    # Solve to machine precision.
    ret = project(e, V, solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu'})

    return sqrt(assemble((ret - exact) * (ret - exact) * dx))


@pytest.mark.parametrize(('degree', 'family', 'expected_convergence'), [
    (1, 'CG', 1.8),
    (2, 'CG', 2.6),
    (3, 'CG', 3.8),
    (0, 'DG', 0.8),
    (1, 'DG', 1.8),
    (2, 'DG', 2.8)])
def test_convergence(degree, family, expected_convergence):
    l2_diff = np.array([run_test(x, degree, family) for x in range(2, 7)])
    conv = np.log2(l2_diff[:-1] / l2_diff[1:])
    assert (conv > expected_convergence).all()

if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

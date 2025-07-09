from test_helmholtz import helmholtz
from test_poisson_strong_bcs import run_test
from firedrake import *
import pytest
import numpy as np


@pytest.mark.parametrize(['params', 'degree', 'quadrilateral'],
                         [(p, d, q)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (1, 2, 3)
                          for q in [False, True]])
def test_poisson_analytic(params, degree, quadrilateral):
    assert (run_test(2, degree, parameters=params, quadrilateral=quadrilateral) < 1.e-9)


@pytest.mark.parametrize(['conv_num', 'degree'],
                         [(p, d)
                          for p, d in zip([1.8, 2.8, 3.8], [1, 2, 3])])
def test_helmholtz(mocker, conv_num, degree):
    # mocker.patch('firedrake.mesh.as_cell', return_value=ufc_triangle().to_ufl("triangle"))
    diff = np.array([helmholtz(i, degree=degree)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_num).all()


@pytest.mark.parametrize(['conv_num', 'degree'],
                         [(p, d)
                          for p, d in zip([2.8, 3.8], [2, 3])])
def test_helmholtz_3d(mocker, conv_num, degree):
    diff = np.array([helmholtz(i, degree=degree, mesh=UnitCubeMesh(2 ** i, 2 ** i, 2 ** i))[0] for i in range(2, 4)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_num).all()

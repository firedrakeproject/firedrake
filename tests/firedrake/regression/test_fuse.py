from test_helmholtz import helmholtz
from test_poisson_strong_bcs import run_test
import pytest
import numpy as np



@pytest.mark.parametrize(['params', 'degree', 'quadrilateral'],
                         [(p, d, q)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (1, 2, 3)
                          for q in [False, True]])
def test_poisson_analytic(params, degree, quadrilateral):
    assert (run_test(2, degree, parameters=params, quadrilateral=False) < 1.e-9)

def test_helmholtz():
    diff = np.array([helmholtz(i)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()
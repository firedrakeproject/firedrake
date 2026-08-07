from firedrake import *
from firedrake.utils import single_mode
import numpy as np
import pytest


@pytest.mark.parametrize('i', range(3, 11))
def test_circumference(i):
    eps = 1e-12
    mesh = CircleManifoldMesh(i, radius=i*i)  # noqa: need for dx
    f = Constant(1.0)
    # 2 * radius * sin(pi/i) * number of sides
    circumference = 2*i*i*np.sin(np.pi/i)*i
    # fp32: absolute round-off scales with the (large) circumference, so use a relative tolerance
    assert np.abs(assemble(f*dx(domain=mesh)) - circumference) < (1e-5 * circumference if single_mode else eps)


def test_pi():
    len = 10
    errors = np.zeros(len)
    for i in range(2, 2+len):
        mesh = CircleManifoldMesh(2**i)  # noqa: need for dx
        f = Constant(1.0)
        errors[i-2] = np.abs(assemble(f*dx(domain=mesh)) - 2*np.pi)

    # circumference converges quadratically to 2*pi
    for i in range(len-1):
        assert ln(errors[i]/errors[i+1])/ln(2) > 1.95

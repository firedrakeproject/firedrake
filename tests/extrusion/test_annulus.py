from firedrake import *
import numpy as np
import pytest


def test_pi():
    len = 7
    errors = np.zeros(len)
    for i in range(2, 2+len):
        m = CircleManifoldMesh(2**i)
        mesh = ExtrudedMesh(m, layers=2**i, layer_height=1.0/(2**i), extrusion_type="radial")
        fs = FunctionSpace(mesh, "DG", 0)
        f = Function(fs).assign(1)
        # area is pi*(2^2) - pi*(1^2) = 3*pi
        errors[i-2] = np.abs(assemble(f*dx) - 3*np.pi)

    # area converges linearly to 3*pi
    for i in range(len-1):
        assert ln(errors[i]/errors[i+1])/ln(2) > 0.95


if __name__ == '__main__':
    pytest.main(os.path.abspath(__file__))

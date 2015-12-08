import numpy as np
import pytest

from firedrake import *


def test_box_1d():
    mesh = UnitIntervalMesh(4)
    x = mesh.coordinates
    f = Function(FunctionSpace(mesh, 'CG', 1))
    f.interpolate(Expression("x[0]"))

    # A caching bug might cause to recall the following value at a later
    # assembly.  We keep this line to have that case tested.
    assert np.allclose(0.5, assemble(f*dx))

    sd = make_subdomain_data(x[0] < 0.5)
    assert np.allclose(0.125, assemble(f*dx(subdomain_data=sd)))

    sd = make_subdomain_data(x[0] > 0.5)
    assert np.allclose(0.375, assemble(f*dx(subdomain_data=sd)))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

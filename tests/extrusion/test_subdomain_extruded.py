import numpy as np
import pytest

from firedrake import *


def test_base_box_3d():
    m = UnitSquareMesh(10, 10, quadrilateral=True)
    mesh = ExtrudedMesh(m, layers=8)
    f = Function(FunctionSpace(mesh, 'CG', 1))
    f.interpolate(Expression("x[0] + 2*x[1] + 4*x[2]"))

    # A caching bug might cause to recall the following value at a later
    # assembly.  We keep this line to have that case tested.
    assert np.allclose(3.5, assemble(f*dx))

    x = m.coordinates
    sd = make_subdomain_data(And(And(0.2 <= x[0], x[0] <= 0.5),
                                 And(0.3 <= x[1], x[1] <= 0.7)))

    # TEMPORARY HACK:
    # Retarget base subdomain into columns of the extruded mesh.
    sd = op2.Subset(mesh.cell_set, sd.indices)
    assert np.allclose(0.402, assemble(f*dx(subdomain_data=sd)))


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

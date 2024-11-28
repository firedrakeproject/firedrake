import numpy as np
import pytest

from firedrake import *


def run_base_box_3d():
    m = UnitSquareMesh(10, 10, quadrilateral=True)
    mesh = ExtrudedMesh(m, layers=8)
    f = Function(FunctionSpace(mesh, 'CG', 1))
    x, y, z = SpatialCoordinate(mesh)
    f.interpolate(x + 2*y + 4*z)

    # A caching bug might cause to recall the following value at a later
    # assembly.  We keep this line to have that case tested.
    assert np.allclose(3.5, assemble(f*dx))

    x = m.coordinates
    sd = SubDomainData(And(And(0.2 <= real(x[0]), real(x[0]) <= 0.5),
                           And(0.3 <= real(x[1]), real(x[1]) <= 0.7)))

    # TEMPORARY HACK:
    # Retarget base subdomain into columns of the extruded mesh.
    sd = op2.Subset(mesh.cell_set, sd.indices)
    assert np.allclose(0.402, assemble(f*dx(subdomain_data=sd)))


def test_base_box_3d():
    run_base_box_3d()


@pytest.mark.parallel(nprocs=3)
def test_base_box_3d_parallel():
    run_base_box_3d()

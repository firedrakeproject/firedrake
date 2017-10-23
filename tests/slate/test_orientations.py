import pytest
from firedrake import *
import numpy as np


@pytest.mark.parametrize(('Mesh', 'hdiv_space', 'degree'),
                         [(UnitIcosahedralSphereMesh, 'RT', 1),
                          (UnitIcosahedralSphereMesh, 'BDM', 1),
                          (UnitCubedSphereMesh, 'RTCF', 1)])
def test_tensors_on_sphere(Mesh, hdiv_space, degree):
    mesh = Mesh(refinement_level=2)
    mesh.init_cell_orientations(Expression(("x[0]", "x[1]", "x[2]")))
    n = FacetNormal(mesh)
    V = FunctionSpace(mesh, hdiv_space, degree)

    x, y, z = SpatialCoordinate(mesh)
    sigma = TrialFunction(V)
    tau = TestFunction(V)
    f = Function(FunctionSpace(mesh, "DG", 0))
    f.interpolate(2*x + 2*y + 2*z)

    mass = f*dot(sigma, tau)*dx
    flux_dS = jump(f*tau, n=n)*dS
    flux_ds = f*dot(tau, n)*ds

    A = assemble(Tensor(mass))
    B = assemble(Tensor(dot(f*tau, n)*dS))
    C = assemble(Tensor(flux_ds))

    refA = assemble(mass)
    refB = assemble(flux_dS)
    refC = assemble(flux_ds)

    assert np.allclose(A.M.values, refA.M.values, rtol=1e-13)
    assert np.allclose(B.dat.data, refB.dat.data, rtol=1e-13)
    assert np.allclose(C.dat.data, refC.dat.data, rtol=1e-13)


if __name__ == '__main__':
    import os
    pytest.main(os.path.abspath(__file__))

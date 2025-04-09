import pytest
from firedrake import *


@pytest.mark.parallel(nprocs=3)
def test_meshes_volume_annulusmesh():
    R = 4
    r = 3
    mesh = AnnulusMesh(R, r, nr=8, nt=128)
    vol = assemble(Constant(1., domain=mesh) * dx)
    exact = pi * (R**2 - r**2)
    assert abs(vol - exact) / exact < .0005


@pytest.mark.parallel(nprocs=3)
def test_meshes_volume_solidtorusmesh():
    R = 7  # major radius
    r = 3  # minor radius
    mesh = SolidTorusMesh(R, r, nR=128, refinement_level=6)
    vol = assemble(Constant(1., domain=mesh) * dx)
    exact = (pi * r * r) * (2 * pi * R)
    assert abs(vol - exact) / exact < .0005

# Test that integrals over the surface of a sphere do the right thing
import pytest
from tests.common import longtest
from firedrake import *


def integrate_sphere_area(radius=1, refinement=2):
    mesh = IcosahedralSphereMesh(radius=radius, refinement_level=refinement)
    fs = FunctionSpace(mesh, "CG", 1, name="fs")

    f = Function(fs)

    f.assign(1)

    exact = 4*pi*radius**2
    return np.abs(assemble(f * dx) - exact) / exact


@longtest
@pytest.mark.parametrize(('radius', 'refinement', 'error'),
                         [(1, 2, 0.02),
                          (10, 2, 0.02),
                          (1, 5, 0.0003),
                          (10, 5, 0.0003)])
def test_surface_area_sphere(radius, refinement, error):
    assert integrate_sphere_area(radius=radius, refinement=refinement) < error

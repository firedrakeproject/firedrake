# Test that integrals over the surface of a sphere do the right thing
import pytest
import numpy as np
from firedrake import *


def integrate_sphere_area(MeshClass, radius=1, refinement=2):
    mesh = MeshClass(radius=radius, refinement_level=refinement)
    fs = FunctionSpace(mesh, "CG", 1, name="fs")

    f = Function(fs)

    f.assign(1)

    exact = 4*pi*radius**2
    return np.abs(assemble(f * dx) - exact) / exact


@pytest.mark.parametrize(('radius', 'refinement', 'error'),
                         [(1, 2, 0.02),
                          (10, 2, 0.02),
                          (1, 5, 0.0004),
                          (10, 5, 0.0004)])
def test_surface_area_icosahedral_sphere(radius, refinement, error):
    assert integrate_sphere_area(IcosahedralSphereMesh, radius=radius, refinement=refinement) < error


@pytest.mark.parametrize(('radius', 'refinement', 'error'),
                         [(1, 2, 0.04),
                          (10, 2, 0.04),
                          (1, 5, 0.0006),
                          (10, 5, 0.0006)])
def test_surface_area_cubed_sphere(radius, refinement, error):
    assert integrate_sphere_area(CubedSphereMesh, radius=radius, refinement=refinement) < error

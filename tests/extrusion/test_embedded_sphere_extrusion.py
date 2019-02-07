import pytest
import numpy as np
from firedrake import *


def integrate_spherical_annulus_volume(MeshClass, radius=1000, refinement=2):
    m = MeshClass(radius=radius, refinement_level=refinement)
    layers = 10
    layer_height = 1.0 / (radius * layers)

    mesh = ExtrudedMesh(m, layers, layer_height=layer_height, extrusion_type='radial')

    fs = FunctionSpace(mesh, 'CG', 1, name="fs")

    f = Function(fs)

    f.assign(1)

    exact = 4 * pi * ((radius + 1.0/radius)**3 - radius**3) / 3
    return np.abs(assemble(f * dx) - exact) / exact


@pytest.mark.parametrize(('radius', 'refinement', 'error'),
                         [(1000, 2, 0.04),
                          (10000, 2, 0.04),
                          (1000, 4, 0.0022),
                          (10000, 4, 0.0022)])
def test_volume_icosahedral_spherical_annulus(radius, refinement, error):
    assert integrate_spherical_annulus_volume(IcosahedralSphereMesh, radius=radius, refinement=refinement) < error


@pytest.mark.parametrize(('radius', 'refinement', 'error'),
                         [(1000, 3, 0.04),
                          (10000, 3, 0.04),
                          (1000, 5, 0.0011),
                          (10000, 5, 0.0011)])
def test_volume_cubed_spherical_annulus(radius, refinement, error):
    assert integrate_spherical_annulus_volume(CubedSphereMesh, radius=radius, refinement=refinement) < error

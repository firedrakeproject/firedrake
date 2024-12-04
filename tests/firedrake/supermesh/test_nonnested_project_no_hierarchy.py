import pytest
import numpy
from firedrake import *
from itertools import product
import weakref


spaces = [("CG", 1), ("CG", 2), ("DG", 0), ("DG", 1)]


@pytest.fixture(params=[(c, f) for c, f in product(spaces, spaces) if c[1] <= f[1]],
                ids=lambda x: "%s%s-%s%s" % (*x[0], *x[1]))
def pairs(request):
    return request.param


@pytest.fixture
def coarse(pairs):
    return pairs[0]


@pytest.fixture
def fine(pairs):
    return pairs[1]


@pytest.mark.parallel(nprocs=3)
def test_project_parallel(coarse, fine):
    # We want all processes in parallel to see the full mesh.
    distribution_parameters = {"partition": True,
                               "overlap_type": (DistributedMeshOverlapType.VERTEX, 10)}
    cmesh = RectangleMesh(2, 2, 1, 1, diagonal="left",
                          distribution_parameters=distribution_parameters)

    fmesh = RectangleMesh(5, 5, 1, 1, diagonal="right",
                          distribution_parameters=distribution_parameters)

    # Mark the two meshes as having a compatible parallel layout
    fmesh._parallel_compatible = {weakref.ref(cmesh)}

    Vc = FunctionSpace(cmesh, *coarse)
    Vf = FunctionSpace(fmesh, *fine)

    c = Function(Vc)
    c.interpolate(SpatialCoordinate(cmesh)**2)
    expect = assemble(c*dx)

    actual = project(c, Vf)

    actual = assemble(actual*dx)

    assert numpy.allclose(expect, actual)

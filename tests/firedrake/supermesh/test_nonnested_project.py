import pytest
import numpy
from firedrake import *
from firedrake.utils import IntType
from itertools import product
import weakref


@pytest.fixture(scope="module")
def hierarchy():

    # We want all processes in parallel to see the full mesh.
    distribution_parameters = {"partition": True,
                               "overlap_type": (DistributedMeshOverlapType.VERTEX, 10)}
    mesh = RectangleMesh(2, 2, 1, 1, diagonal="left",
                         distribution_parameters=distribution_parameters)

    mesh2 = RectangleMesh(5, 5, 1, 1, diagonal="right",
                          distribution_parameters=distribution_parameters)

    mesh.init()
    mesh2.init()
    coarse_to_fine = numpy.tile(numpy.arange(mesh2.num_cells(), dtype=IntType),
                                (mesh.num_cells(), 1))

    fine_to_coarse = numpy.tile(numpy.arange(mesh.num_cells(), dtype=IntType),
                                (mesh2.num_cells(), 1))

    hierarchy = HierarchyBase((mesh, mesh2), [coarse_to_fine], [None, fine_to_coarse],
                              nested=False)
    return hierarchy


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


def test_project(hierarchy, coarse, fine):
    cmesh, fmesh = hierarchy

    Vc = FunctionSpace(cmesh, *coarse)
    Vf = FunctionSpace(fmesh, *fine)

    c = Function(Vc)
    c.interpolate(SpatialCoordinate(cmesh)**2)
    expect = assemble(c*dx)

    actual = project(c, Vf)

    actual = assemble(actual*dx)

    assert numpy.allclose(expect, actual)


@pytest.mark.parallel(nprocs=3)
def test_project_parallel(hierarchy, coarse, fine):
    cmesh, fmesh = hierarchy

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

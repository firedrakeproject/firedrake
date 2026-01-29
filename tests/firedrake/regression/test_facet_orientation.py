"""Tests if nodes shared between cells (e.g. nodes on vertices, facets etc.)
are interpreted consistently on CG spaces by the cells that share them.
"""

from os.path import abspath, dirname, join
from functools import partial
import pytest
import numpy as np
from firedrake import *

cwd = abspath(dirname(__file__))

meshes = [
    partial(UnitSquareMesh, 5, 5, quadrilateral=False),
    partial(UnitSquareMesh, 5, 5, quadrilateral=True),
    partial(UnitIcosahedralSphereMesh, 2),
    partial(UnitCubedSphereMesh, 3),
    partial(Mesh, join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh")),
]


def run_consistent_facet_orientation(mesh_thunk, variant="equispaced", **kwargs):
    mesh = mesh_thunk(**kwargs)
    x = SpatialCoordinate(mesh)
    degree = 3
    V = FunctionSpace(mesh, "CG", degree, variant=variant)  # continuous space
    if variant == "equispaced":
        W = FunctionSpace(mesh, "DG", degree, variant=variant)  # discontinuous space
    else:
        W = FunctionSpace(mesh, BrokenElement(V.ufl_element()))  # discontinuous space

    Q = FunctionSpace(mesh, "DG", 0)  # result space

    expression = x[0]*(x[0] + sqrt(2.0)) + x[1]
    f = Function(V).interpolate(expression)
    g = Function(W).interpolate(expression)

    q = Function(Q)

    domain = "{[i]: 0 <= i < C.dofs}"
    instructions = """
    for i
        R[0, 0] = fmax(real(R[0, 0]), abs(C[i, 0] - D[i, 0]))
    end
    """
    par_loop((domain, instructions), dx, {'C': (f, READ), 'D': (g, READ), 'R': (q, RW)})

    assert np.allclose(q.dat.data, 0.0)


@pytest.mark.parametrize('mesh_thunk', meshes)
def test_consistent_facet_orientation(mesh_thunk):
    run_consistent_facet_orientation(mesh_thunk)


@pytest.mark.parallel(nprocs=2)
@pytest.mark.parametrize('variant', ("equispaced", "integral"))
@pytest.mark.parametrize('mesh_thunk', meshes)
def test_consistent_facet_orientation_parallel(mesh_thunk, variant):
    dp = {"overlap_type": (DistributedMeshOverlapType.NONE, 0)}
    run_consistent_facet_orientation(mesh_thunk, variant=variant, distribution_parameters=dp)

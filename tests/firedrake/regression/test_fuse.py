from test_helmholtz import helmholtz
from test_poisson_strong_bcs import run_test
from test_steady_advection_3D import run_near_to_far
from fuse.cells import ufc_triangle
from firedrake.utils import IntType
from firedrake import *
import os
import pytest
import numpy as np
import firedrake.cython.dmcommon as dmcommon

@pytest.fixture(scope="module", autouse=True)
def set_env():
    os.environ["FIREDRAKE_USE_FUSE"] = "True"


@pytest.mark.parametrize(['params', 'degree', 'quadrilateral'],
                         [(p, d, q)
                          for p in [{}, {'snes_type': 'ksponly', 'ksp_type': 'preonly', 'pc_type': 'lu'}]
                          for d in (1, 2, 3)
                          for q in [False, True]])
def test_poisson_analytic(params, degree, quadrilateral):
    assert (run_test(2, degree, parameters=params, quadrilateral=quadrilateral) < 1.e-9)


@pytest.mark.parametrize(['conv_num', 'degree'],
                         [(p, d)
                          for p, d in zip([1.8, 2.8, 3.8], [1, 2, 3])])
def test_helmholtz(mocker, conv_num, degree):
    # mocker.patch('firedrake.mesh.as_cell', return_value=ufc_triangle().to_ufl("triangle"))
    diff = np.array([helmholtz(i, degree=degree)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_num).all()


@pytest.mark.parametrize(['conv_num', 'degree'],
                         [(p, d)
                          for p, d in zip([ 2.8, 3.8], [2, 3])])
def test_helmholtz_3d(mocker, conv_num, degree):
    diff = np.array([helmholtz(i, degree=degree, mesh=UnitCubeMesh(2 ** i, 2 ** i, 2 ** i))[0] for i in range(2, 4)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_num).all()


def test_reorder_closure():
    mesh = UnitTetrahedronMesh()
    mesh.init()
    plex = mesh.topology_dm
    tdim = plex.getDimension()

    # Cell numbering and global vertex numbering
    cell_numbering = mesh._cell_numbering
    vertex_numbering = mesh._vertex_numbering.createGlobalSection(plex.getPointSF())

    cell = mesh.ufl_cell()
    plex.setName('firedrake_default_topology_fuse')
    #   TODO find better way of branching here
    topology = cell.to_fiat().topology
    closureSize = sum([len(ents) for _, ents in topology.items()])
    verts_per_entity = np.zeros(len(topology), dtype=IntType)
    entity_per_cell = np.zeros(len(topology), dtype=IntType)
    for d, ents in topology.items():
        verts_per_entity[d] = len(ents[0])
        entity_per_cell[d] = len(ents)
    print(entity_per_cell)
    print(dmcommon.derive_closure_ordering(plex, cell_numbering, closureSize, entity_per_cell, verts_per_entity))


from firedrake import *
from firedrake.mg.utils import get_level


def test_submesh_hierarchy():
    nx = 4
    refine = 2
    mesh = UnitSquareMesh(nx, nx)
    mh = MeshHierarchy(mesh, refine)

    submesh = Submesh(mesh, mesh.topological_dimension-1, 3)
    submh = MeshHierarchy(submesh, refine)

    fmesh = mh[-1]
    fsubmesh = submh[-1]
    assert fsubmesh.submesh_parent is not None
    assert fsubmesh.submesh_parent is fmesh


def test_mesh_sequence_hierarchy():
    nx = 4
    refine = 2
    mesh = UnitSquareMesh(nx, nx)
    submesh = Submesh(mesh, mesh.topological_dimension-1, 3)

    base = MeshSequence([mesh, submesh])
    mh = MeshHierarchy(base, refine)
    fmesh, fsubmesh = mh[-1]

    assert fmesh == get_level(mesh)[0][-1]
    assert fsubmesh == get_level(submesh)[0][-1]

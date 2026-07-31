import pytest
from firedrake import *


def test_submesh_parent():
    mesh = UnitIntervalMesh(2)

    M = FunctionSpace(mesh, "DG", 0)
    m = Function(M)
    m.dat.data[0] = 1

    cell_marker = 100
    parent = RelabeledMesh(mesh, [m], [cell_marker])

    submesh = Submesh(parent, parent.topological_dimension, cell_marker)
    assert submesh.topology.submesh_parent is parent.topology
    assert submesh.submesh_parent is parent


def test_submesh_redistribute_codim():
    # The entities of a submesh of non-zero co-dimension are not the entities
    # of its parent, so they can not inherit its orientations.
    mesh = UnitSquareMesh(2, 2)
    with pytest.raises(NotImplementedError):
        Submesh(mesh, subdomain_id="on_boundary", redistribute=True)

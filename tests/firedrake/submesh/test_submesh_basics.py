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

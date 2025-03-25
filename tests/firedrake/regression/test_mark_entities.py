import pytest
from firedrake import *


@pytest.mark.parallel(nprocs=8)
def test_mark_entities_mark_points_with_function_array():
    # Mark cells in f0.
    # +---+---+            +---+
    # | \ | \ |  mark      |   |
    # +---+---+ ----->     +---+
    # | \ | \ |
    # +---+---+
    # Mark facets in f1.
    # +---+---+
    # | \ | \ |  mark        \
    # +---+---+ -----> +---+
    # | \ | \ |
    # +---+---+
    my_cell_label = 999
    my_facet_label = 4

    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)
    V0 = FunctionSpace(mesh, "DP", 0)
    f0 = Function(V0).interpolate(conditional(And(And(x > .6, x < .9),
                                                  And(y > .6, y < .9)), 1., 0.))
    V1 = FunctionSpace(mesh, "HDiv Trace", 0)
    f1 = Function(V1).interpolate(conditional(Or(And(x < .4,
                                                     And(y > .4, y < .6)),
                                                 And(And(x > .6, x < .9),
                                                     And(y > .6, y < .9))), 1., 0.))
    mesh = RelabeledMesh(mesh,
                         [f0, f1],
                         [my_cell_label, my_facet_label])
    # Check integrals.
    v = assemble(Constant(1, domain=mesh) * dx)
    assert abs(v - 1.0) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * dx(my_cell_label))
    assert abs(v - .25) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * dS)
    assert abs(v - (4 * .5 + 4 * .5 * sqrt(2))) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * dS(my_facet_label))
    assert abs(v - (1 * .5 + 1 * .5 * sqrt(2))) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * dS(unmarked))
    assert abs(v - (3 * .5 + 3 * .5 * sqrt(2))) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * dS((my_facet_label, unmarked)))
    assert abs(v - (4 * .5 + 4 * .5 * sqrt(2))) < 1.e-10


@pytest.mark.parallel(nprocs=7)
def test_mark_entities_overlapping_facet_subdomains():
    my_facet_label = 777
    removed_label = 1

    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)
    V1 = FunctionSpace(mesh, "HDiv Trace", 0)
    f1 = Function(V1).interpolate(conditional(And(x > .5, y > .9), 1., 0.))
    mesh = RelabeledMesh(mesh,
                         [f1, Function(V1)],
                         [my_facet_label, removed_label])
    # Check integrals.
    v = assemble(Constant(1, domain=mesh) * ds(my_facet_label))
    assert abs(v - 0.5) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * ds((my_facet_label, 4)))
    assert abs(v - 1.5) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * ds(removed_label))
    assert abs(v - 0.0) < 1.e-10
    v = assemble(Constant(1, domain=mesh) * ds(unmarked))
    assert abs(v - 1.0) < 1.e-10


def test_mark_entities_mesh_mark_entities_3d_hex():
    label_name = "test_label"
    label_value = 999
    mesh = UnitCubeMesh(2, 2, 2, hexahedral=True)
    x, y, z = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "Q", 2)
    f = Function(V).interpolate(conditional(And(x > .6, And(y > .4, y < .6)), 1, 0))
    mesh.mark_entities(f, label_value, label_name=label_name)
    plex = mesh.topology.topology_dm
    label = plex.getLabel(label_name)
    assert label.getStratumIS(label_value).getSize() == 2
    assert all(label.getStratumIS(label_value).getIndices() == [51, 57])


def test_mark_entities_mesh_mark_entities_2d():
    # +---+---+
    # | \ | \ |  mark        \
    # +---+---+ -----> +---+
    # | \ | \ |
    # +---+---+
    label_name = "test_label"
    label_value = 999
    mesh = UnitSquareMesh(2, 2)
    x, y = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "HDiv Trace", 0)
    f = Function(V).interpolate(conditional(Or(And(x < .4,
                                                   And(y > .4, y < .6)),
                                               And(And(x > .6, x < .9),
                                                   And(y > .6, y < .9))), 1., 0.))
    mesh.mark_entities(f, label_value, label_name=label_name)
    plex = mesh.topology.topology_dm
    label = plex.getLabel(label_name)
    assert label.getStratumIS(label_value).getSize() == 2
    assert all(label.getStratumIS(label_value).getIndices() == [20, 30])


def test_mark_entities_mesh_mark_entities_1d():
    label_name = "test_label"
    label_value = 999
    mesh = UnitIntervalMesh(2)
    x, = SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "P", 1)
    f = Function(V).interpolate(conditional(x < 0.25, 1., 0.))
    mesh.mark_entities(f, label_value, label_name=label_name)
    plex = mesh.topology.topology_dm
    label = plex.getLabel(label_name)
    assert label.getStratumIS(label_value).getSize() == 1
    assert all(label.getStratumIS(label_value).getIndices() == [2])

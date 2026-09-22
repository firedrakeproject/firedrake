import numpy as np
import pytest

from firedrake import (
    BrokenMesh,
    DistributedMeshOverlapType,
    Function,
    FunctionSpace,
    Measure,
    RelabeledMesh,
    SpatialCoordinate,
    Submesh,
    UnitSquareMesh,
    assemble,
    conditional,
    lt,
)


def broken_mesh():
    mesh = UnitSquareMesh(
        3,
        2,
        quadrilateral=True,
        distribution_parameters={
            "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
        },
    )
    _, y = SpatialCoordinate(mesh)
    marker_space = FunctionSpace(mesh, "HDiv Trace", 0)
    marker = Function(marker_space).interpolate(
        conditional(lt(abs(y - 0.5), 1.0e-12), 1, 0)
    )
    parent = RelabeledMesh(mesh, [marker], [1])
    return parent, BrokenMesh(parent, 1, reorder=False)


@pytest.mark.parallel(nprocs=[1, 2])
def test_broken_mesh_dof_section():
    parent, broken = broken_mesh()
    entity_dofs = np.zeros(broken.topological_dimension + 1, dtype=np.int32)
    entity_dofs[0] = 1
    parent_section, _ = parent.create_section(entity_dofs)
    broken_section, _ = broken.create_section(entity_dofs)

    assert broken.submesh_parent is parent
    assert broken_section.getStorageSize() > parent_section.getStorageSize()
    assert broken_section.getStorageSize() == broken.num_vertices()

    parent_map = broken.submesh_child_exterior_facet_parent_interior_facet_map.values.ravel()
    parent_map = parent_map[parent_map >= 0]
    assert len(parent_map) > len(np.unique(parent_map))
    _, integral_type = broken.topology.trans_mesh_entity_map(parent.topology, "interior_facet", 1, None)
    assert integral_type == "broken_facet"
    parent_side_map = broken.submesh_parent_interior_facet_child_exterior_facet_map
    gamma_indices = parent.measure_set("interior_facet", 1).indices
    assert parent_side_map.arity == 2
    assert np.all(parent_side_map.values_with_halo[gamma_indices] >= 0)


@pytest.mark.parallel(nprocs=[1, 2])
def test_broken_mesh_standard_spaces_and_cross_mesh_trace():
    parent, broken = broken_mesh()
    parent_space = FunctionSpace(parent, "CG", 1)
    broken_space = FunctionSpace(broken, "CG", 1)
    gamma = Submesh(
        parent,
        parent.topological_dimension - 1,
        1,
        label_name="Face Sets",
        reorder=False,
    )
    gamma_space = FunctionSpace(gamma, "CG", 1)

    assert gamma.submesh_parent is parent
    assert broken_space.dim() > parent_space.dim()
    assert gamma_space.dim() > 0
    parent_trace_map = broken_space.topological.entity_node_map(
        parent.topology,
        "interior_facet",
        1,
        None,
    )
    assert parent_trace_map.arity == 8

    u = Function(broken_space)
    cell_nodes = broken_space.cell_node_map().values
    coordinate_nodes = broken.coordinates.function_space().cell_node_map().values
    cell_y = broken.coordinates.dat.data_ro_with_halos[coordinate_nodes, 1].mean(axis=1)
    assert not np.any(np.isclose(cell_y, 0.5))
    u.dat.data_with_halos[:] = 0
    u.dat.data_with_halos[np.unique(cell_nodes[cell_y < 0.5])] = 1
    u.dat.data_with_halos[np.unique(cell_nodes[cell_y > 0.5])] = 2
    q = Function(gamma_space).assign(1)
    dS_gamma = Measure(
        "dS",
        domain=parent,
        subdomain_id=1,
        intersect_measures=(Measure("dx", gamma), Measure("dS", broken)),
    )
    jump = u("+") - u("-")
    assert assemble(q * jump**2 * dS_gamma) == pytest.approx(1.0)

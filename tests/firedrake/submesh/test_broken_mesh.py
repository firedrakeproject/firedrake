import numpy as np
import pytest
from firedrake import *


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


GAMMA = 99


def cracked_cube(crack_length, n=2):
    """Return a unit cube, the crack Γ = {z = 1/2, x < crack_length}, and the cube broken along Γ."""
    mesh = UnitCubeMesh(n, n, n, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, _, z = SpatialCoordinate(mesh)
    on_gamma = And(lt(abs(z - 0.5), 1e-12), lt(x, crack_length))
    marker = Function(FunctionSpace(mesh, "HDiv Trace", 0))
    marker.interpolate(conditional(on_gamma, 1, 0))
    mesh = RelabeledMesh(mesh, [marker], [GAMMA])
    gamma = Submesh(mesh, mesh.topological_dimension - 1, GAMMA)
    return mesh, gamma, BrokenMesh(mesh, GAMMA)


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("element,trace_element", [
    (("CG", 1), ("CG", 1)),
    (("CG", 3), ("CG", 3)),
    (("RT", 1), ("DG", 0)),
    (("RT", 3), ("DG", 2)),
    (("N1curl", 2), ("N1curl", 2)),
])
def test_broken_mesh_duplicates_trace(element, trace_element):
    """Cutting the cube in two duplicates every dof of the trace space on Γ."""
    mesh, gamma, broken = cracked_cube(crack_length=1)
    V = FunctionSpace(mesh, *element)
    V_broken = FunctionSpace(broken, *element)
    V_trace = FunctionSpace(gamma, *trace_element)
    assert V_broken.dim() == V.dim() + V_trace.dim()


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("degree", [1, 3])
def test_broken_mesh_crack_tip_is_not_split(degree):
    """The n edges of the crack tip {x = z = 1/2} carry n * degree + 1 nodes that are not duplicated."""
    n = 2
    mesh, gamma, broken = cracked_cube(crack_length=0.5, n=n)
    V = FunctionSpace(mesh, "CG", degree)
    V_broken = FunctionSpace(broken, "CG", degree)
    V_trace = FunctionSpace(gamma, "CG", degree)
    assert V_broken.dim() == V.dim() + V_trace.dim() - (n * degree + 1)


def polynomials(mesh, degree, shape):
    """Return two different polynomials of the given degree and shape."""
    x, y, z = SpatialCoordinate(mesh)
    below = (1 + x + 2*y + 3*z) ** degree
    above = -(x - y + z) ** degree
    if shape == (3,):
        return below * as_vector([1, 2, 3]), above * as_vector([3, 1, 2])
    return below, above


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("family,degree,k", [
    ("CG", 2, 2),
    ("CG", 3, 3),
    ("RT", 1, 0),
    ("RT", 3, 2),
])
def test_broken_mesh_jump(family, degree, k):
    """Across Γ, u = p below and u = q above has the jump p - q, for p and q of degree k."""
    mesh, gamma, broken = cracked_cube(crack_length=1)
    V = FunctionSpace(broken, family, degree)
    below = Function(FunctionSpace(broken, "DG", 0))
    below.interpolate(conditional(lt(SpatialCoordinate(broken)[2], 0.5), 1, 0))
    p, q = polynomials(broken, k, V.value_shape)
    u = Function(V).interpolate(below * p + (1 - below) * q)

    W = TensorFunctionSpace(gamma, "DG", k, shape=V.value_shape)
    w = TestFunction(W)
    n = FacetNormal(mesh)
    dS_gamma = Measure("dS", domain=mesh, subdomain_id=GAMMA,
                       intersect_measures=(Measure("dx", gamma), Measure("dS", broken)))
    jump_u = assemble(inner(jump(u) * n("+")[2], w) * dS_gamma)

    p, q = polynomials(gamma, k, V.value_shape)
    jump_exact = assemble(inner(p - q, w) * dx(domain=gamma))
    assert np.allclose(jump_u.dat.data_ro, jump_exact.dat.data_ro)

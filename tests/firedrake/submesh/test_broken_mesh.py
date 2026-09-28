import numpy as np
import pytest
from firedrake import *


def broken_mesh():
    nx, ny = 3, 2
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    mesh = UnitSquareMesh(nx, ny, quadrilateral=True,
                          distribution_parameters=distribution_parameters)
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


def cracked_cube(crack_length, n=2, hexahedral=False):
    """Return a unit cube, the crack Γ = {z = 1/2, x < crack_length}, and the cube broken along Γ."""
    mesh = UnitCubeMesh(n, n, n, hexahedral=hexahedral, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, _, z = SpatialCoordinate(mesh)
    on_gamma = And(lt(abs(z - 0.5), 1e-12), lt(x, crack_length))
    # mark_entities marks hexahedral facets with Q2, and simplicial facets with HDiv Trace.
    marker_element = ("Q", 2) if hexahedral else ("HDiv Trace", 0)
    marker = Function(FunctionSpace(mesh, *marker_element))
    marker.interpolate(conditional(on_gamma, 1, 0))
    mesh = RelabeledMesh(mesh, [marker], [GAMMA])
    gamma = Submesh(mesh, mesh.topological_dimension - 1, GAMMA)
    return mesh, gamma, BrokenMesh(mesh, GAMMA)


def plex_cones(mesh):
    """Return the cone and the cone orientation of every point of the mesh DMPlex."""
    plex = mesh.topology_dm
    return [(plex.getCone(p).tolist(), plex.getConeOrientation(p).tolist())
            for p in range(*plex.getChart())]


@pytest.mark.parallel(nprocs=[1, 3])
def test_broken_mesh_reorients_parent_plex():
    """Orienting Γ to break the mesh updates the parent DMPlex and its caches.

    Some facets of the disk mesh on Γ = {x = 0} are not oriented consistently.
    """
    mesh = UnitDiskMesh(3, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, _ = SpatialCoordinate(mesh)
    marker = Function(FunctionSpace(mesh, "HDiv Trace", 0))
    marker.interpolate(conditional(lt(abs(x), 1e-12), 1, 0))
    mesh = RelabeledMesh(mesh, [marker], [GAMMA])
    topology = mesh.topology
    cones = plex_cones(mesh)
    cell_closure = topology.cell_closure
    entity_orientations = topology.entity_orientations
    local_cell_orientation_dat = topology.local_cell_orientation_dat
    BrokenMesh(mesh, GAMMA)
    changed = int(plex_cones(mesh) != cones)
    assert mesh.comm.allreduce(changed) > 0
    assert topology.cell_closure is not cell_closure
    assert topology.entity_orientations is not entity_orientations
    assert topology.local_cell_orientation_dat is not local_cell_orientation_dat
    orientation_changed = int(
        not np.array_equal(topology.entity_orientations, entity_orientations)
    )
    assert mesh.comm.allreduce(orientation_changed) > 0


@pytest.mark.parallel(nprocs=[1, 3])
def test_broken_mesh_accepts_multiple_subdomains():
    """Breaking a surface marked by several values includes every marked facet."""
    nx, ny = 3, 2
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    mesh = UnitSquareMesh(nx, ny, quadrilateral=True,
                          distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    marker_left = Function(FunctionSpace(mesh, "HDiv Trace", 0)).interpolate(
        conditional(And(lt(abs(y - 0.5), 1.0e-12), lt(x, 0.5)), 1, 0)
    )
    marker_right = Function(FunctionSpace(mesh, "HDiv Trace", 0)).interpolate(
        conditional(And(lt(abs(y - 0.5), 1.0e-12), ge(x, 0.5)), 1, 0)
    )
    parent = RelabeledMesh(mesh, [marker_left, marker_right], [GAMMA, GAMMA + 1])
    broken = BrokenMesh(parent, (GAMMA, GAMMA + 1), reorder=False)
    gamma = Submesh(
        parent,
        parent.topological_dimension - 1,
        (GAMMA, GAMMA + 1),
        label_name="Face Sets",
        reorder=False,
    )

    V = FunctionSpace(parent, "CG", 1)
    V_broken = FunctionSpace(broken, "CG", 1)
    V_trace = FunctionSpace(gamma, "CG", 1)
    assert V_broken.dim() == V.dim() + V_trace.dim()


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("hexahedral,element,trace_element", [
    (False, ("CG", 1), ("CG", 1)),
    (False, ("CG", 3), ("CG", 3)),
    (False, ("RT", 1), ("DG", 0)),
    (False, ("RT", 3), ("DG", 2)),
    (False, ("N1curl", 2), ("N1curl", 2)),
    (True, ("Q", 1), ("Q", 1)),
    (True, ("Q", 3), ("Q", 3)),
])
def test_broken_mesh_duplicates_trace(hexahedral, element, trace_element):
    """Cutting the cube in two duplicates every dof of the trace space on Γ."""
    mesh, gamma, broken = cracked_cube(crack_length=1, hexahedral=hexahedral)
    V = FunctionSpace(mesh, *element)
    V_broken = FunctionSpace(broken, *element)
    V_trace = FunctionSpace(gamma, *trace_element)
    assert V_broken.dim() == V.dim() + V_trace.dim()


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("hexahedral", [False, True])
@pytest.mark.parametrize("degree", [1, 3])
def test_broken_mesh_crack_tip_is_not_split(degree, hexahedral):
    """The n edges of the crack tip {x = z = 1/2} carry n * degree + 1 nodes that are not duplicated."""
    n = 2
    mesh, gamma, broken = cracked_cube(crack_length=0.5, n=n, hexahedral=hexahedral)
    V = FunctionSpace(mesh, "CG", degree)
    V_broken = FunctionSpace(broken, "CG", degree)
    V_trace = FunctionSpace(gamma, "CG", degree)
    assert V_broken.dim() == V.dim() + V_trace.dim() - (n * degree + 1)


def t_junction_square(n=8):
    """Return a unit square with Γ₁ = {y = 1/2} and Γ₂ = {x = 1/2, y > 1/2}, which ends on Γ₁."""
    mesh = UnitSquareMesh(n, n, distribution_parameters={
        "partitioner_type": "simple",
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, y = SpatialCoordinate(mesh)
    markers = []
    for on_gamma in (lt(abs(y - 0.5), 1e-12), And(lt(abs(x - 0.5), 1e-12), gt(y, 0.5))):
        marker = Function(FunctionSpace(mesh, "HDiv Trace", 0))
        markers.append(marker.interpolate(conditional(on_gamma, 1, 0)))
    return RelabeledMesh(mesh, markers, [GAMMA, GAMMA + 1])


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("junctions,num_unsplit", [("split", 1), ("unsplit", 2)])
def test_broken_mesh_t_junction(junctions, num_unsplit):
    """Γ₂ is not split where it ends on Γ₁, and with junctions="unsplit" neither is Γ₁."""
    mesh = t_junction_square()
    broken = BrokenMesh(mesh, (GAMMA, GAMMA + 1), junctions=junctions)
    traces = [Submesh(mesh, mesh.topological_dimension - 1, subid) for subid in (GAMMA, GAMMA + 1)]
    V = FunctionSpace(mesh, "CG", 1)
    V_broken = FunctionSpace(broken, "CG", 1)
    V_traces = [FunctionSpace(trace, "CG", 1) for trace in traces]
    assert V_broken.dim() == V.dim() + sum(V_trace.dim() for V_trace in V_traces) - num_unsplit


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("junctions", ["split", "unsplit"])
def test_broken_mesh_t_junction_jump(junctions):
    """u = q + (x - 1/2)(1 + y) below Γ₁ and u = q + (y - 1/2)(1 + x) right of Γ₂ jump by these terms.

    Both terms vanish at the junction, so u is continuous there with either value of junctions.
    """
    mesh = t_junction_square()
    broken = BrokenMesh(mesh, (GAMMA, GAMMA + 1), junctions=junctions)
    x, y = SpatialCoordinate(broken)
    DG0 = FunctionSpace(broken, "DG", 0)
    below = Function(DG0).interpolate(conditional(lt(y, 0.5), 1, 0))
    right = Function(DG0).interpolate(conditional(And(gt(y, 0.5), gt(x, 0.5)), 1, 0))
    q = (1 + x + 2*y)**2
    u = Function(FunctionSpace(broken, "CG", 2))
    u.interpolate(q + below * (x - 0.5) * (1 + y) + right * (y - 0.5) * (1 + x))

    n = FacetNormal(mesh)
    for subid, component in ((GAMMA, 1), (GAMMA + 1, 0)):
        gamma = Submesh(mesh, mesh.topological_dimension - 1, subid)
        w = TestFunction(FunctionSpace(gamma, "DG", 2))
        dS_gamma = Measure("dS", domain=mesh, subdomain_id=subid,
                           intersect_measures=(Measure("dx", gamma), Measure("dS", broken)))
        jump_u = assemble(inner(jump(u) * n("+")[component], w) * dS_gamma)

        x, y = SpatialCoordinate(gamma)
        jump_exact = (x - 0.5) * (1 + y) if subid == GAMMA else -(y - 0.5) * (1 + x)
        jump_exact = assemble(inner(jump_exact, w) * dx(domain=gamma))
        assert np.allclose(jump_u.dat.data_ro, jump_exact.dat.data_ro)


def test_broken_mesh_rejects_junction():
    """A single surface cannot be broken across a junction."""
    mesh = t_junction_square()
    with pytest.raises(ValueError, match="junction"):
        BrokenMesh(mesh, (GAMMA, GAMMA + 1))


def polynomials(mesh, degree, shape):
    """Return two different polynomials of the given degree and shape."""
    x, y, z = SpatialCoordinate(mesh)
    below = (1 + x + 2*y + 3*z) ** degree
    above = -(x - y + z) ** degree
    if shape == (3,):
        return below * as_vector([1, 2, 3]), above * as_vector([3, 1, 2])
    return below, above


@pytest.mark.parallel(nprocs=[1, 3])
@pytest.mark.parametrize("hexahedral,family,degree,k", [
    (False, "CG", 2, 2),
    (False, "CG", 3, 3),
    (False, "RT", 1, 0),
    (False, "RT", 3, 2),
    (True, "Q", 2, 2),
    (True, "Q", 3, 3),
])
def test_broken_mesh_jump(hexahedral, family, degree, k):
    """Across Γ, u = p below and u = q above has the jump p - q, for p and q of degree k."""
    mesh, gamma, broken = cracked_cube(crack_length=1, hexahedral=hexahedral)
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

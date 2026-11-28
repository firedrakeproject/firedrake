from functools import partial

import numpy as np
import pytest
from pytest_mpi.parallel_assert import parallel_assert
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
    marker = Function(marker_space)
    marker.interpolate(
        conditional(lt(abs(y - 0.5), 1.0e-12), 1, 0)
    )
    parent = RelabeledMesh(mesh, [marker], [1])
    return parent, BrokenMesh(parent, 1, reorder=False)


@pytest.mark.parallel([1, 2])
def test_broken_mesh_duplicates_surface_vertices():
    parent, broken = broken_mesh()
    gamma = Submesh(parent, parent.topological_dimension - 1, 1)
    V = FunctionSpace(parent, "CG", 1)
    V_broken = FunctionSpace(broken, "CG", 1)
    V_trace = FunctionSpace(gamma, "CG", 1)

    assert V_broken.dim() == V.dim() + V_trace.dim()


@pytest.mark.parallel([1, 2])
def test_broken_mesh_jump_integral_with_codimension_one_submesh():
    parent, broken = broken_mesh()
    gamma = Submesh(parent, parent.topological_dimension - 1, 1)
    V = FunctionSpace(broken, "CG", 1)
    Q = FunctionSpace(broken, "DG", 0)
    W = FunctionSpace(gamma, "CG", 1)
    _, y = SpatialCoordinate(broken)
    below = Function(Q)
    below.interpolate(conditional(lt(y, 0.5), 1, 0))
    u = Function(V)
    u.interpolate(2 - below)
    q = Function(W)
    q.assign(1)
    dS_gamma = Measure(
        "dS",
        domain=parent,
        subdomain_id=1,
        intersect_measures=(Measure("dx", gamma), Measure("dS", broken)),
    )

    assert assemble(q * jump(u)**2 * dS_gamma) == pytest.approx(1.0)


@pytest.mark.parallel([1, 2, 3])
def test_broken_mesh_matrix_action():
    _, broken = broken_mesh()
    V = FunctionSpace(broken, "CG", 1)
    x, y = SpatialCoordinate(broken)
    u = Function(V)
    u.interpolate(1 + x * y)
    trial = TrialFunction(V)
    v = TestFunction(V)
    A = assemble(inner(trial, v) * dx)
    Au = assemble(inner(u, v) * dx)
    with u.dat.vec_ro as u_vec, Au.dat.vec_ro as Au_vec:
        assert (A.petscmat * u_vec - Au_vec).norm() < 1e-12


GAMMA = 99


def cracked_cube(crack_length, n=2, *, hexahedral, marker_element):
    """Return a unit cube, the crack Γ = {z = 1/2, x < crack_length}, and the cube broken along Γ."""
    mesh = UnitCubeMesh(n, n, n, hexahedral=hexahedral, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, _, z = SpatialCoordinate(mesh)
    on_gamma = And(lt(abs(z - 0.5), 1e-12), lt(x, crack_length))
    W = FunctionSpace(mesh, *marker_element)
    marker = Function(W)
    marker.interpolate(conditional(on_gamma, 1, 0))
    mesh = RelabeledMesh(mesh, [marker], [GAMMA])
    gamma = Submesh(mesh, mesh.topological_dimension - 1, GAMMA)
    return mesh, gamma, BrokenMesh(mesh, GAMMA)


tetrahedral_crack = partial(cracked_cube, hexahedral=False, marker_element=("HDiv Trace", 0))
hexahedral_crack = partial(cracked_cube, hexahedral=True, marker_element=("Q", 2))


@pytest.mark.parallel([1, 3])
def test_broken_mesh_preserves_parent_facet_integrals():
    mesh = UnitDiskMesh(3, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, _ = SpatialCoordinate(mesh)
    W = FunctionSpace(mesh, "HDiv Trace", 0)
    marker = Function(W)
    marker.interpolate(conditional(lt(abs(x), 1e-12), 1, 0))
    mesh = RelabeledMesh(mesh, [marker], [GAMMA])

    V = FunctionSpace(mesh, "DG", 2)
    x, y = SpatialCoordinate(mesh)
    u = Function(V)
    u.interpolate(x + y**2)
    v = TestFunction(V)
    normal = FacetNormal(mesh)
    integrand = inner(u("+") * normal("+")[0], jump(v)) * dS(GAMMA, domain=mesh)
    expected = assemble(integrand)

    BrokenMesh(mesh, GAMMA)
    actual = assemble(integrand)
    with expected.dat.vec_ro as expected_vec, actual.dat.vec_ro as actual_vec:
        assert (actual_vec - expected_vec).norm() < 1e-12


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("label_name", ["Face Sets", "firedrake_cohesive_label_0"])
def test_broken_mesh_duplicates_facets_from_multiple_subdomains(label_name):
    """Breaking a surface marked by several values includes every marked facet."""
    nx, ny = 3, 2
    distribution_parameters = {
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1),
    }
    mesh = UnitSquareMesh(nx, ny, quadrilateral=True,
                          distribution_parameters=distribution_parameters)
    x, y = SpatialCoordinate(mesh)
    W = FunctionSpace(mesh, "HDiv Trace", 0)
    marker_left = Function(W)
    marker_left.interpolate(
        conditional(And(lt(abs(y - 0.5), 1.0e-12), lt(x, 0.5)), 1, 0)
    )
    marker_right = Function(W)
    marker_right.interpolate(
        conditional(And(lt(abs(y - 0.5), 1.0e-12), ge(x, 0.5)), 1, 0)
    )
    mesh.mark_entities(marker_left, GAMMA, label_name)
    mesh.mark_entities(marker_right, GAMMA + 1, label_name)
    broken = BrokenMesh(mesh, (GAMMA, GAMMA + 1), label_name=label_name, reorder=False)
    gamma = Submesh(
        mesh,
        mesh.topological_dimension - 1,
        (GAMMA, GAMMA + 1),
        label_name=label_name,
        reorder=False,
    )

    V = FunctionSpace(mesh, "CG", 1)
    V_broken = FunctionSpace(broken, "CG", 1)
    V_trace = FunctionSpace(gamma, "CG", 1)
    assert V_broken.dim() == V.dim() + V_trace.dim()


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("trace_case", [
    pytest.param((tetrahedral_crack, ("CG", 1), ("CG", 1)), id="tet-CG1"),
    pytest.param((tetrahedral_crack, ("CG", 3), ("CG", 3)), id="tet-CG3"),
    pytest.param((tetrahedral_crack, ("RT", 1), ("DG", 0)), id="tet-RT1"),
    pytest.param((tetrahedral_crack, ("RT", 3), ("DG", 2)), id="tet-RT3"),
    pytest.param((tetrahedral_crack, ("N1curl", 2), ("N1curl", 2)), id="tet-N1curl2"),
    pytest.param((hexahedral_crack, ("Q", 1), ("Q", 1)), id="hex-Q1"),
    pytest.param((hexahedral_crack, ("Q", 3), ("Q", 3)), id="hex-Q3"),
])
def test_broken_mesh_duplicates_trace(trace_case):
    """Cutting the cube in two duplicates every dof of the trace space on Γ."""
    make_crack, element, trace_element = trace_case
    mesh, gamma, broken = make_crack(crack_length=1)
    V = FunctionSpace(mesh, *element)
    V_broken = FunctionSpace(broken, *element)
    V_trace = FunctionSpace(gamma, *trace_element)
    assert V_broken.dim() == V.dim() + V_trace.dim()


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("make_crack", [tetrahedral_crack, hexahedral_crack], ids=["tet", "hex"])
@pytest.mark.parametrize("degree", [1, 3])
def test_broken_mesh_crack_tip_dofs_are_shared(degree, make_crack):
    """The n edges of the crack tip {x = z = 1/2} carry n * degree + 1 nodes that are not duplicated."""
    n = 2
    mesh, gamma, broken = make_crack(crack_length=0.5, n=n)
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
    W = FunctionSpace(mesh, "HDiv Trace", 0)
    markers = []
    for on_gamma in (lt(abs(y - 0.5), 1e-12), And(lt(abs(x - 0.5), 1e-12), gt(y, 0.5))):
        marker = Function(W)
        marker.interpolate(conditional(on_gamma, 1, 0))
        markers.append(marker)
    return RelabeledMesh(mesh, markers, [GAMMA, GAMMA + 1])


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("junction", [
    pytest.param(("split", 1), id="split"),
    pytest.param(("unsplit", 2), id="unsplit"),
])
def test_broken_mesh_junction_dofs_are_shared(junction):
    """Γ₂ is not split where it ends on Γ₁, and with junctions="unsplit" neither is Γ₁."""
    mesh = t_junction_square()
    junctions, num_unsplit = junction
    broken = BrokenMesh(mesh, (GAMMA, GAMMA + 1), junctions=junctions)
    traces = [Submesh(mesh, mesh.topological_dimension - 1, subid) for subid in (GAMMA, GAMMA + 1)]
    V = FunctionSpace(mesh, "CG", 1)
    V_broken = FunctionSpace(broken, "CG", 1)
    V_traces = [FunctionSpace(trace, "CG", 1) for trace in traces]
    assert V_broken.dim() == V.dim() + sum(V_trace.dim() for V_trace in V_traces) - num_unsplit


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("junctions", ["split", "unsplit"])
def test_broken_mesh_t_junction_jump(junctions):
    """u = q + (x - 1/2)(1 + y) below Γ₁ and u = q + (y - 1/2)(1 + x) right of Γ₂ jump by these terms.

    Both terms vanish at the junction, so u is continuous there with either value of junctions.
    """
    mesh = t_junction_square()
    broken = BrokenMesh(mesh, (GAMMA, GAMMA + 1), junctions=junctions)
    x, y = SpatialCoordinate(broken)
    DG0 = FunctionSpace(broken, "DG", 0)
    below = Function(DG0)
    below.interpolate(conditional(lt(y, 0.5), 1, 0))
    right = Function(DG0)
    right.interpolate(conditional(And(gt(y, 0.5), gt(x, 0.5)), 1, 0))
    q = (1 + x + 2*y)**2
    V = FunctionSpace(broken, "CG", 2)
    u = Function(V)
    u.interpolate(q + below * (x - 0.5) * (1 + y) + right * (y - 0.5) * (1 + x))

    n = FacetNormal(mesh)
    for subid, component in ((GAMMA, 1), (GAMMA + 1, 0)):
        gamma = Submesh(mesh, mesh.topological_dimension - 1, subid)
        W = FunctionSpace(gamma, "DG", 2)
        w = TestFunction(W)
        dS_gamma = Measure("dS", domain=mesh, subdomain_id=subid,
                           intersect_measures=(Measure("dx", gamma), Measure("dS", broken)))
        jump_u = assemble(inner(jump(u) * n("+")[component], w) * dS_gamma)

        x, y = SpatialCoordinate(gamma)
        jump_exact = (x - 0.5) * (1 + y) if subid == GAMMA else -(y - 0.5) * (1 + x)
        jump_exact = assemble(inner(jump_exact, w) * dx(domain=gamma))
        parallel_assert(np.allclose(jump_u.dat.data_ro, jump_exact.dat.data_ro),
                        "The assembled jump differs from the exact jump")


def test_broken_mesh_rejects_junction():
    """A single surface cannot be broken across a junction."""
    mesh = t_junction_square()
    with pytest.raises(ValueError, match="junction"):
        BrokenMesh(mesh, (GAMMA, GAMMA + 1))


@pytest.mark.parallel([1, 3])
def test_broken_mesh_crossing_faults() -> None:
    """Two crossing faults have independent displacement jumps."""
    mesh = UnitCubeMesh(2, 2, 2, distribution_parameters={
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, y, _ = SpatialCoordinate(mesh)
    W = FunctionSpace(mesh, "HDiv Trace", 0)
    markers = []
    for c in (x, y):
        marker = Function(W)
        marker.interpolate(conditional(lt(abs(c - 0.5), 1e-12), 1, 0))
        markers.append(marker)
    mesh = RelabeledMesh(mesh, markers, [GAMMA, GAMMA + 1])
    broken = BrokenMesh(mesh, (GAMMA, GAMMA + 1), junctions="split")

    x, y, z = SpatialCoordinate(broken)
    DG0 = FunctionSpace(broken, "DG", 0)
    left = Function(DG0)
    left.interpolate(conditional(lt(x, 0.5), 1, 0))
    below = Function(DG0)
    below.interpolate(conditional(lt(y, 0.5), 1, 0))
    slip_x = as_vector((0, 1, 0))
    slip_y = as_vector((1, 0, 0))
    V = VectorFunctionSpace(broken, "CG", 1)
    u = Function(V)
    u.interpolate(as_vector((x, y, z)) + left * slip_x + below * slip_y)

    n = FacetNormal(mesh)
    dS_fault = Measure("dS", domain=mesh, intersect_measures=(Measure("dS", broken),))
    for subid, component, slip in ((GAMMA, 0, slip_x), (GAMMA + 1, 1, slip_y)):
        error = jump(u) * n("+")[component] - slip
        assert assemble(inner(error, error) * dS_fault(subid)) == pytest.approx(0, abs=1e-12)


@pytest.mark.parallel(3)
def test_broken_mesh_jump_on_a_partial_fault():
    mesh = UnitCubeMesh(4, 4, 4, distribution_parameters={
        "partitioner_type": "simple",
        "overlap_type": (DistributedMeshOverlapType.RIDGE, 1)})
    x, y, _ = SpatialCoordinate(mesh)
    on_fault = And(lt(abs(y - 0.5), 1e-12), lt(x, 0.75))
    W = FunctionSpace(mesh, "HDiv Trace", 0)
    marker = Function(W)
    marker.interpolate(conditional(on_fault, 1, 0))
    mesh = RelabeledMesh(mesh, [marker], [GAMMA])
    broken = BrokenMesh(mesh, GAMMA)

    x, y, _ = SpatialCoordinate(broken)
    Q = FunctionSpace(broken, "DG", 0)
    below = Function(Q)
    below.interpolate(conditional(lt(y, 0.5), 1, 0))
    V = FunctionSpace(broken, "CG", 1)
    u = Function(V)
    u.interpolate(below * max_value(0, 0.75 - x))

    x, _, _ = SpatialCoordinate(mesh)
    normal = FacetNormal(mesh)
    expected = 0.75 - x("+")
    error = jump(u) * normal("+")[1] - expected
    dS_fault = Measure("dS", domain=mesh, subdomain_id=GAMMA,
                       intersect_measures=(Measure("dS", broken),))
    assert assemble(inner(error, error) * dS_fault) == pytest.approx(0, abs=1e-12)


def polynomials(mesh, degree, shape):
    """Return two different polynomials of the given degree and shape."""
    x, y, z = SpatialCoordinate(mesh)
    below = (1 + x + 2*y + 3*z) ** degree
    above = -(x - y + z) ** degree
    if shape == (3,):
        return below * as_vector([1, 2, 3]), above * as_vector([3, 1, 2])
    return below, above


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("jump_case", [
    pytest.param((tetrahedral_crack, ("CG", 2), 2), id="tet-CG2"),
    pytest.param((tetrahedral_crack, ("CG", 3), 3), id="tet-CG3"),
    pytest.param((tetrahedral_crack, ("RT", 1), 0), id="tet-RT1"),
    pytest.param((tetrahedral_crack, ("RT", 3), 2), id="tet-RT3"),
    pytest.param((hexahedral_crack, ("Q", 2), 2), id="hex-Q2"),
    pytest.param((hexahedral_crack, ("Q", 3), 3), id="hex-Q3"),
])
def test_broken_mesh_jump(jump_case):
    """Across Γ, u = p below and u = q above has the jump p - q, for p and q of degree k."""
    make_crack, element, k = jump_case
    mesh, gamma, broken = make_crack(crack_length=1)
    V = FunctionSpace(broken, *element)
    Q = FunctionSpace(broken, "DG", 0)
    below = Function(Q)
    below.interpolate(conditional(lt(SpatialCoordinate(broken)[2], 0.5), 1, 0))
    p, q = polynomials(broken, k, V.value_shape)
    u = Function(V)
    u.interpolate(below * p + (1 - below) * q)

    W = TensorFunctionSpace(gamma, "DG", k, shape=V.value_shape)
    w = TestFunction(W)
    n = FacetNormal(mesh)
    dS_gamma = Measure("dS", domain=mesh, subdomain_id=GAMMA,
                       intersect_measures=(Measure("dx", gamma), Measure("dS", broken)))
    jump_u = assemble(inner(jump(u) * n("+")[2], w) * dS_gamma)

    p, q = polynomials(gamma, k, V.value_shape)
    jump_exact = assemble(inner(p - q, w) * dx(domain=gamma))
    parallel_assert(np.allclose(jump_u.dat.data_ro, jump_exact.dat.data_ro),
                    "The assembled jump differs from the exact jump")

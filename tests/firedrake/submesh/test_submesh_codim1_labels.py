import pytest
import numpy as np
from firedrake import *


def _get_dm_vertex_coords(dm):
    """Return ``{vertex_point: np.array([x, y, ...])}`` from the DM coordinate section."""
    coord_sec = dm.getCoordinateSection()
    coord_arr = dm.getCoordinatesLocal().array_r
    vStart, vEnd = dm.getDepthStratum(0)
    result = {}
    for v in range(vStart, vEnd):
        off = coord_sec.getOffset(v)
        dof = coord_sec.getDof(v)
        result[v] = coord_arr[off:off + dof].copy()
    return result


def _entity_vertex_coords(dm, point, vc):
    """Return vertex coordinates for all vertices in the closure of *point*."""
    vStart, vEnd = dm.getDepthStratum(0)
    closure = dm.getTransitiveClosure(point)[0]
    verts = [c for c in closure if vStart <= c < vEnd]
    if not verts:
        return np.empty((0, dm.getCoordinateDim()))
    return np.array([vc[v] for v in verts])


def _all_at(cs, axis, value):
    return np.allclose(cs[:, axis], value)


# ---------------------------------------------------------------------------
# 3D -> 2D helpers
# ---------------------------------------------------------------------------

FACE_TAG_3D = 100
EDGE_A = 1
EDGE_B = 2
EDGE_C = 3


def _make_cube_with_edge_sets(extra_edge_c=False, ncells=2):
    """UnitCubeMesh with Face Sets on z=0 boundary faces and Edge Sets
    on selected boundary edges of the z=0 face.

    If *extra_edge_c* is True an additional label EDGE_C is placed on edges
    at z=1, x=0 (not part of the z=0 submesh).
    """
    mesh = UnitCubeMesh(ncells, ncells, ncells)
    dm = mesh.topology_dm
    # Remove default Face Sets so we have full control over labels.
    if dm.hasLabel("Face Sets"):
        dm.removeLabel("Face Sets")
    dm.createLabel("Face Sets")
    vc = _get_dm_vertex_coords(dm)
    eStart, eEnd = dm.getDepthStratum(1)
    fStart, fEnd = dm.getDepthStratum(2)

    for f in range(fStart, fEnd):
        cs = _entity_vertex_coords(dm, f, vc)
        if cs.shape[0] >= 3 and _all_at(cs, 2, 0.0) and dm.getSupportSize(f) == 1:
            dm.setLabelValue("Face Sets", f, FACE_TAG_3D)

    dm.createLabel("Edge Sets")
    for e in range(eStart, eEnd):
        cs = _entity_vertex_coords(dm, e, vc)
        if cs.shape[0] < 2:
            continue
        if _all_at(cs, 2, 0.0) and _all_at(cs, 0, 0.0):
            dm.setLabelValue("Edge Sets", e, EDGE_A)
        elif _all_at(cs, 2, 0.0) and _all_at(cs, 0, 1.0):
            dm.setLabelValue("Edge Sets", e, EDGE_B)
        elif extra_edge_c and _all_at(cs, 2, 1.0) and _all_at(cs, 0, 0.0):
            dm.setLabelValue("Edge Sets", e, EDGE_C)

    return mesh


# ---------------------------------------------------------------------------
# 2D -> 1D helpers
# ---------------------------------------------------------------------------

FACE_TAG_2D = 200
VTX_A = 1
VTX_B = 2


def _make_square_with_vertex_sets(ncells=2):
    """UnitSquareMesh with Face Sets on y=0 boundary edges and Vertex
    Sets on the corner vertices of that edge.
    """
    mesh = UnitSquareMesh(ncells, ncells)
    dm = mesh.topology_dm
    # Remove default Face Sets so we have full control over labels.
    if dm.hasLabel("Face Sets"):
        dm.removeLabel("Face Sets")
    dm.createLabel("Face Sets")
    vc = _get_dm_vertex_coords(dm)
    vStart, vEnd = dm.getDepthStratum(0)
    eStart, eEnd = dm.getDepthStratum(1)

    for e in range(eStart, eEnd):
        cs = _entity_vertex_coords(dm, e, vc)
        if cs.shape[0] >= 2 and _all_at(cs, 1, 0.0) and dm.getSupportSize(e) == 1:
            dm.setLabelValue("Face Sets", e, FACE_TAG_2D)

    dm.createLabel("Vertex Sets")
    for v in range(vStart, vEnd):
        coords = vc[v]
        if np.isclose(coords[1], 0.0) and np.isclose(coords[0], 0.0):
            dm.setLabelValue("Vertex Sets", v, VTX_A)
        elif np.isclose(coords[1], 0.0) and np.isclose(coords[0], 1.0):
            dm.setLabelValue("Vertex Sets", v, VTX_B)

    return mesh


# ---------------------------------------------------------------------------
# Tests – 3D parent -> 2D submesh (Edge Sets → Face Sets)
# ---------------------------------------------------------------------------

def _check_edge_sets_propagated(ncells=2):
    mesh = _make_cube_with_edge_sets(ncells=ncells)
    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    markers = submesh.exterior_facets.unique_markers
    assert markers is not None
    marker_set = set(markers)
    assert EDGE_A in marker_set, f"EDGE_A={EDGE_A} not in {marker_set}"
    assert EDGE_B in marker_set, f"EDGE_B={EDGE_B} not in {marker_set}"


def test_submesh_codim1_edge_sets_propagated():
    """Parent 'Edge Sets' tags appear in submesh exterior_facets.unique_markers."""
    _check_edge_sets_propagated()


def test_submesh_codim1_edge_sets_excludes_nonsubmesh():
    """Edge Sets on parent edges outside the submesh are excluded."""
    mesh = _make_cube_with_edge_sets(extra_edge_c=True)
    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    markers = submesh.exterior_facets.unique_markers
    assert markers is not None
    marker_set = set(markers)
    assert EDGE_A in marker_set
    assert EDGE_B in marker_set
    assert EDGE_C not in marker_set, f"EDGE_C={EDGE_C} should not be in {marker_set}"


def test_submesh_codim1_unlabeled_get_default_value():
    """Exterior facets without a parent Edge Sets label get a fresh default value."""
    mesh = _make_cube_with_edge_sets()
    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    markers = submesh.exterior_facets.unique_markers
    assert markers is not None
    marker_set = set(markers)
    # Default must be larger than all Edge Sets and inherited Face Sets values.
    default_vals = marker_set - {EDGE_A, EDGE_B, FACE_TAG_3D}
    assert len(default_vals) == 1, f"Expected one default value, got {default_vals}"
    default_val = default_vals.pop()
    assert default_val > max(EDGE_A, EDGE_B), (
        f"Default {default_val} should be > max(EDGE_A, EDGE_B)={max(EDGE_A, EDGE_B)}"
    )


def _check_ds_edge_sets(ncells=2):
    """Integrate 1 over each tagged boundary of the 2D submesh."""
    mesh = _make_cube_with_edge_sets(ncells=ncells)
    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    # EDGE_A labels edges at x=0, z=0: total length = 1
    assert abs(assemble(Constant(1.) * ds(EDGE_A, domain=submesh)) - 1.0) < 1e-12
    # EDGE_B labels edges at x=1, z=0: total length = 1
    assert abs(assemble(Constant(1.) * ds(EDGE_B, domain=submesh)) - 1.0) < 1e-12


def test_submesh_codim1_ds_edge_sets():
    """ds(tag) with propagated Edge Sets integrates the correct boundary length."""
    _check_ds_edge_sets()


# ---------------------------------------------------------------------------
# Tests – 2D parent -> 1D submesh (Vertex Sets → Face Sets)
# ---------------------------------------------------------------------------

def test_submesh_codim1_vertex_sets_propagated():
    """Parent 'Vertex Sets' tags appear in 1D submesh exterior_facets.unique_markers."""
    mesh = _make_square_with_vertex_sets()
    submesh = Submesh(mesh, 1, FACE_TAG_2D)
    markers = submesh.exterior_facets.unique_markers
    assert markers is not None
    marker_set = set(markers)
    assert VTX_A in marker_set, f"VTX_A={VTX_A} not in {marker_set}"
    assert VTX_B in marker_set, f"VTX_B={VTX_B} not in {marker_set}"


def test_submesh_codim1_ds_vertex_sets():
    """ds(tag) with propagated Vertex Sets works on a 1D submesh."""
    mesh = _make_square_with_vertex_sets()
    submesh = Submesh(mesh, 1, FACE_TAG_2D)
    # Each tagged vertex is a single point (0-D measure = 1)
    assert abs(assemble(Constant(1.) * ds(VTX_A, domain=submesh)) - 1.0) < 1e-12
    assert abs(assemble(Constant(1.) * ds(VTX_B, domain=submesh)) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# Tests – no parent label
# ---------------------------------------------------------------------------

def test_submesh_codim1_no_parent_edge_sets():
    """Codim-1 submesh with no Edge Sets on the parent still labels exterior facets."""
    mesh = UnitCubeMesh(2, 2, 2)
    dm = mesh.topology_dm
    if dm.hasLabel("Face Sets"):
        dm.removeLabel("Face Sets")
    dm.createLabel("Face Sets")
    vc = _get_dm_vertex_coords(dm)
    fStart, fEnd = dm.getDepthStratum(2)

    for f in range(fStart, fEnd):
        cs = _entity_vertex_coords(dm, f, vc)
        if cs.shape[0] >= 3 and _all_at(cs, 2, 0.0) and dm.getSupportSize(f) == 1:
            dm.setLabelValue("Face Sets", f, FACE_TAG_3D)

    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    markers = submesh.exterior_facets.unique_markers
    assert markers is not None
    assert len(markers) > 0


# ---------------------------------------------------------------------------
# Parallel – marker propagation
# ---------------------------------------------------------------------------


@pytest.mark.parallel(nprocs=2)
def test_submesh_codim1_edge_sets_propagated_2_procs():
    _check_edge_sets_propagated()


@pytest.mark.parallel(nprocs=3)
def test_submesh_codim1_edge_sets_propagated_3_procs():
    _check_edge_sets_propagated(ncells=4)


# ---------------------------------------------------------------------------
# Parallel – boundary length integration (ds)
# ---------------------------------------------------------------------------


@pytest.mark.parallel(nprocs=2)
def test_submesh_codim1_ds_edge_sets_2_procs():
    _check_ds_edge_sets()


@pytest.mark.parallel(nprocs=3)
def test_submesh_codim1_ds_edge_sets_3_procs():
    _check_ds_edge_sets(ncells=4)


# ---------------------------------------------------------------------------
# Parallel – coordinate expression integration
# ---------------------------------------------------------------------------

def _check_boundary_coordinate_integral(ncells=2):
    """Integrate coordinate expressions over tagged boundaries.

    On the z=0 submesh:
      EDGE_A is at x=0, y in [0,1]  =>  int(y dy) = 1/2,  int(x dy) = 0
      EDGE_B is at x=1, y in [0,1]  =>  int(y dy) = 1/2,  int(x dy) = 1
    """
    mesh = _make_cube_with_edge_sets(ncells=ncells)
    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    x, y, z = SpatialCoordinate(submesh)
    assert abs(assemble(y * ds(EDGE_A, domain=submesh)) - 0.5) < 1e-12
    assert abs(assemble(x * ds(EDGE_A, domain=submesh))) < 1e-12
    assert abs(assemble(y * ds(EDGE_B, domain=submesh)) - 0.5) < 1e-12
    assert abs(assemble(x * ds(EDGE_B, domain=submesh)) - 1.0) < 1e-12


@pytest.mark.parallel(nprocs=2)
def test_submesh_codim1_boundary_coord_integral_2_procs():
    _check_boundary_coordinate_integral()


@pytest.mark.parallel(nprocs=3)
def test_submesh_codim1_boundary_coord_integral_3_procs():
    _check_boundary_coordinate_integral(ncells=4)


# ---------------------------------------------------------------------------
# Parallel – FunctionSpace interpolation + boundary integration
# ---------------------------------------------------------------------------

def _check_interpolate_and_integrate(ncells=2):
    """Interpolate x^2+y^2 on the submesh and integrate over tagged boundaries.

    EDGE_A (x=0): int_0^1 y^2 dy = 1/3
    EDGE_B (x=1): int_0^1 (1+y^2) dy = 4/3
    """
    mesh = _make_cube_with_edge_sets(ncells=ncells)
    submesh = Submesh(mesh, 2, FACE_TAG_3D)
    V = FunctionSpace(submesh, "CG", 2)
    x, y, z = SpatialCoordinate(submesh)
    f = Function(V).interpolate(x**2 + y**2)
    assert abs(assemble(f * ds(EDGE_A)) - 1. / 3) < 1e-12
    assert abs(assemble(f * ds(EDGE_B)) - 4. / 3) < 1e-12


@pytest.mark.parallel(nprocs=2)
def test_submesh_codim1_interpolate_integrate_2_procs():
    _check_interpolate_and_integrate()


@pytest.mark.parallel(nprocs=3)
def test_submesh_codim1_interpolate_integrate_3_procs():
    _check_interpolate_and_integrate(ncells=4)


# ---------------------------------------------------------------------------
# Parallel – 2D -> 1D (Vertex Sets)
# ---------------------------------------------------------------------------

def _check_vertex_sets_ds(ncells=2):
    """Integrate over vertex-tagged boundaries of a 1D submesh."""
    mesh = _make_square_with_vertex_sets(ncells=ncells)
    submesh = Submesh(mesh, 1, FACE_TAG_2D)
    markers = submesh.exterior_facets.unique_markers
    assert markers is not None
    marker_set = set(markers)
    assert VTX_A in marker_set
    assert VTX_B in marker_set
    assert abs(assemble(Constant(1.) * ds(VTX_A, domain=submesh)) - 1.0) < 1e-12
    assert abs(assemble(Constant(1.) * ds(VTX_B, domain=submesh)) - 1.0) < 1e-12


@pytest.mark.parallel(nprocs=2)
def test_submesh_codim1_vertex_sets_ds_2_procs():
    _check_vertex_sets_ds()


@pytest.mark.parallel(nprocs=3)
def test_submesh_codim1_vertex_sets_ds_3_procs():
    _check_vertex_sets_ds(ncells=4)

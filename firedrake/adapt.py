"""Adaptive mesh refinement helpers."""
import numpy as np
import petsctools
from ufl import max_value

from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Mesh, DISTRIBUTION_PARAMETERS_NOOP
from firedrake.netgen import _snap_to_netgen, _curve_netgen_mesh
from firedrake.petsc import PETSc


# PETSc's DMAdaptFlag value requesting refinement, for the adapt label.
DM_ADAPT_REFINE = 1

ADAPT_LABEL = "_adaptive_dmplex_adapt"


def _adapt_marked_cells(mesh, cell_marker):
    """Refine the cells of ``mesh`` marked by ``cell_marker`` and return the refined DMPlex."""
    dm = mesh.topology_dm
    ncoarse = mesh.cell_set.size

    # Save the transform, so that the refined DMPlex can tell which of its
    # points came from which point of ``dm``.
    dm.setSaveTransform()

    with PETSc.Log.Event("AdaptiveRefine: mark cells"):
        dm.createLabel(ADAPT_LABEL)
        adapt_label = dm.getLabel(ADAPT_LABEL)
        adapt_indicator = np.zeros(cell_marker.dat.data_ro_with_halos.shape, dtype=IntType)
        adapt_indicator[:ncoarse] = cell_marker.dat.data_ro.real > 0
        dmcommon.mark_points_with_function_array(
            dm, cell_marker.function_space().dm.getLocalSection(), 0,
            adapt_indicator, adapt_label, DM_ADAPT_REFINE,
        )

    parameters = {"dm_plex_transform_type": "refine_sbr"}
    try:
        # options_prefix="" is essential
        with petsctools.inserted_options(parameters=parameters, options_prefix=""):
            with PETSc.Log.Event("AdaptiveRefine: adaptLabel"):
                new_dm = dm.adaptLabel(ADAPT_LABEL)
    finally:
        # Ensure the temporary label is removed even if adaptation fails
        dm.removeLabel(ADAPT_LABEL)

    # The transform propagates every label, including the temporary adapt
    # label and the coarse mesh's stale pyop2_core/owned/ghost point
    # classification. Mesh() skips recomputing that classification if it's
    # already present, so it must be dropped here to force a fresh one for
    # the new mesh's own point count and distribution.
    for label in ("pyop2_core", "pyop2_owned", "pyop2_ghost", ADAPT_LABEL):
        if new_dm.hasLabel(label):
            new_dm.removeLabel(label)

    return new_dm


def _copy_adaptive_refinement_metadata(source_mesh, target_mesh):
    """Copy mesh-construction metadata from a mesh onto its adaptively-derived successor."""
    target_mesh._distribution_parameters = dict(source_mesh._distribution_parameters)
    target_mesh._did_reordering = source_mesh._did_reordering
    target_mesh._tolerance = source_mesh.tolerance


def refine_marked_elements(mesh, cell_marker):
    """Adaptively refine or coarsen a mesh using a DG0 marking function.

    Positive integer marker values request repeated refinement of the
    corresponding cells, and negative values request coarsening. After each
    round, the vertices of a Netgen mesh are snapped onto its geometry. At the
    end, the coordinates are curved to their original degree.

    Parameters
    ----------
    mesh
        The mesh to adapt.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``: cells with a
        positive value ``n`` are refined ``n`` times. Cells with a negative
        value ``-n`` ask to undo ``n`` rounds of refinement, which can go
        past the adaptive parent of ``mesh`` to its own ancestors, but not
        past the mesh that has no adaptive parent. A cell of
        an ancestor is coarsened only as far as all the cells that it was
        refined into ask.

    Returns
    -------
    MeshGeometry
        The adapted mesh. Its ``_adaptive_parent`` is ``mesh`` after a
        refinement, and the adaptive parent of ``mesh``, coarsened as far as
        needed, after a coarsening.
        A marker with both signs first coarsens ``mesh``, and then refines
        the coarsened mesh, which becomes the adaptive parent. Its ``_adaptive_fine_to_coarse_points`` is the DMPlex point of that
        parent that each of its DMPlex points was refined from.

    """
    with cell_marker.dat.vec_ro as v:
        _, num_coarsenings = v.min()
        _, num_refinements = v.max()
    if num_coarsenings < 0 and mesh._adaptive_parent is not None:
        coarsened = _coarsen_marked_elements(mesh, cell_marker)
        if num_refinements <= 0:
            return coarsened
        # A cell with a positive marker stops the coarsening of its ancestor,
        # so the coarsened mesh still has the cells that the positive markers refine.
        from firedrake.mg.interface import prolong
        refinements = Function(cell_marker.function_space()).interpolate(max_value(cell_marker, 0))
        return refine_marked_elements(coarsened, prolong(refinements, Function(FunctionSpace(coarsened, "DG", 0))))
    # Always run at least one adaptation pass, even when no cell is marked,
    # so that a fresh mesh (with its own cell maps) is produced uniformly.
    num_refinements = max(int(np.rint(num_refinements)), 1)

    current_mesh = mesh
    current_mark = cell_marker
    fine_to_coarse_points = np.arange(*mesh.topology_dm.getChart(), dtype=IntType)
    is_netgen = hasattr(mesh, "netgen_mesh")
    for ref in range(num_refinements):
        new_dm = _adapt_marked_cells(current_mesh, current_mark)
        if is_netgen:
            ngmesh = _snap_to_netgen(new_dm, mesh.netgen_mesh)
        fine_to_coarse_points = impl.compose_points(
            fine_to_coarse_points, dmcommon.transform_source_points(new_dm))
        with PETSc.Log.Event("AdaptiveRefine: Mesh()"):
            current_mesh = Mesh(
                new_dm,
                dim=mesh.geometric_dimension,
                reorder=False,
                distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                comm=mesh.comm,
                tolerance=mesh.tolerance,
            )
        if is_netgen:
            current_mesh.netgen_mesh = ngmesh
            current_mesh.netgen_flags = mesh.netgen_flags
        if ref < num_refinements - 1:
            with PETSc.Log.Event("AdaptiveRefine: re-mark"):
                # A cell asking for n refinements stays marked until n rounds
                # have happened, so its descendants inherit n minus the number
                # of rounds so far.
                _, fine_to_coarse = impl.coarse_to_fine_cells(mesh, current_mesh, fine_to_coarse_points)
                ancestor = fine_to_coarse[:, 0]
                refined = ancestor >= 0
                current_mark = Function(FunctionSpace(current_mesh, "DG", 0))
                current_mark.dat.data_wo[refined] = \
                    cell_marker.dat.data_ro[ancestor[refined]] - (ref + 1)

    final_mesh = current_mesh
    if is_netgen:
        coordinates = mesh.coordinates.function_space()
        with PETSc.Log.Event("AdaptiveRefine: recurve netgen coords"):
            final_mesh = _curve_netgen_mesh(final_mesh, coordinates.ufl_element().degree(),
                                            cg_field=not coordinates.finat_element.is_dg())

    final_mesh._adaptive_parent = mesh
    final_mesh._adaptive_fine_to_coarse_points = fine_to_coarse_points
    final_mesh._adaptive_marker = cell_marker.copy(deepcopy=True)
    _copy_adaptive_refinement_metadata(mesh, final_mesh)
    return final_mesh


def _coarsen_marked_elements(mesh, cell_marker):
    """Refine the adaptive parent of ``mesh`` again, with fewer rounds where ``cell_marker`` asks."""
    parent = mesh._adaptive_parent
    if parent is None:
        raise ValueError("Only an adaptively refined mesh can be coarsened")
    coarse_to_fine, _ = impl.coarse_to_fine_cells(parent, mesh, mesh._adaptive_fine_to_coarse_points)
    # The -1 padding of coarse_to_fine reads the appended entry, which never raises the maximum.
    requests = np.append(np.rint(cell_marker.dat.data_ro.real), -np.inf)
    rounds = np.maximum(-requests[coarse_to_fine].max(axis=1), 0)
    stored = mesh._adaptive_marker.dat.data_ro.real
    marker = Function(mesh._adaptive_marker.function_space())
    marker.dat.data_wo[:] = np.maximum(stored - rounds, 0)

    # The rounds that the refinement of the parent cannot undo coarsen the parent itself.
    remaining = Function(marker.function_space())
    remaining.dat.data_wo[:] = np.minimum(stored - rounds, 0)
    with remaining.dat.vec_ro as v:
        _, most_remaining = v.min()
    if most_remaining < 0 and parent._adaptive_parent is not None:
        from firedrake.mg.interface import prolong
        coarsened = refine_marked_elements(parent, remaining)
        marker = prolong(marker, Function(FunctionSpace(coarsened, "DG", 0)))
        parent = coarsened
    return refine_marked_elements(parent, marker)


def _adaptive_ancestors(mesh):
    """Yield ``mesh`` and each of its adaptive ancestors, with the map of the DMPlex points of ``mesh`` to theirs."""
    points = np.arange(*mesh.topology_dm.getChart(), dtype=IntType)
    while mesh is not None:
        yield mesh, points
        if mesh._adaptive_parent is not None:
            points = impl.compose_points(mesh._adaptive_fine_to_coarse_points, points)
        mesh = mesh._adaptive_parent


def transfer_cell_maps(source, target):
    """Return the candidate cell maps for the transfer between two adapted meshes.

    The candidates of a cell are the cells of the other mesh that come from
    the same cell of the closest common adaptive ancestor. Each mesh can be
    finer than the other in some regions and coarser in others.

    Parameters
    ----------
    source, target
        The meshes to transfer between.

    Returns
    -------
    tuple or None
        The ``coarse_to_fine_cells`` and ``fine_to_coarse_cells`` arrays that
        `~firedrake.mg.mesh.HierarchyBase` takes for the levels
        ``[source, target]``, or ``None`` if the meshes have no common
        adaptive ancestor.

    """
    target_ancestors = list(_adaptive_ancestors(target))
    for ancestor, source_points in _adaptive_ancestors(source):
        target_points = next((p for m, p in target_ancestors if m is ancestor), None)
        if target_points is not None:
            break
    else:
        return None
    ancestor_to_source, source_to_ancestor = impl.coarse_to_fine_cells(ancestor, source, source_points)
    ancestor_to_target, target_to_ancestor = impl.coarse_to_fine_cells(ancestor, target, target_points)
    source_to_target = ancestor_to_target[source_to_ancestor[:, 0]]
    target_to_source = ancestor_to_source[target_to_ancestor[:, 0]]
    # Every ancestor cell has a descendant, so the first entry of a row can replace its -1 padding.
    return tuple(np.where(m < 0, m[:, :1], m) for m in (source_to_target, target_to_source))

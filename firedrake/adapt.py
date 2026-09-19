"""Adaptive mesh refinement helpers."""
import numpy as np
import petsctools

from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Mesh, Submesh, DISTRIBUTION_PARAMETERS_NOOP
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
    if hasattr(source_mesh, "netgen_mesh") and not hasattr(target_mesh, "netgen_mesh"):
        target_mesh.netgen_mesh = source_mesh.netgen_mesh
    if hasattr(source_mesh, "netgen_flags") and not hasattr(target_mesh, "netgen_flags"):
        target_mesh.netgen_flags = source_mesh.netgen_flags


def _redistribute_adaptive_refined_mesh(coarse_mesh, refined_mesh, redistribute=True):
    """Redistribute an adaptively refined mesh if it has empty ranks.

    Parameters
    ----------
    coarse_mesh : firedrake.mesh.MeshGeometry
        The mesh that was refined.
    refined_mesh : firedrake.mesh.MeshGeometry
        The result of refining ``coarse_mesh``.
    redistribute : bool
        If ``True``, redistribute ``refined_mesh`` when it has empty ranks.

    Returns
    -------
    firedrake.mesh.MeshGeometry
        ``refined_mesh``, or a redistributed `~firedrake.mesh.Submesh` of it.

    """
    _copy_adaptive_refinement_metadata(coarse_mesh, refined_mesh)
    if not (redistribute and refined_mesh.has_empty_rank):
        return refined_mesh
    redist_mesh = Submesh(refined_mesh, redistribute=True, name=refined_mesh.name)
    _copy_adaptive_refinement_metadata(refined_mesh, redist_mesh)
    return redist_mesh


def refine_marked_elements(mesh, cell_marker, redistribute=True):
    """Adaptively refine a mesh using a DG0 marking function.

    Positive integer marker values request repeated refinement of the
    corresponding cells. The vertices of a Netgen mesh are snapped onto its
    geometry after each round, and the coordinates are curved to their
    original degree at the end.

    Parameters
    ----------
    mesh
        The mesh to refine.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``: cells with a
        positive value ``n`` are refined ``n`` times.
    redistribute
        If ``True``, redistribute the refined mesh when the coarse mesh
        has empty ranks.

    Returns
    -------
    MeshGeometry
        The adaptively refined mesh, with ``_adaptive_parent`` set to
        ``mesh`` and ``_adaptive_fine_to_coarse_points`` set to the DMPlex
        point of ``mesh`` that each of its DMPlex points was refined from.

    """
    with cell_marker.dat.vec_ro as v:
        _, num_refinements = v.max()
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
            fine_to_coarse_points, impl.transform_source_points(new_dm))
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

    # The redistribution step can return a different mesh, so record the
    # refinement provenance on whichever mesh it returns.
    final_mesh = _redistribute_adaptive_refined_mesh(
        mesh, final_mesh, redistribute=redistribute
    )
    final_mesh._adaptive_parent = mesh
    final_mesh._adaptive_fine_to_coarse_points = fine_to_coarse_points
    return final_mesh

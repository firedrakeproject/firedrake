"""Adaptive mesh refinement helpers."""
import numpy as np
import petsctools

from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Mesh, DISTRIBUTION_PARAMETERS_NOOP
from firedrake.netgen import _transfer_high_order_coordinates
from firedrake.petsc import PETSc


# PETSc's DMAdaptFlag value requesting refinement, for the adapt label.
DM_ADAPT_REFINE = 1

# Label holding, on every cell, the number of the cell of the original mesh it
# descends from. The refinement transform propagates labels from a cell to its
# children, so this stays relative to the original mesh however many times we
# refine.
PARENT_LABEL = "_adaptive_dmplex_parent"

ADAPT_LABEL = "_adaptive_dmplex_adapt"

REFINED_LABEL = "adaptive_refined"


def _adapt_marked_cells(mesh, cell_marker):
    """Refine the cells of ``mesh`` marked by ``cell_marker`` and return the refined DMPlex."""
    dm = mesh.topology_dm
    ncoarse = mesh.cell_set.size

    with PETSc.Log.Event("AdaptiveRefine: mark cells"):
        dm.createLabel(ADAPT_LABEL)
        adapt_label = dm.getLabel(ADAPT_LABEL)
        adapt_indicator = np.zeros(cell_marker.dat.data_ro_with_halos.shape, dtype=IntType)
        adapt_indicator[:ncoarse] = cell_marker.dat.data_ro.real > 0
        dmcommon.mark_points_with_function_array(
            dm, cell_marker.function_space().dm.getSection(), 0,
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


def refine_marked_elements(mesh, cell_marker):
    """Adaptively refine a mesh using a DG0 marking function.

    Positive integer marker values request repeated refinement of the
    corresponding cells. Curved Netgen meshes are re-curved to the
    original coordinate degree after refinement.

    Parameters
    ----------
    mesh
        The mesh to refine.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``: cells with a
        positive value ``n`` are refined ``n`` times.

    Returns
    -------
    MeshGeometry
        The adaptively refined mesh, with ``adaptive_parent`` set to
        ``mesh`` and ``adaptive_cell_maps`` set to the
        ``(coarse_to_fine, fine_to_coarse)`` cell maps relative to it.

    """
    with cell_marker.dat.vec_ro as v:
        _, num_refinements = v.max()
    # Always run at least one adaptation pass, even when no cell is marked,
    # so that a fresh mesh (with its own cell maps) is produced uniformly.
    num_refinements = max(int(np.rint(num_refinements)), 1)

    coarse_dm = mesh.topology_dm
    with PETSc.Log.Event("AdaptiveRefine: set_adaptive_parent_label"):
        impl.set_adaptive_parent_label(coarse_dm, mesh._cell_numbering, PARENT_LABEL)

    current_mesh = mesh
    current_mark = cell_marker
    try:
        for ref in range(num_refinements):
            new_dm = _adapt_marked_cells(current_mesh, current_mark)
            with PETSc.Log.Event("AdaptiveRefine: Mesh()"):
                current_mesh = Mesh(
                    new_dm,
                    dim=mesh.geometric_dimension,
                    reorder=False,
                    distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
                    comm=mesh.comm,
                    tolerance=mesh.tolerance,
                )
            with PETSc.Log.Event("AdaptiveRefine: adaptive_parent_child_cell_maps"):
                coarse_to_fine, fine_to_coarse = impl.adaptive_parent_child_cell_maps(
                    coarse_dm, new_dm, current_mesh._cell_numbering, PARENT_LABEL
                )
            if ref < num_refinements - 1:
                with PETSc.Log.Event("AdaptiveRefine: re-mark"):
                    # A cell asking for n refinements stays marked until n rounds
                    # have happened, so its descendants inherit n minus the number
                    # of rounds so far.
                    ancestor = fine_to_coarse[:, 0]
                    refined = ancestor >= 0
                    current_mark = Function(FunctionSpace(current_mesh, "DG", 0))
                    current_mark.dat.data_wo[refined] = \
                        cell_marker.dat.data_ro[ancestor[refined]] - (ref + 1)
    finally:
        # Ensure the temporary label is removed even if adaptation fails
        coarse_dm.removeLabel(PARENT_LABEL)

    final_mesh = current_mesh
    if hasattr(mesh, "netgen_mesh"):
        order = mesh.coordinates.function_space().ufl_element().degree()
        if order > 1:
            with PETSc.Log.Event("AdaptiveRefine: recurve netgen coords"):
                final_mesh = _transfer_high_order_coordinates(mesh, final_mesh, order)

    final_mesh.topology_dm.removeLabel(PARENT_LABEL)
    final_mesh.adaptive_parent = mesh
    final_mesh.adaptive_cell_maps = (coarse_to_fine, fine_to_coarse)
    _copy_adaptive_refinement_metadata(mesh, final_mesh)
    return final_mesh


def mark_refined_entities(mesh, name: str = REFINED_LABEL, value: int = 1):
    """Label the mesh entities whose star contains a genuinely refined cell.

    A cell of ``mesh`` counts as genuinely refined when the cell of
    `~firedrake.mesh.MeshGeometry.adaptive_parent` it descends from was split
    into more than one cell. The label holds those cells and their closure,
    which is exactly the set of entities whose star meets the refined region.

    This is the set of entities a local smoother on an adaptively refined
    level wants to build patches around: pass the label's name to
    ``pc_star_construct_label`` (`~firedrake.preconditioners.asm.ASMStarPC`)
    or to ``patch_pc_patch_construct_label``
    (`~firedrake.preconditioners.patch.PatchPC`).

    Parameters
    ----------
    mesh
        A mesh produced by `refine_marked_elements`.
    name
        The name to give the label on ``mesh.topology_dm``. An existing label
        of that name is overwritten.
    value
        The stratum to mark the entities with.

    Returns
    -------
    PETSc.DMLabel or None
        The label, which is also left on ``mesh.topology_dm``, or `None` if
        ``mesh`` has no `~firedrake.mesh.MeshGeometry.adaptive_parent`.

    Notes
    -----
    A mesh with no adaptive parent, such as the coarsest mesh of a hierarchy or
    a uniformly refined level, has no refined region to single out, so this
    returns `None` rather than a label. A smoother reads that as relaxing
    everywhere, which is what a level that is uniformly new needs.

    """
    if mesh.adaptive_cell_maps is None:
        return None
    coarse_to_fine, fine_to_coarse = mesh.adaptive_cell_maps

    # A fine cell was genuinely split if its parent has more than one child.
    # Conformity fixups mean that this is not the same as the cells that were
    # marked for refinement.
    parent = fine_to_coarse[:, 0]
    refined = parent >= 0
    refined[refined] = np.count_nonzero(coarse_to_fine[parent[refined]] >= 0, axis=1) > 1

    V = FunctionSpace(mesh, "DG", 0)
    indicator = np.zeros(V.dof_dset.total_size, dtype=IntType)
    indicator[:mesh.cell_set.size] = refined[:mesh.cell_set.size]

    dm = mesh.topology_dm
    if dm.hasLabel(name):
        dm.removeLabel(name)
    dm.createLabel(name)
    label = dm.getLabel(name)
    dmcommon.mark_points_with_function_array(dm, V.dm.getSection(), 0, indicator, label, value)
    # The closure of the refined cells is precisely the entities whose star
    # contains one of them
    dm.labelComplete(label)
    return label

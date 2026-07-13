"""Adaptive mesh refinement helpers."""
import numpy as np

from pyop2.mpi import MPI
from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.petsc import PETSc
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Mesh, DISTRIBUTION_PARAMETERS_NOOP
from firedrake.netgen import _recurve_netgen_mesh
from firedrake.redist import (RedistributedMeshTransfer, dm_has_empty_rank,
                              redistribute_dm)


DM_ADAPT_REFINE = 1


def _refine_marked_elements_once(mesh, cell_marker):
    """Refine marked cells once and return parent-child cell maps."""
    dm = mesh.topology_dm
    ncoarse = mesh.cell_set.size

    parent_name = "_adaptive_dmplex_parent"
    adapt_name = "_adaptive_dmplex_adapt"
    dm.createLabel(parent_name)
    dm.createLabel(adapt_name)
    impl.set_adaptive_parent_label(dm, mesh._cell_numbering, ncoarse, parent_name)
    adapt_label = dm.getLabel(adapt_name)
    adapt_indicator = np.zeros(cell_marker.dat.data_ro_with_halos.shape, dtype=IntType)
    adapt_indicator[:ncoarse] = cell_marker.dat.data_ro.real > 0
    dmcommon.mark_points_with_function_array(
        dm, cell_marker.function_space().dm.getSection(), 0,
        adapt_indicator, adapt_label, DM_ADAPT_REFINE,
    )

    opts = PETSc.Options()
    had_prev = opts.hasName("dm_plex_transform_type")
    prev = opts.getString("dm_plex_transform_type", "") if had_prev else None
    opts["dm_plex_transform_type"] = "refine_sbr"
    try:
        new_dm = dm.adaptLabel(adapt_name)
    finally:
        if had_prev:
            opts["dm_plex_transform_type"] = prev
        else:
            opts.delValue("dm_plex_transform_type")
        dm.removeLabel(parent_name)
        dm.removeLabel(adapt_name)

    # The transform propagates every label, including the temporary adapt label.
    for label in ("pyop2_core", "pyop2_owned", "pyop2_ghost", adapt_name):
        if new_dm.hasLabel(label):
            new_dm.removeLabel(label)

    new_mesh = Mesh(
        new_dm,
        dim=mesh.geometric_dimension,
        reorder=False,
        distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
        comm=mesh.comm,
        tolerance=mesh.tolerance,
    )

    coarse_to_fine, fine_to_coarse = impl.adaptive_parent_child_cell_maps(
        new_dm, new_mesh._cell_numbering, ncoarse, new_mesh.cell_set.size,
        parent_name,
    )
    new_dm.removeLabel(parent_name)

    return new_mesh, coarse_to_fine, fine_to_coarse


def _copy_adaptive_refinement_metadata(source_mesh, target_mesh):
    """Copy mesh-construction metadata from a mesh onto its adaptively-derived successor."""
    target_mesh._distribution_parameters = dict(source_mesh._distribution_parameters)
    target_mesh._did_reordering = source_mesh._did_reordering
    target_mesh._tolerance = source_mesh.tolerance
    if hasattr(source_mesh, "netgen_mesh") and not hasattr(target_mesh, "netgen_mesh"):
        target_mesh.netgen_mesh = source_mesh.netgen_mesh
    if hasattr(source_mesh, "netgen_flags") and not hasattr(target_mesh, "netgen_flags"):
        target_mesh.netgen_flags = source_mesh.netgen_flags


def _needs_adaptive_redistribution(mesh):
    """Return whether ``mesh`` has an empty rank."""
    return dm_has_empty_rank(mesh.topology_dm)


def _redistribute_adaptive_refined_mesh(coarse_mesh, transfer_mesh,
                                        redistribute=True):
    """Redistribute an adaptively refined mesh if the coarse mesh has empty ranks."""
    _copy_adaptive_refinement_metadata(coarse_mesh, transfer_mesh)

    needs_redist = (redistribute and coarse_mesh.comm.size > 1
                    and _needs_adaptive_redistribution(transfer_mesh))
    if not needs_redist:
        return transfer_mesh

    redist_parameters = dict(coarse_mesh._distribution_parameters)
    redist_parameters["partition"] = True
    redist_dm = transfer_mesh.topology_dm.clone()
    _, point_sf = redistribute_dm(redist_dm, redist_parameters)

    redist_topology_mesh = Mesh(
        redist_dm,
        dim=transfer_mesh.geometric_dimension,
        reorder=False,
        distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
        comm=transfer_mesh.comm,
        tolerance=transfer_mesh.tolerance,
    )
    _copy_adaptive_refinement_metadata(transfer_mesh, redist_topology_mesh)

    redist_transfer = RedistributedMeshTransfer(
        transfer_mesh, redist_topology_mesh, point_sf
    )
    Vredist = transfer_mesh.coordinates.function_space().reconstruct(
        mesh=redist_topology_mesh
    )
    redist_coordinates = Function(Vredist)
    redist_transfer.orig2redist(transfer_mesh.coordinates, redist_coordinates)
    redist_mesh = Mesh(redist_coordinates, name=transfer_mesh.name)
    _copy_adaptive_refinement_metadata(redist_topology_mesh, redist_mesh)
    redist_mesh.redist = RedistributedMeshTransfer(
        transfer_mesh, redist_mesh, point_sf
    )
    return redist_mesh


def refine_marked_elements(mesh, cell_marker, redistribute=True):
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
    redistribute
        If ``True``, redistribute the refined mesh when the coarse mesh
        has empty ranks.

    Returns
    -------
    MeshGeometry
        The adaptively refined mesh, with ``_adaptive_cell_maps`` set
        to the ``(coarse_to_fine, fine_to_coarse)`` cell maps relative
        to ``mesh``.
    """
    with cell_marker.dat.vec_ro as v:
        _, max_rounds = v.max()
    # Always run at least one adaptation pass, even when no cell is marked,
    # so that a fresh mesh (with its own cell maps) is produced uniformly.
    max_rounds = max(int(np.rint(max_rounds)), 1)

    current_mesh = mesh
    current_mark = cell_marker
    fine_to_coarse_total = None
    for round_idx in range(max_rounds):
        new_mesh, _, f2c = _refine_marked_elements_once(current_mesh, current_mark)
        parent = f2c[:, 0]
        if fine_to_coarse_total is None:
            fine_to_coarse_total = f2c.copy()
        else:
            composed = np.full_like(f2c, -1)
            valid = parent >= 0
            composed[valid, 0] = fine_to_coarse_total[parent[valid], 0]
            fine_to_coarse_total = composed

        if round_idx < max_rounds - 1:
            next_mark = Function(FunctionSpace(new_mesh, "DG", 0))
            valid = (parent >= 0)
            next_mark.dat.data_wo[valid] = np.maximum(current_mark.dat.data_ro[parent[valid]] - 1, 0)
            current_mark = next_mark
        current_mesh = new_mesh

    ncoarse = mesh.cell_set.size
    children = [[] for _ in range(ncoarse)]
    for fine_cell, parent in enumerate(fine_to_coarse_total[:, 0]):
        if parent >= 0:
            children[parent].append(fine_cell)
    max_children = max((len(c) for c in children), default=0)
    max_children = mesh.comm.allreduce(max_children, MPI.MAX)
    coarse_to_fine_total = np.full((ncoarse, max_children), -1, dtype=IntType)
    for coarse_cell, fine_cells in enumerate(children):
        coarse_to_fine_total[coarse_cell, :len(fine_cells)] = fine_cells

    final_mesh = current_mesh
    if hasattr(mesh, "netgen_mesh"):
        order = mesh.coordinates.function_space().ufl_element().degree()
        if order > 1:
            final_mesh = _recurve_netgen_mesh(mesh, final_mesh, order)

    final_mesh._adaptive_cell_maps = (coarse_to_fine_total, fine_to_coarse_total)
    final_mesh = _redistribute_adaptive_refined_mesh(
        mesh, final_mesh, redistribute=redistribute
    )
    final_mesh._adaptive_cell_maps = (coarse_to_fine_total, fine_to_coarse_total)
    redist = getattr(final_mesh, "redist", None)
    if redist is not None:
        redist.orig._adaptive_cell_maps = (coarse_to_fine_total, fine_to_coarse_total)
    return final_mesh

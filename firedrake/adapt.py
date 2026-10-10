"""Adaptive mesh refinement helpers."""
import numpy as np
import petsctools

from firedrake.cython import dmcommon
from firedrake.cython import mgimpl as impl
from firedrake.utils import IntType
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import Mesh, MeshGeometry, MeshSequenceGeometry, DISTRIBUTION_PARAMETERS_NOOP
from firedrake.netgen import _snap_to_netgen, _curve_netgen_mesh
from firedrake.petsc import PETSc


# PETSc's DMAdaptFlag values requesting refinement and coarsening, for the adapt label.
DM_ADAPT_REFINE = 1
DM_ADAPT_COARSEN = 2

ADAPT_LABEL = "_adaptive_dmplex_adapt"


def _adapt_marked_cells(mesh: MeshGeometry, cell_marker: Function) -> PETSc.DMPlex:
    """Refine and coarsen the marked cells of a mesh in one round.

    Parameters
    ----------
    mesh
        The mesh to adapt.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``. The cells with a
        positive value are refined, and the cells with a negative value are
        coarsened.

    Returns
    -------
    PETSc.DMPlex
        The adapted DMPlex. After a coarsening, it can be the DMPlex of an
        adaptive ancestor of ``mesh``, and otherwise its coarse DM is the
        DMPlex that it was transformed from. If no cell changes, its coarse
        DM is not set.

    """
    dm = mesh.topology_dm
    ncoarse = mesh.cell_set.size

    # Save the transform, so that the adapted DMPlex can tell which of its
    # points came from which point of the DMPlex that it was transformed from.
    dm.setSaveTransform()

    with PETSc.Log.Event("AdaptiveRefine: mark cells"):
        dm.createLabel(ADAPT_LABEL)
        adapt_label = dm.getLabel(ADAPT_LABEL)
        section = cell_marker.function_space().dm.getLocalSection()
        values = cell_marker.dat.data_ro.real
        for flag, marked in ((DM_ADAPT_REFINE, values > 0), (DM_ADAPT_COARSEN, values < 0)):
            adapt_indicator = np.zeros(cell_marker.dat.data_ro_with_halos.shape, dtype=IntType)
            adapt_indicator[:ncoarse] = marked
            dmcommon.mark_points_with_function_array(dm, section, 0, adapt_indicator, adapt_label, flag)

    parameters = {"dm_plex_transform_type": "refine_sbr"}
    try:
        # options_prefix="" is essential
        with petsctools.inserted_options(parameters=parameters, options_prefix=""):
            with PETSc.Log.Event("AdaptiveRefine: adaptLabel"):
                new_dm = dm.adaptLabel(ADAPT_LABEL)
    finally:
        # Ensure the temporary label is removed even if adaptation fails
        dm.removeLabel(ADAPT_LABEL)
    return new_dm


def _copy_adaptive_refinement_metadata(source_mesh, target_mesh):
    """Copy mesh-construction metadata from a mesh onto its adaptively-derived successor."""
    target_mesh._distribution_parameters = dict(source_mesh._distribution_parameters)
    target_mesh._did_reordering = source_mesh._did_reordering
    target_mesh._tolerance = source_mesh.tolerance


def refine_marked_elements(mesh, cell_marker):
    """Adaptively refine or coarsen a mesh using a DG0 marking function.

    Positive integer marker values request repeated refinement of the
    corresponding cells, and negative values request one round of
    coarsening. PETSc coarsens a cell of an adaptive ancestor only if all the
    cells that it was refined into ask. The vertices of a Netgen mesh are
    snapped onto its geometry after each round, and the coordinates are
    curved to their original degree at the end.

    Parameters
    ----------
    mesh
        The mesh to adapt.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``: cells with a
        positive value ``n`` are refined ``n`` times. If any value is
        negative, only one round is done, which can coarsen past the adaptive
        parent of ``mesh``.

    Returns
    -------
    MeshGeometry
        The adapted mesh, with ``_adaptive_parent`` set to the closest
        adaptive ancestor that it was transformed from, and
        ``_adaptive_fine_to_coarse_points`` set to the DMPlex point of that
        ancestor that each of its DMPlex points comes from. This is ``mesh``
        itself if no cell changes, or an adaptive ancestor of ``mesh`` if the
        coarsening undoes all the refinement after it.

    """
    marker = Function(cell_marker.function_space())
    marker.dat.data_wo[:] = np.rint(cell_marker.dat.data_ro.real)
    with marker.dat.vec_ro as v:
        _, lowest = v.min()
        _, highest = v.max()
    # A coarsening makes a mesh that is not a refinement of ``mesh``, so the
    # marker values cannot be carried over to another round.
    num_rounds = 1 if lowest < 0 else int(highest)
    if num_rounds <= 0:
        return mesh

    ancestors = {m.topology_dm.handle: m for m, _ in _adaptive_ancestors(mesh)}
    current_mesh = mesh
    current_mark = marker
    is_netgen = hasattr(mesh, "netgen_mesh")
    for ref in range(num_rounds):
        new_dm = _adapt_marked_cells(current_mesh, current_mark)
        if new_dm.handle in ancestors:
            # The coarsening undid all the refinement after this ancestor.
            return ancestors[new_dm.handle]
        if not new_dm.getCoarseDM().handle:
            # PETSc set no coarse DM, so it did not coarsen.
            if highest <= 0:
                return mesh
            new_dm.setCoarseDM(current_mesh.topology_dm)
        # Follow the coarse DMs back to the closest adaptive ancestor.
        dm = new_dm
        fine_to_coarse_points = np.arange(*dm.getChart(), dtype=IntType)
        while dm.handle not in ancestors:
            fine_to_coarse_points = impl.compose_points(dmcommon.transform_source_points(dm), fine_to_coarse_points)
            dm = dm.getCoarseDM()
        # The transform propagates every label, including the temporary adapt
        # label and the coarse mesh's stale pyop2_core/owned/ghost point
        # classification. Mesh() skips recomputing that classification if it's
        # already present, so it must be dropped here to force a fresh one for
        # the new mesh's own point count and distribution.
        for label in ("pyop2_core", "pyop2_owned", "pyop2_ghost", ADAPT_LABEL):
            if new_dm.hasLabel(label):
                new_dm.removeLabel(label)
        if is_netgen:
            ngmesh = _snap_to_netgen(new_dm, mesh.netgen_mesh)
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
        if ref < num_rounds - 1:
            with PETSc.Log.Event("AdaptiveRefine: re-mark"):
                # A cell asking for n refinements stays marked until n rounds
                # have happened, so its descendants inherit n minus the number
                # of rounds so far.
                _, fine_to_coarse = impl.coarse_to_fine_cells(mesh, current_mesh, fine_to_coarse_points)
                ancestor = fine_to_coarse[:, 0]
                refined = ancestor >= 0
                current_mark = Function(FunctionSpace(current_mesh, "DG", 0))
                current_mark.dat.data_wo[refined] = \
                    marker.dat.data_ro[ancestor[refined]] - (ref + 1)

    final_mesh = current_mesh
    if is_netgen:
        coordinates = mesh.coordinates.function_space()
        with PETSc.Log.Event("AdaptiveRefine: recurve netgen coords"):
            final_mesh = _curve_netgen_mesh(final_mesh, coordinates.ufl_element().degree(),
                                            cg_field=not coordinates.finat_element.is_dg())

    final_mesh._adaptive_parent = ancestors[dm.handle]
    final_mesh._adaptive_fine_to_coarse_points = fine_to_coarse_points
    _copy_adaptive_refinement_metadata(mesh, final_mesh)
    return final_mesh


def _adaptive_ancestors(mesh: MeshGeometry):
    """Yield ``mesh`` and each of its adaptive ancestors, with the map of the DMPlex points of ``mesh`` to theirs."""
    points = np.arange(*mesh.topology_dm.getChart(), dtype=IntType)
    while mesh is not None:
        yield mesh, points
        if mesh._adaptive_parent is not None:
            points = impl.compose_points(mesh._adaptive_fine_to_coarse_points, points)
        mesh = mesh._adaptive_parent


def adapted_cell_maps(coarse: MeshGeometry, fine: MeshGeometry) -> tuple | None:
    """Return the cell maps between two meshes adapted from a common ancestor.

    A cell of one mesh is mapped to the cells of the other that come from
    the same cell of the closest common adaptive ancestor. If ``coarse`` is
    an adaptive ancestor of ``fine``, these are the children of each coarse
    cell and the parent of each fine cell. In general, each mesh can be
    finer than the other in some regions and coarser in others.

    Parameters
    ----------
    coarse, fine
        The meshes on the coarse and fine levels of a hierarchy.

    Returns
    -------
    tuple or None
        The ``coarse_to_fine_cells`` and ``fine_to_coarse_cells`` arrays that
        `~firedrake.mg.mesh.HierarchyBase` takes for the levels
        ``[coarse, fine]``, and the ``fine_to_coarse_points`` of ``fine``
        if ``coarse`` is its adaptive ancestor, or else ``None``. The result
        is ``None`` if the meshes have no common adaptive ancestor.

    """
    fine_ancestors = dict(_adaptive_ancestors(fine))
    for ancestor, coarse_points in _adaptive_ancestors(coarse):
        if ancestor in fine_ancestors:
            break
    else:
        return None
    fine_points = fine_ancestors[ancestor]
    ancestor_to_coarse, coarse_to_ancestor = impl.coarse_to_fine_cells(ancestor, coarse, coarse_points)
    ancestor_to_fine, fine_to_ancestor = impl.coarse_to_fine_cells(ancestor, fine, fine_points)
    coarse_to_fine = ancestor_to_fine[coarse_to_ancestor[:, 0]]
    fine_to_coarse = ancestor_to_coarse[fine_to_ancestor[:, 0]]
    # op2.Map cannot hold the -1 padding of fine_to_coarse, and every ancestor
    # cell has a descendant, so the first entry of a row can replace it.
    fine_to_coarse = np.where(fine_to_coarse < 0, fine_to_coarse[:, :1], fine_to_coarse)
    fine_to_coarse_points = fine_points if ancestor is coarse else None
    return coarse_to_fine, fine_to_coarse, fine_to_coarse_points


def follow_adaptive_parents(mesh: MeshGeometry | MeshSequenceGeometry) -> None:
    """Make each component of an adapted mesh the finest level of its hierarchy.

    The levels above the nearest adaptive ancestor that is a level are
    replaced by the chain of adaptive parents.

    Parameters
    ----------
    mesh
        An adapted mesh, or a sequence of adapted meshes.

    """
    from firedrake.mg.utils import get_level

    for component in set(mesh):
        hierarchy, _ = get_level(component)
        chain = [component]
        ancestor = component._adaptive_parent
        while ancestor is not None and ancestor not in hierarchy:
            chain.append(ancestor)
            ancestor = ancestor._adaptive_parent
        if ancestor is None:
            ancestor = hierarchy[0]
            chain = [] if component is ancestor else [component]
        target = [*hierarchy[:hierarchy.meshes.index(ancestor) + 1], *reversed(chain)]
        # The levels that already match the target keep their cell maps.
        keep = 0
        while keep < min(len(hierarchy), len(target)) and hierarchy[keep] is target[keep]:
            keep += 1
        while len(hierarchy) > keep:
            hierarchy.remove_mesh()
        for m in target[keep:]:
            hierarchy.add_mesh(m)
    if isinstance(mesh, MeshSequenceGeometry):
        mesh.set_hierarchy()

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

    Each round of adaptation refines the cells with a positive marker value
    once, by skeleton-based refinement, and undoes one round of requested
    refinement for the cells with a negative marker value. PETSc coarsens a
    cell of an adaptive ancestor only if all the cells that it was refined
    into ask, and then refines the ancestor again without it. After each
    round, the vertices of a Netgen mesh are snapped onto its geometry. At the
    end, the coordinates are curved to their original degree.

    Parameters
    ----------
    mesh
        The mesh to adapt.
    cell_marker
        A DG0 `~firedrake.function.Function` on ``mesh``. If no value is
        negative, the cells with a positive value ``n`` are refined ``n``
        times. Otherwise, one round is done: the cells with a positive value
        are refined once, and the cells with a negative value are coarsened
        once, which can go past the adaptive parent of ``mesh`` to its own
        ancestors. To coarsen by more rounds, call this function again.

    Returns
    -------
    MeshGeometry
        The adapted mesh, which is ``mesh`` itself if ``cell_marker``
        changes no cell, and an adaptive ancestor of ``mesh`` if the
        coarsening undoes all the refinement after it. Its
        ``_adaptive_parent`` is the closest adaptive ancestor of ``mesh`` that
        it was transformed from: ``mesh`` after a refinement, and an earlier
        ancestor after a coarsening. Its ``_adaptive_fine_to_coarse_points``
        is the DMPlex point of the adaptive parent that each of its DMPlex
        points comes from.

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

    ancestors = _adaptive_ancestor_dms(mesh)
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
                _, fine_to_coarse_points = _closest_adaptive_ancestor(new_dm, ancestors)
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

    final_mesh._adaptive_parent, final_mesh._adaptive_fine_to_coarse_points = \
        _closest_adaptive_ancestor(final_mesh.topology_dm, ancestors)
    _copy_adaptive_refinement_metadata(mesh, final_mesh)
    return final_mesh


def _adaptive_ancestor_dms(mesh: MeshGeometry) -> dict[int, MeshGeometry]:
    """Return ``mesh`` and each of its adaptive ancestors, keyed by the handle of its DMPlex."""
    ancestors = {}
    while mesh is not None:
        ancestors[mesh.topology_dm.handle] = mesh
        mesh = mesh._adaptive_parent
    return ancestors


def _closest_adaptive_ancestor(dm: PETSc.DMPlex, ancestors: dict[int, MeshGeometry]) -> tuple[MeshGeometry, np.ndarray]:
    """Find the mesh that an adapted DMPlex comes from.

    Parameters
    ----------
    dm
        A DMPlex made by `_adapt_marked_cells`, or by earlier rounds of it.
    ancestors
        Meshes keyed by the handle of their DMPlex, as returned by
        `_adaptive_ancestor_dms`.

    Returns
    -------
    tuple
        The first mesh of ``ancestors`` that the chain of coarse DMs of
        ``dm`` reaches, and the map of the DMPlex points of ``dm`` to its
        points, composed from the transforms saved along the chain.

    """
    points = np.arange(*dm.getChart(), dtype=IntType)
    while dm.handle not in ancestors:
        points = impl.compose_points(dmcommon.transform_source_points(dm), points)
        dm = dm.getCoarseDM()
    return ancestors[dm.handle], points


def _adaptive_ancestors(mesh):
    """Yield ``mesh`` and each of its adaptive ancestors, with the map of the DMPlex points of ``mesh`` to theirs."""
    points = np.arange(*mesh.topology_dm.getChart(), dtype=IntType)
    while mesh is not None:
        yield mesh, points
        if mesh._adaptive_parent is not None:
            points = impl.compose_points(mesh._adaptive_fine_to_coarse_points, points)
        mesh = mesh._adaptive_parent


def adapted_cell_maps(coarse, fine):
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
    fine_ancestors = list(_adaptive_ancestors(fine))
    for ancestor, coarse_points in _adaptive_ancestors(coarse):
        fine_points = next((p for m, p in fine_ancestors if m is ancestor), None)
        if fine_points is not None:
            break
    else:
        return None
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
    """Make an adapted mesh the finest level, above its adaptive ancestors.

    For each distinct component of ``mesh``, the chain of adaptive parents
    is followed down to the first mesh that is a level of the hierarchy of
    that component. The levels above that mesh are replaced by the chain
    and the component. If no adaptive ancestor is a level, the component is
    put directly above the coarsest level, unless the component is the
    coarsest level. A `~firedrake.mesh.MeshSequenceGeometry` then takes its
    levels from its components again.

    Parameters
    ----------
    mesh
        A mesh whose components are adapted from levels of their
        hierarchies. A component can itself be a level, if the adaptation
        returned an ancestor of the finest mesh.

    """
    from firedrake.mg.utils import get_level

    for component in set(mesh):
        hierarchy, _ = get_level(component)
        chain = [component]
        ancestor = component._adaptive_parent
        while ancestor is not None:
            ancestor_hierarchy, level = get_level(ancestor)
            if ancestor_hierarchy is hierarchy and hierarchy[level] is ancestor:
                break
            chain.append(ancestor)
            ancestor = ancestor._adaptive_parent
        else:
            chain, level = [component], 0
        target = [*hierarchy[:level + 1], *reversed(chain)]
        if target[0] is component:
            target = [component]
        # The levels that already match the target keep their cell maps.
        keep = level + 1
        while keep < min(len(hierarchy), len(target)) and hierarchy[keep] is target[keep]:
            keep += 1
        while len(hierarchy) > keep:
            hierarchy.remove_mesh()
        for m in target[keep:]:
            hierarchy.add_mesh(m)
    if isinstance(mesh, MeshSequenceGeometry):
        mesh.set_hierarchy()

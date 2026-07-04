from collections import defaultdict
from fractions import Fraction

import numpy as np

from firedrake.mesh import Mesh, MeshGeometry
from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.halo import _get_mtype as get_mpi_type
from firedrake.halo import MPI
from firedrake.mg import HierarchyBase
from firedrake.mg.utils import set_level
from firedrake.netgen import netgen_distribute
from firedrake.utils import IntType

__all__ = ["AdaptiveMeshHierarchy"]


class AdaptiveMeshHierarchy(HierarchyBase):
    """
    HierarchyBase for hierarchies of adaptively refined meshes.

    Parameters
    ----------
    base_mesh
        The coarsest mesh in the hierarchy.
    nested: bool
        A flag to indicate whether the meshes are nested.
    redistribute: bool
        If ``True``, keep adaptively refined meshes redistributed when
        this is needed to avoid empty ranks and use an internal
        parent-owned mesh for transfer operators.

    """
    def __init__(self, base_mesh: MeshGeometry, nested: bool = True,
                 redistribute: bool = True):
        self.meshes = []
        self._meshes = []
        self.coarse_to_fine_cells = {}
        self.fine_to_coarse_cells = {Fraction(0, 1): None}
        self.refinements_per_level = 1
        self.nested = nested
        self.redistribute = redistribute
        self._shared_data_cache = defaultdict(dict)
        self.add_mesh(base_mesh)

    def add_mesh(self, mesh: MeshGeometry,
                 coarse_to_fine_cells=None,
                 fine_to_coarse_cells=None):
        """
        Adds a mesh into the hierarchy.

        Parameters
        ----------
        mesh
            The mesh to be added to the finest level.
        coarse_to_fine_cells
            Optional map from cells on the previous finest level to cells on
            ``mesh``.
        fine_to_coarse_cells
            Optional map from cells on ``mesh`` to cells on the previous
            finest level.
        """
        level = len(self.meshes)
        if level > 0 and (coarse_to_fine_cells is None or fine_to_coarse_cells is None):
            parent_cell_numbers = getattr(mesh, "_adaptive_parent_cell_numbers", None)
            if parent_cell_numbers is None:
                parent_cell_numbers = _netgen_parent_cell_numbers(self.meshes[-1], mesh)
                if parent_cell_numbers is not None:
                    mesh._adaptive_parent_cell_numbers = parent_cell_numbers
            if parent_cell_numbers is not None:
                transfer_mesh = _parent_owned_transfer_mesh(
                    self.meshes[-1], mesh, parent_cell_numbers
                )
                coarse_to_fine_cells, fine_to_coarse_cells = _adaptive_cell_maps(
                    self.meshes[-1], transfer_mesh, parent_cell_numbers
                )
                has_empty_rank = mesh.comm.allreduce(mesh.cell_set.size == 0,
                                                     op=MPI.LOR)
                if (self.redistribute and has_empty_rank
                        and transfer_mesh is not mesh):
                    mesh.redist = RedistMesh(transfer_mesh, mesh)
                else:
                    mesh = transfer_mesh

        self._meshes.append(mesh)
        self.meshes.append(mesh)
        set_level(mesh, self, level)
        redist = getattr(mesh, "redist", None)
        if redist is not None:
            set_level(redist.orig, self, level)

        if hasattr(mesh, "netgen_mesh"):
            mesh._adaptive_netgen_num_cells = len(_netgen_cells(mesh))

        if level > 0 and coarse_to_fine_cells is not None and fine_to_coarse_cells is not None:
            self.coarse_to_fine_cells[Fraction(level - 1, 1)] = coarse_to_fine_cells
            self.fine_to_coarse_cells[Fraction(level, 1)] = fine_to_coarse_cells

    def adapt(self, eta: Function | Cofunction, theta: float):
        """
        Adds a new mesh to the hierarchy by locally refining the finest mesh
        with a simplified variant of Dorfler marking. The finest mesh must
        come from a netgen mesh.

        Parameters
        ----------
        eta
            A DG0 :class:`~firedrake.function.Function` with the local error estimator.
        theta
            The threshold for marking as a fraction of the maximum error.

        Note
        ----
        Dorfler marking involves sorting all of the elements by decreasing
        error estimator and taking the minimal set that exceeds some fixed
        fraction of the total error. What this code implements is the simpler
        variant that doesn't have a proof of convergence (as far as I know)
        but works as well in practice.

        """
        if not isinstance(eta, (Function, Cofunction)):
            raise TypeError(f"eta must be a Function or Cofunction, not a {type(eta).__name__}")
        M = eta.function_space()
        if M.finat_element.space_dimension() != 1:
            raise ValueError("eta must be a Function or Cofunction in DG0")
        mesh = self.meshes[-1]
        if M.mesh() is not mesh:
            raise ValueError("eta must be defined on the finest mesh of the hierarchy")

        # Take the maximum over all processes
        with eta.dat.vec_ro as evec:
            _, eta_max = evec.max()

        threshold = theta * eta_max
        should_refine = eta.dat.data_ro > threshold

        markers = Function(M)
        markers.dat.data_wo[should_refine] = 1

        refined_mesh = mesh.refine_marked_elements(markers)
        self.add_mesh(refined_mesh)
        return self.meshes[-1]


def _netgen_cells(mesh):
    tdim = mesh.topological_dimension
    if tdim == 2:
        return mesh.netgen_mesh.Elements2D()
    elif tdim == 3:
        return mesh.netgen_mesh.Elements3D()
    raise NotImplementedError("Adaptive hierarchy maps are only implemented in dimension 2 and 3.")


def _netgen_cell_count(mesh):
    return getattr(mesh, "_adaptive_netgen_num_cells", len(_netgen_cells(mesh)))


def _distribute_cell_data(mesh, values):
    DG0 = FunctionSpace(mesh, "DG", 0)
    data = np.asarray(netgen_distribute(DG0, values), dtype=IntType)
    cstart, cend = mesh.topology_dm.getHeightStratum(0)
    cell_numbers = np.fromiter(
        (mesh._cell_numbering.getOffset(cell) for cell in range(cstart, cend)),
        dtype=IntType,
        count=cend - cstart,
    )
    ncell = min(data.shape[0], cell_numbers.size)
    result = np.full(mesh.cell_set.total_size, -1, dtype=IntType)
    valid = (0 <= cell_numbers[:ncell]) & (cell_numbers[:ncell] < result.size)
    result[cell_numbers[:ncell][valid]] = data[:ncell][valid]
    return result


def _adaptive_cell_maps(coarse_mesh, fine_mesh, parent_cell_numbers):
    if not (hasattr(coarse_mesh, "netgen_mesh") and hasattr(fine_mesh, "netgen_mesh")):
        return None, None

    coarse_cell_numbers = _distribute_cell_data(
        coarse_mesh, np.arange(_netgen_cell_count(coarse_mesh), dtype=IntType)
    )
    fine_parent_numbers = _distribute_cell_data(
        fine_mesh, np.asarray(parent_cell_numbers, dtype=IntType)
    )

    coarse_owned = coarse_mesh.cell_set.size
    fine_owned = fine_mesh.cell_set.size

    coarse_owned_lookup = {
        int(cell_number): cell
        for cell, cell_number in enumerate(coarse_cell_numbers[:coarse_owned])
    }

    fine_to_coarse = np.full((fine_owned, 1), -1, dtype=IntType)
    for fine_cell, parent_number in enumerate(fine_parent_numbers[:fine_owned]):
        fine_to_coarse[fine_cell, 0] = coarse_owned_lookup.get(int(parent_number), -1)

    children = [[] for _ in range(coarse_owned)]
    for fine_cell, parent_number in enumerate(fine_parent_numbers[:fine_owned]):
        coarse_cell = coarse_owned_lookup.get(int(parent_number))
        if coarse_cell is not None:
            children[coarse_cell].append(fine_cell)

    max_children = max((len(local_children) for local_children in children), default=0)
    coarse_to_fine = np.full((coarse_owned, max(1, max_children)), -1, dtype=IntType)
    for coarse_cell, local_children in enumerate(children):
        coarse_to_fine[coarse_cell, :len(local_children)] = local_children

    return coarse_to_fine, fine_to_coarse


def _netgen_parent_cell_numbers(coarse_mesh, fine_mesh):
    if not (hasattr(coarse_mesh, "netgen_mesh") and hasattr(fine_mesh, "netgen_mesh")):
        return None

    tdim = fine_mesh.topological_dimension
    try:
        parents = (fine_mesh.netgen_mesh.parentelements if tdim == 3
                   else fine_mesh.netgen_mesh.parentsurfaceelements)
    except AttributeError:
        return None

    parents = np.asarray(parents.NumPy()["i"], dtype=IntType)
    nfine = len(_netgen_cells(fine_mesh))
    ncoarse = _netgen_cell_count(coarse_mesh)
    if parents.shape[0] < nfine or ncoarse == 0:
        return None

    indices = np.arange(nfine, dtype=IntType)
    while True:
        refined = indices >= ncoarse
        if not refined.any():
            break
        parent_indices = parents[indices[refined]]
        if (parent_indices < 0).any():
            return None
        indices[refined] = parent_indices
    return indices


class RedistMesh:
    """Transfer data between a parent-owned mesh and its redistributed mesh."""

    def __init__(self, orig, redist):
        self.orig = orig
        self.redist = redist

    def _section_sf(self, root, leaf):
        point_sf = self.redist.sfBC_orig
        root_section = root.function_space().dm.getDefaultSection()
        leaf_section = leaf.function_space().dm.getDefaultSection()
        remote_offsets, _ = point_sf.distributeSection(root_section, leaf_section)
        return point_sf.createSectionSF(root_section, remote_offsets, leaf_section)

    def orig2redist(self, source, target):
        section_sf = self._section_sf(source, target)
        dtype, _ = get_mpi_type(source.dat)
        section_sf.bcastBegin(dtype,
                              source.dat.data_ro_with_halos,
                              target.dat.data_wo_with_halos,
                              MPI.REPLACE)
        section_sf.bcastEnd(dtype,
                            source.dat.data_ro_with_halos,
                            target.dat.data_wo_with_halos,
                            MPI.REPLACE)

    def redist2orig(self, source, target):
        section_sf = self._section_sf(target, source)
        dtype, _ = get_mpi_type(source.dat)
        section_sf.reduceBegin(dtype,
                               source.dat.data_ro_with_halos,
                               target.dat.data_wo_with_halos,
                               MPI.REPLACE)
        section_sf.reduceEnd(dtype,
                             source.dat.data_ro_with_halos,
                             target.dat.data_wo_with_halos,
                             MPI.REPLACE)


def _parent_owner_partition(coarse_mesh, parent_cell_numbers):
    comm = coarse_mesh.comm
    coarse_cell_numbers = _distribute_cell_data(
        coarse_mesh, np.arange(_netgen_cell_count(coarse_mesh), dtype=IntType)
    )
    owned_cell_numbers = coarse_cell_numbers[:coarse_mesh.cell_set.size]
    owned = np.empty((owned_cell_numbers.size, 2), dtype=IntType)
    owned[:, 0] = owned_cell_numbers
    owned[:, 1] = comm.rank
    gathered = comm.gather(owned, root=0)

    if comm.rank != 0:
        return None, None

    parent_owner = {}
    for cells in gathered:
        for cell_number, rank in cells:
            parent_owner[int(cell_number)] = int(rank)

    buckets = [[] for _ in range(comm.size)]
    for fine_cell, parent_number in enumerate(parent_cell_numbers):
        try:
            owner = parent_owner[int(parent_number)]
        except KeyError as exc:
            raise RuntimeError(
                "Cannot determine owner for adaptive parent cell "
                f"{int(parent_number)}"
            ) from exc
        buckets[owner].append(fine_cell)

    sizes = [len(bucket) for bucket in buckets]
    points = np.asarray(
        [point for bucket in buckets for point in bucket], dtype=IntType
    )
    return sizes, points


def _parent_owned_transfer_mesh(coarse_mesh, mesh, parent_cell_numbers):
    if mesh.comm.size == 1 or mesh.sfBC_orig is None:
        return mesh

    distribution_parameters = dict(mesh._distribution_parameters)
    distribution_parameters["partition"] = _parent_owner_partition(
        coarse_mesh, parent_cell_numbers
    )
    transfer_mesh = Mesh(
        mesh.netgen_mesh,
        reorder=mesh._did_reordering,
        distribution_parameters=distribution_parameters,
        comm=mesh.comm,
        netgen_flags=mesh.netgen_flags,
        tolerance=mesh.tolerance,
    )
    # The parent-owner partition is specific to this transfer mesh; do not
    # reuse it if this mesh is later adaptively refined.
    transfer_mesh._distribution_parameters = mesh._distribution_parameters
    transfer_mesh._adaptive_parent_cell_numbers = parent_cell_numbers
    return transfer_mesh

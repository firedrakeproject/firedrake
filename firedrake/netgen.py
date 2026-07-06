'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake

This file was copied from ngsPETSc.
'''
import numpy as np
from scipy.spatial.distance import cdist

from pyop2.mpi import COMM_WORLD, MPI
from firedrake.petsc import PETSc
from firedrake.cython.dmcommon import DistributedMeshOverlapType
from firedrake.mesh import DISTRIBUTION_PARAMETERS_NOOP, Mesh
from firedrake.utils import IntType
import firedrake

# Netgen and ngsPETSc are not available when the documentation is getting built
# because they do not have ARM wheels.
try:
    import netgen.meshing as ngm
    from netgen.meshing import MeshingParameters
    from ngsPETSc import MeshMapping
except ImportError:
    pass

try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type(None)


def netgen_distribute(V: firedrake.functionspaceimpl.WithGeometryBase,
                      netgen_data: np.ndarray):
    """
    Distribute data from the netgen layout into the DMPlex layout.

    Parameters
    ----------
    V
        The target function space defining the DMPlex layout.
    netgen_data
        The data in the layout of the underlying netgen mesh.

    Returns
    -------
    ``np.ndarray``
        The data in the target DMPlex layout.

    """
    netgen_data = np.asarray(netgen_data)
    mesh = V.mesh()
    sf = mesh.sfBC_orig
    if sf is None:
        # This mesh was not redistributed at construction.
        # This means that the underlying netgen mesh represents
        # the local part of the mesh owned by this process.
        # Therefore the netgen data is already distributed.
        plex_data = netgen_data
    else:
        plex = mesh.topology_dm
        nshape = netgen_data.shape
        dtype = netgen_data.dtype

        sfBCInv = sf.createInverse()
        section = V.dm.getDefaultSection()
        vec = V.dof_dset.layout_vec
        section0, vec0 = plex.distributeField(sfBCInv, section, vec)
        vec0.set(0)
        plex_data = None
        for i in np.ndindex(V.shape):
            di = netgen_data[(..., *i)].flatten()
            vec0[:len(di)] = di
            _, vec = plex.distributeField(sf, section0, vec0)
            arr = vec.getArray()
            if plex_data is None:
                plex_data = np.empty(arr.shape + V.shape, dtype=dtype)
            plex_data[(..., *i)] = arr
        plex_data = plex_data.reshape(-1, *nshape[1:])
    return plex_data


def adaptive_netgen_cells(mesh):
    tdim = mesh.topological_dimension
    if tdim == 2:
        return mesh.netgen_mesh.Elements2D()
    elif tdim == 3:
        return mesh.netgen_mesh.Elements3D()
    raise NotImplementedError("Adaptive refinement is only implemented in dimension 2 and 3.")


def _adaptive_netgen_cell_data(mesh, values):
    from firedrake.functionspace import FunctionSpace

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


def _adaptive_parent_owner_partition(coarse_mesh, parent_cell_numbers):
    comm = coarse_mesh.comm
    coarse_cell_numbers = _adaptive_netgen_cell_data(
        coarse_mesh,
        np.arange(len(adaptive_netgen_cells(coarse_mesh)), dtype=IntType),
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


def _set_adaptive_refined_mesh_metadata(mesh, netgen_mesh, netgen_flags,
                                        distribution_parameters,
                                        parent_cell_numbers):
    mesh.netgen_mesh = netgen_mesh
    mesh.netgen_flags = netgen_flags
    mesh._distribution_parameters = dict(distribution_parameters)
    mesh._adaptive_parent_cell_numbers = parent_cell_numbers
    mesh._adaptive_parent_owned = True


def make_adaptive_refined_mesh(coarse_mesh, netgen_mesh, netgen_flags,
                               parent_cell_numbers, redistribute=True,
                               balancing=0.15):
    distribution_parameters = dict(coarse_mesh._distribution_parameters)
    parent_parameters = dict(distribution_parameters)
    if coarse_mesh.comm.size > 1:
        parent_parameters["partition"] = _adaptive_parent_owner_partition(
            coarse_mesh, parent_cell_numbers
        )

    transfer_parameters = dict(parent_parameters)
    # Preserve transfer_parameters["partition"]: this is the shell
    # partition that keeps each child cell on its parent cell's rank.
    transfer_parameters["overlap_type"] = (
        DistributedMeshOverlapType.NONE, 0
    )
    transfer_mesh = Mesh(
        netgen_mesh,
        reorder=coarse_mesh._did_reordering,
        distribution_parameters=transfer_parameters,
        comm=coarse_mesh.comm,
        netgen_flags=netgen_flags,
        tolerance=coarse_mesh.tolerance,
    )
    _set_adaptive_refined_mesh_metadata(
        transfer_mesh, netgen_mesh, netgen_flags,
        distribution_parameters, parent_cell_numbers,
    )

    if coarse_mesh.comm.size == 1:
        return transfer_mesh

    num_cells = transfer_mesh.cell_set.size
    avg_cells = transfer_mesh.comm.allreduce(num_cells, op=MPI.SUM) / transfer_mesh.comm.size
    max_cells = transfer_mesh.comm.allreduce(num_cells, op=MPI.MAX)
    needs_redist = max_cells > (1 + balancing) * avg_cells

    if not (redistribute and needs_redist):
        overlap_type, overlap = distribution_parameters.get(
            "overlap_type", (DistributedMeshOverlapType.FACET, 1)
        )
        if overlap_type == DistributedMeshOverlapType.NONE and overlap == 0:
            return transfer_mesh
        refined_mesh = Mesh(
            netgen_mesh,
            reorder=coarse_mesh._did_reordering,
            distribution_parameters=parent_parameters,
            comm=coarse_mesh.comm,
            netgen_flags=netgen_flags,
            tolerance=coarse_mesh.tolerance,
        )
        _set_adaptive_refined_mesh_metadata(
            refined_mesh, netgen_mesh, netgen_flags,
            distribution_parameters, parent_cell_numbers,
        )
        return refined_mesh

    from firedrake.mg.utils import RedistributedMeshTransfer, redistribute_dm

    redist_dm = transfer_mesh.topology_dm.clone()
    redist_parameters = dict(distribution_parameters)
    redist_parameters["partition"] = True
    point_sf_orig, point_sf = redistribute_dm(redist_dm, redist_parameters)

    refined_mesh = Mesh(
        redist_dm,
        dim=coarse_mesh.geometric_dimension,
        reorder=coarse_mesh._did_reordering,
        distribution_parameters=DISTRIBUTION_PARAMETERS_NOOP,
        comm=coarse_mesh.comm,
        tolerance=coarse_mesh.tolerance,
    )
    _set_adaptive_refined_mesh_metadata(
        refined_mesh, netgen_mesh, netgen_flags,
        distribution_parameters, parent_cell_numbers,
    )
    if transfer_mesh.sfBC_orig is not None and point_sf_orig is not None:
        refined_mesh.sfBC_orig = transfer_mesh.sfBC_orig.compose(point_sf_orig)
        refined_mesh.sfBC = (refined_mesh.sfBC_orig if point_sf is point_sf_orig
                             else transfer_mesh.sfBC_orig.compose(point_sf))
    refined_mesh.redist = RedistributedMeshTransfer(transfer_mesh, refined_mesh, point_sf)
    return refined_mesh


def refine_marked_elements(mesh, mark, netgen_flags=None, redistribute=True,
                           balancing=0.15):
    if netgen_flags is None:
        netgen_flags = mesh.netgen_flags
    tdim = mesh.topological_dimension
    if tdim not in {2, 3}:
        raise NotImplementedError("No implementation for dimension other than 2 and 3.")
    with mark.dat.vec as mvec:
        if mesh.sfBC_orig is None:
            cstart, cend = mesh.topology_dm.getHeightStratum(0)
            cellNum = list(map(mesh._cell_numbering.getOffset, range(cstart, cend)))
            mark_np = mvec.getArray()[cellNum]
        else:
            sfBCInv = mesh.sfBC_orig.createInverse()
            _, mvec0 = mesh.topology_dm.distributeField(sfBCInv,
                                                        mesh._cell_numbering,
                                                        mvec)
            mark_np = mvec0.getArray()
    cells = (mesh.netgen_mesh.Elements3D() if tdim == 3
             else mesh.netgen_mesh.Elements2D())
    mark_np = mark_np[:len(cells)]
    max_refs = 0 if mark_np.size == 0 else int(mark_np.max())
    netgen_mesh = mesh.netgen_mesh.Copy()
    parent_cell_numbers = np.arange(mark_np.size, dtype=IntType)
    refine_faces = netgen_flags.get("refine_faces", False)
    for r in range(max_refs):
        cells = netgen_mesh.Elements3D() if tdim == 3 else netgen_mesh.Elements2D()
        cells.NumPy()["refine"] = (mark_np[:len(cells)] > 0)
        if tdim == 3:
            faces = netgen_mesh.Elements2D()
            faces.NumPy()["refine"] = refine_faces
        netgen_mesh.Refine(adaptive=True)
        mark_np -= 1
        parents = netgen_mesh.parentelements if tdim == 3 else netgen_mesh.parentsurfaceelements
        parents = parents.NumPy()["i"]
        num_fine_cells = parents.shape[0]
        num_coarse_cells = mark_np.size
        indices = np.arange(num_fine_cells, dtype=IntType)
        while (indices >= num_coarse_cells).any():
            fine_cells = (indices >= num_coarse_cells)
            indices[fine_cells] = parents[indices[fine_cells]]
        parent_cell_numbers = parent_cell_numbers[indices]
        if r < max_refs - 1:
            mark_np = mark_np[indices]

    return make_adaptive_refined_mesh(mesh, netgen_mesh, netgen_flags,
                                      parent_cell_numbers, redistribute,
                                      balancing)


def adaptive_cell_maps(coarse_mesh, fine_mesh, parent_cell_numbers):
    if not (hasattr(coarse_mesh, "netgen_mesh") and hasattr(fine_mesh, "netgen_mesh")):
        return None, None

    coarse_cell_numbers = _adaptive_netgen_cell_data(
        coarse_mesh, np.arange(len(adaptive_netgen_cells(coarse_mesh)), dtype=IntType)
    )
    fine_parent_numbers = _adaptive_netgen_cell_data(
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


def adaptive_parent_cell_numbers(coarse_mesh, fine_mesh):
    if not (hasattr(coarse_mesh, "netgen_mesh") and hasattr(fine_mesh, "netgen_mesh")):
        return None

    tdim = fine_mesh.topological_dimension
    try:
        parents = (fine_mesh.netgen_mesh.parentelements if tdim == 3
                   else fine_mesh.netgen_mesh.parentsurfaceelements)
    except AttributeError:
        return None

    parents = np.asarray(parents.NumPy()["i"], dtype=IntType)
    nfine = len(adaptive_netgen_cells(fine_mesh))
    ncoarse = len(adaptive_netgen_cells(coarse_mesh))
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


@PETSc.Log.EventDecorator()
def find_permutation(points_a: np.ndarray, points_b: np.ndarray,
                     tol: float = 1e-5):
    """ Find all permutations between a list of two sets of points.

    Given two numpy arrays of shape (ncells, npoints, dim) containing
    floating point coordinates for each cell, determine each index
    permutation that takes `points_a` to `points_b`. Ie:
    ```
    permutation = find_permutation(points_a, points_b)
    assert np.allclose(points_a[permutation], points_b, rtol=0, atol=tol)
    ```
    """
    if points_a.shape != points_b.shape:
        raise ValueError("`points_a` and `points_b` must have the same shape.")

    p = [np.where(cdist(a, b).T < tol)[1] for a, b in zip(points_a, points_b)]

    if len(p) == 0:
        return p

    try:
        permutation = np.array(p, ndmin=2)
    except ValueError as e:
        raise ValueError(
            "It was not possible to find a permutation for every cell"
            " within the provided tolerance"
        ) from e

    if permutation.shape != points_a.shape[0:2]:
        raise ValueError(
            "It was not possible to find a permutation for every cell"
            " within the provided tolerance"
        )

    return permutation


def splitToQuads(plex, dim, comm):
    """Split a Netgen mesh into quads using a PETSc transform."""
    # TODO: Improve support quad meshing.
    # @pef  Get netgen to make a quad-dominant mesh, and then only split the triangles.
    #       Current implementation will make for poor-quality meshes.
    if dim == 2:
        transform = PETSc.DMPlexTransform().create(comm=comm)
        transform.setType(PETSc.DMPlexTransformType.REFINETOBOX)
        transform.setDM(plex)
        transform.setUp()
    else:
        raise RuntimeError("Splitting to quads is only possible for 2D meshes.")
    newplex = transform.apply(plex)
    return newplex


splitTypes = {"Alfeld": lambda x: x.SplitAlfeld(),
              "Powell-Sabin": lambda x: x.SplitPowellSabin()}


class FiredrakeMesh:
    '''
    This class creates a Firedrake mesh from Netgen/NGSolve meshes.

    :arg mesh: the mesh object, it can be either a Netgen/NGSolve mesh or a PETSc DMPlex
    :param netgen_flags: The dictionary of flags to be passed to ngsPETSc.
    :arg comm: the MPI communicator.
    '''
    def __init__(self, mesh, netgen_flags, user_comm=COMM_WORLD):
        self.comm = user_comm
        # Parsing netgen flags
        if not isinstance(netgen_flags, dict):
            netgen_flags = {}
        split2tets = netgen_flags.get("split_to_tets", False)
        split = netgen_flags.get("split", False)
        quad = netgen_flags.get("quad", False)
        optMoves = netgen_flags.get("optimisation_moves", False)
        # Checking the mesh format
        if isinstance(mesh, (ngs.comp.Mesh, ngm.Mesh)):
            if split2tets:
                mesh = mesh.Split2Tets()
            if split:
                # Split mesh this includes Alfeld and Powell-Sabin
                splitTypes[split](mesh)
            if optMoves:
                # Optimises the mesh, for example smoothing
                if mesh.dim == 2:
                    mesh.OptimizeMesh2d(MeshingParameters(optimize2d=optMoves))
                elif mesh.dim == 3:
                    mesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=optMoves))
                else:
                    raise ValueError("Only 2D and 3D meshes can be optimised.")
            # We create the plex from the netgen mesh
            self.meshMap = MeshMapping(mesh, comm=self.comm)
            # We apply the DMPLEX transform
            if quad:
                newplex = splitToQuads(self.meshMap.petscPlex, mesh.dim, comm=self.comm)
                self.meshMap = MeshMapping(newplex)
        elif isinstance(mesh, PETSc.DMPlex):
            self.meshMap = MeshMapping(mesh)
        else:
            raise ValueError("Mesh format not recognised.")

'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake

This file was copied from ngsPETSc.
'''
import numpy as np
import numpy.typing as npt
from petsc4py import PETSc
from scipy.spatial.distance import cdist

import firedrake as fd

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


def netgen_distribute(V, netgen_data):
    netgen_data = np.asarray(netgen_data)
    mesh = V.mesh()
    sf = mesh.sfBC_orig
    if sf is not None:
        plex = mesh.topology_dm
        nshape = netgen_data.shape
        dtype = netgen_data.dtype

        section = V.dm.getDefaultSection()
        marked = V.dof_dset.layout_vec
        sfBCInv = sf.createInverse()
        section0, marked0 = plex.distributeField(sfBCInv, section, marked)
        plex_data = None

        marked0.set(0)
        d = netgen_data
        for i in np.ndindex(V.shape):
            di = d[(..., *i)].flatten()
            marked0[:len(di)] = di
            _, marked = plex.distributeField(sf, section0, marked0)
            arr = marked.getArray()
            if plex_data is None:
                plex_data = np.empty(arr.shape + V.shape, dtype=dtype)
            plex_data[(..., *i)] = arr
        plex_data = plex_data.reshape(-1, *nshape[1:])
    else:
        plex_data = netgen_data
    return plex_data


def plex_to_netgen_numbering(mesh):
    plex = mesh.topology_dm
    fstart, fend = plex.getHeightStratum(0)
    cellNum = list(map(mesh._cell_numbering.getOffset, range(fstart, fend)))
    return np.asarray(cellNum)


@PETSc.Log.EventDecorator()
def find_permutation(points_a: npt.NDArray[np.inexact], points_b: npt.NDArray[np.inexact],
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
    def __init__(self, mesh, netgen_flags=None, user_comm=fd.COMM_WORLD):
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

'''
This module contains all the functions related to wrapping NGSolve meshes to Firedrake
We adopt the same docstring conventions as the Firedrake project, since this part of
the package will only be used in combination with Firedrake.
'''
try:
    import firedrake as fd
    from firedrake.__future__ import interpolate
except ImportError:
    fd = None

import numpy as np
from petsc4py import PETSc

import netgen
import netgen.meshing as ngm
from netgen.meshing import MeshingParameters
try:
    import ngsolve as ngs
except ImportError:
    class ngs:
        "dummy class"
        class comp:
            "dummy class"
            Mesh = type(None)

from ngsPETSc import MeshMapping
from ngsPETSc.utils.utils import find_permutation

def flagsUtils(flags, option, default):
    '''
    utility fuction used to parse Netgen flag options
    '''
    try:
        return flags[option]
    except KeyError:
        return default

def refineMarkedElements(self, mark, netgen_flags={}):
    '''
    This method is used to refine a mesh based on a marking function
    which is a Firedrake DG0 function.

    :arg mark: the marking function which is a Firedrake DG0 function.
    :arg netgen_flags: the dictionary of flags to be passed to ngsPETSc.
    It includes the option:
        - refine_faces, which is a boolean specifyiong if you want to refine faces.

    '''
    DistParams = self._distribution_parameters
    els = {2: self.netgen_mesh.Elements2D, 3: self.netgen_mesh.Elements3D}
    dim = self.geometric_dimension()
    refine_faces = flagsUtils(netgen_flags, "refine_faces", False)
    if dim in [2,3]:
        with mark.dat.vec as marked:
            marked0 = marked
            getIdx = self._cell_numbering.getOffset
            if self.sfBCInv is not None:
                getIdx = lambda x: x #pylint: disable=C3001
                _, marked0 = self.topology_dm.distributeField(self.sfBCInv,
                                                              self._cell_numbering,
                                                              marked)
            if self.comm.Get_rank() == 0:
                mark = marked0.getArray()
                max_refs = np.max(mark)
                for _ in range(int(max_refs)):
                    for i, el in enumerate(els[dim]()):
                        if mark[getIdx(i)] > 0:
                            el.refine = True
                        else:
                            el.refine = False
                    if not refine_faces and dim == 3:
                        self.netgen_mesh.Elements2D().NumPy()["refine"] = 0
                    self.netgen_mesh.Refine(adaptive=True)
                    mark = mark-np.ones(mark.shape)
                return fd.Mesh(self.netgen_mesh, distribution_parameters=DistParams, comm=self.comm)
            return fd.Mesh(netgen.libngpy._meshing.Mesh(dim),
                           distribution_parameters=DistParams, comm=self.comm)
    else:
        raise NotImplementedError("No implementation for dimension other than 2 and 3.")



@PETSc.Log.EventDecorator()
def curveField(self, order, permutation_tol=1e-8, location_tol=1e-1, cg_field=False):
    '''
    This method returns a curved mesh as a Firedrake function.

    :arg order: the order of the curved mesh.
    :arg permutation_tol: tolerance used to construct the permutation of the reference element.
    :arg location_tol: tolerance used to locate the cell a point belongs to.
    :arg cg_field: return a CG function field representing the mesh, rather than the
                   default DG field.

    '''
    # Check if the mesh is a surface mesh or two dimensional mesh
    if len(self.netgen_mesh.Elements3D()) == 0:
        ng_element = self.netgen_mesh.Elements2D
    else:
        ng_element = self.netgen_mesh.Elements3D
    ng_dimension = len(ng_element())
    geom_dim = self.geometric_dimension()

    # Construct the mesh as a Firedrake function
    if cg_field:
        firedrake_space = fd.VectorFunctionSpace(self, "CG", order)
    else:
        low_order_element = self.coordinates.function_space().ufl_element().sub_elements[0]
        ufl_element = low_order_element.reconstruct(degree=order)
        firedrake_space = fd.VectorFunctionSpace(self, fd.BrokenElement(ufl_element))
    new_coordinates = fd.assemble(interpolate(self.coordinates, firedrake_space))

    # Compute reference points using fiat
    fiat_element = new_coordinates.function_space().finat_element.fiat_equivalent
    entity_ids = fiat_element.entity_dofs()
    nodes = fiat_element.dual_basis()
    ref = []
    for dim in entity_ids:
        for entity in entity_ids[dim]:
            for dof in entity_ids[dim][entity]:
                # Assert singleton point for each node.
                pt, = nodes[dof].get_point_dict().keys()
                ref.append(pt)
    reference_space_points = np.array(ref)

    # Curve the mesh on rank 0 only
    if self.comm.rank == 0:
        # Construct numpy arrays for physical domain data
        physical_space_points = np.zeros(
            (ng_dimension, reference_space_points.shape[0], geom_dim)
        )
        curved_space_points = np.zeros(
            (ng_dimension, reference_space_points.shape[0], geom_dim)
        )
        self.netgen_mesh.CalcElementMapping(reference_space_points, physical_space_points)
        self.netgen_mesh.Curve(order)
        self.netgen_mesh.CalcElementMapping(reference_space_points, curved_space_points)
        curved = ng_element().NumPy()["curved"]
        # Broadcast a boolean array identifying curved cells
        curved = self.comm.bcast(curved, root=0)
        physical_space_points = physical_space_points[curved]
        curved_space_points = curved_space_points[curved]
    else:
        curved = self.comm.bcast(None, root=0)
        # Construct numpy arrays as buffers to receive physical domain data
        ncurved = np.sum(curved)
        physical_space_points = np.zeros(
            (ncurved, reference_space_points.shape[0], geom_dim)
        )
        curved_space_points = np.zeros(
            (ncurved, reference_space_points.shape[0], geom_dim)
        )

    # Broadcast curved cell point data
    self.comm.Bcast(physical_space_points, root=0)
    self.comm.Bcast(curved_space_points, root=0)
    cell_node_map = new_coordinates.cell_node_map()

    # Select only the points in curved cells
    barycentres = np.average(physical_space_points, axis=1)
    ng_index = [*map(lambda x: self.locate_cell(x, tolerance=location_tol), barycentres)]

    # Select only the indices of points owned by this rank
    owned = [(0 <= ii < len(cell_node_map.values)) if ii is not None else False for ii in ng_index]

    # Select only the points owned by this rank
    physical_space_points = physical_space_points[owned]
    curved_space_points = curved_space_points[owned]
    barycentres = barycentres[owned]
    ng_index = [idx for idx, o in zip(ng_index, owned) if o]

    # Get the PyOP2 indices corresponding to the netgen indices
    pyop2_index = []
    for ngidx in ng_index:
        pyop2_index.extend(cell_node_map.values[ngidx])

    # Find the correct coordinate permutation for each cell
    # NB: Coordinates must be cast to real when running Firedrake in complex mode
    permutation = find_permutation(
        physical_space_points,
        new_coordinates.dat.data[pyop2_index].reshape(
            physical_space_points.shape
        ).astype(np.float64, copy=False),
        tol=permutation_tol
    )

    # Apply the permutation to each cell in turn
    for ii, p in enumerate(curved_space_points):
        curved_space_points[ii] = p[permutation[ii]]

    # Assign the curved coordinates to the dat
    new_coordinates.dat.data[pyop2_index] = curved_space_points.reshape(-1, geom_dim)

    return new_coordinates

def splitToQuads(plex, dim, comm):
    '''
    This method splits a Netgen mesh to quads, using a PETSc transform.
    TODO: Improve support quad meshing.
        @pef  Get netgen to make a quad-dominant mesh, and then only split the triangles.
              Current implementation will make for poor-quality meshes.
    '''
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
    def __init__(self, mesh, netgen_flags, user_comm=fd.COMM_WORLD):
        self.comm = user_comm
        #Parsing netgen flags
        if not isinstance(netgen_flags, dict):
            netgen_flags = {}
        split2tets = flagsUtils(netgen_flags, "split_to_tets", False)
        split = flagsUtils(netgen_flags, "split", False)
        quad = flagsUtils(netgen_flags, "quad", False)
        optMoves = flagsUtils(netgen_flags, "optimisation_moves", False)
        #Checking the mesh format
        if isinstance(mesh,(ngs.comp.Mesh,ngm.Mesh)):
            if split2tets:
                mesh = mesh.Split2Tets()
            if split:
                #Split mesh this includes Alfeld and Powell-Sabin
                splitTypes[split](mesh)
            if optMoves:
                #Optimises the mesh, for example smoothing
                if mesh.dim == 2:
                    mesh.OptimizeMesh2d(MeshingParameters(optimize2d=optMoves))
                elif mesh.dim == 3:
                    mesh.OptimizeVolumeMesh(MeshingParameters(optimize3d=optMoves))
                else:
                    raise ValueError("Only 2D and 3D meshes can be optimised.")
            #We create the plex from the netgen mesh
            self.meshMap = MeshMapping(mesh, comm=self.comm)
            #We apply the DMPLEX transform
            if quad:
                newplex = splitToQuads(self.meshMap.petscPlex, mesh.dim, comm=self.comm)
                self.meshMap = MeshMapping(newplex)
        elif isinstance(mesh, PETSc.DMPlex):
            self.meshMap = MeshMapping(mesh)
        else:
            raise ValueError("Mesh format not recognised.")

    def createFromTopology(self, topology, name, comm):
        '''
        Internal method to construct a mesh from a mesh topology, copied from Firedrake.

        :arg topology: the mesh topology

        :arg name: the mesh name

        '''
        cell = topology.ufl_cell()
        geometric_dim = topology.topology_dm.getCoordinateDim()
        element = fd.VectorElement("Lagrange", cell, 1, dim=geometric_dim)
        # Create mesh object
        self.firedrakeMesh = fd.MeshGeometry.__new__(fd.MeshGeometry, element, comm)
        self.firedrakeMesh._init_topology(topology)
        self.firedrakeMesh.name = name
        # Adding Netgen mesh and inverse sfBC as attributes
        self.firedrakeMesh.netgen_mesh = self.meshMap.ngMesh
        if self.firedrakeMesh.sfBC is not None:
            self.firedrakeMesh.sfBCInv = self.firedrakeMesh.sfBC.createInverse()
        else:
            self.firedrakeMesh.sfBCInv = None
        #Generating ngs to Firedrake cell index map
        #Adding refine_marked_elements and curve_field methods
        setattr(fd.MeshGeometry, "refine_marked_elements", refineMarkedElements)
        setattr(fd.MeshGeometry, "curve_field", curveField)

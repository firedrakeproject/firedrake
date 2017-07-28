import numpy as np
import ctypes
import os
import sys
import ufl
import weakref
from collections import defaultdict
from ufl.classes import ReferenceGrad

from pyop2.datatypes import IntType
from pyop2 import op2
from pyop2.mpi import COMM_WORLD, dup_comm, free_comm
from pyop2.profiling import timed_function, timed_region
from pyop2.utils import as_tuple

import firedrake.dmplex as dmplex
import firedrake.expression as expression
import firedrake.extrusion_utils as eutils
import firedrake.spatialindex as spatialindex
import firedrake.utils as utils
from firedrake.interpolation import interpolate
from firedrake.logging import info_red
from firedrake.parameters import parameters
from firedrake.petsc import PETSc


__all__ = ['Mesh', 'ExtrudedMesh', 'SubDomainData']


_cells = {
    1: {2: "interval"},
    2: {3: "triangle", 4: "quadrilateral"},
    3: {4: "tetrahedron"}
}


class _Facets(object):
    """Wrapper class for facet interation information on a :func:`Mesh`

    .. warning::

       The unique_markers argument **must** be the same on all processes."""
    def __init__(self, mesh, classes, kind, facet_cell, local_facet_number, markers=None,
                 unique_markers=None):

        self.mesh = mesh

        classes = as_tuple(classes, int, 4)
        self.classes = classes

        self.kind = kind
        assert(kind in ["interior", "exterior"])
        if kind == "interior":
            self._rank = 2
        else:
            self._rank = 1

        self.facet_cell = facet_cell

        self.local_facet_number = local_facet_number

        # assert that markers is a proper subset of unique_markers
        if markers is not None:
            for marker in markers:
                assert (marker in unique_markers), \
                    "Every marker has to be contained in unique_markers"

        self.markers = markers
        self.unique_markers = [] if unique_markers is None else unique_markers
        self._subsets = {}

    @utils.cached_property
    def set(self):
        size = self.classes
        if isinstance(self.mesh, ExtrudedMeshTopology):
            if self.kind == "interior":
                base = self.mesh._base_mesh.interior_facets.set
            else:
                base = self.mesh._base_mesh.exterior_facets.set
            return op2.ExtrudedSet(base, layers=self.mesh.layers)
        return op2.Set(size, "%sFacets" % self.kind.capitalize()[:3],
                       comm=self.mesh.comm)

    @utils.cached_property
    def _null_subset(self):
        '''Empty subset for the case in which there are no facets with
        a given marker value. This is required because not all
        markers need be represented on all processors.'''

        return op2.Subset(self.set, [])

    def measure_set(self, integral_type, subdomain_id,
                    all_integer_subdomain_ids=None):
        """Return an iteration set appropriate for the requested integral type.

        :arg integral_type: The type of the integral (should be a facet measure).
        :arg subdomain_id: The subdomain of the mesh to iterate over.
             Either an integer, an iterable of integers or the special
             subdomains ``"everywhere"`` or ``"otherwise"``.
        :arg all_integer_subdomain_ids: Information to interpret the
             ``"otherwise"`` subdomain.  ``"otherwise"`` means all
             entities not explicitly enumerated by the integer
             subdomains provided here.  For example, if
             all_integer_subdomain_ids is empty, then ``"otherwise" ==
             "everywhere"``.  If it contains ``(1, 2)``, then
             ``"otherwise"`` is all entities except those marked by
             subdomains 1 and 2.

         :returns: A :class:`pyop2.Subset` for iteration.
        """
        if integral_type in ("exterior_facet_bottom",
                             "exterior_facet_top",
                             "interior_facet_horiz"):
            # these iterate over the base cell set
            return self.mesh.cell_subset(subdomain_id, all_integer_subdomain_ids)
        elif not (integral_type.startswith("exterior_") or
                  integral_type.startswith("interior_")):
            raise ValueError("Don't know how to construct measure for '%s'" % integral_type)
        if subdomain_id == "everywhere":
            return self.set
        if subdomain_id == "otherwise":
            if all_integer_subdomain_ids is None:
                return self.set
            key = ("otherwise", ) + all_integer_subdomain_ids
            try:
                return self._subsets[key]
            except KeyError:
                ids = [np.where(self.markers == sid)[0]
                       for sid in all_integer_subdomain_ids]
                to_remove = np.unique(np.concatenate(ids))
                indices = np.arange(self.set.total_size, dtype=np.int32)
                indices = np.delete(indices, to_remove)
                return self._subsets.setdefault(key, op2.Subset(self.set, indices))
        else:
            return self.subset(subdomain_id)

    def subset(self, markers):
        """Return the subset corresponding to a given marker value.

        :param markers: integer marker id or an iterable of marker ids
            (or ``None``, for an empty subset).
        """
        if self.markers is None:
            return self._null_subset
        markers = as_tuple(markers, int)
        try:
            return self._subsets[markers]
        except KeyError:
            # check that the given markers are valid
            if len(set(markers).difference(self.unique_markers)) > 0:
                invalid = set(markers).difference(self.unique_markers)
                raise LookupError("{0} are not a valid markers (not in {1})".format(invalid, self.unique_markers))

            # build a list of indices corresponding to the subsets selected by
            # markers
            indices = np.concatenate([np.nonzero(self.markers == i)[0]
                                      for i in markers])
            return self._subsets.setdefault(markers, op2.Subset(self.set, indices))

    @utils.cached_property
    def local_facet_dat(self):
        """Dat indicating which local facet of each adjacent
        cell corresponds to the current facet."""

        return op2.Dat(op2.DataSet(self.set, self._rank), self.local_facet_number,
                       np.uintc, "%s_%s_local_facet_number" % (self.mesh.name, self.kind))

    @utils.cached_property
    def facet_cell_map(self):
        """Map from facets to cells."""
        return op2.Map(self.set, self.mesh.cell_set, self._rank, self.facet_cell,
                       "facet_to_cell_map")


def _from_gmsh(filename, comm=None):
    """Read a Gmsh .msh file from `filename`.

    :kwarg comm: Optional communicator to build the mesh on (defaults to
        COMM_WORLD).
    """
    comm = comm or COMM_WORLD
    # Create a read-only PETSc.Viewer
    gmsh_viewer = PETSc.Viewer().create(comm=comm)
    gmsh_viewer.setType("ascii")
    gmsh_viewer.setFileMode("r")
    gmsh_viewer.setFileName(filename)
    gmsh_plex = PETSc.DMPlex().createGmsh(gmsh_viewer, comm=comm)

    if gmsh_plex.hasLabel("Face Sets"):
        boundary_ids = gmsh_plex.getLabelIdIS("Face Sets").getIndices()
        gmsh_plex.createLabel("boundary_ids")
        for bid in boundary_ids:
            faces = gmsh_plex.getStratumIS("Face Sets", bid).getIndices()
            for f in faces:
                gmsh_plex.setLabelValue("boundary_ids", f, bid)

    return gmsh_plex


def _from_exodus(filename, comm):
    """Read an Exodus .e or .exo file from `filename`.

    :arg comm: communicator to build the mesh on.
    """
    plex = PETSc.DMPlex().createExodusFromFile(filename, comm=comm)

    if plex.hasLabel("Face Sets"):
        boundary_ids = plex.getLabelIdIS("Face Sets").getIndices()
        plex.createLabel("boundary_ids")
        for bid in boundary_ids:
            faces = plex.getStratumIS("Face Sets", bid).getIndices()
            for f in faces:
                plex.setLabelValue("boundary_ids", f, bid)

    return plex


def _from_cgns(filename, comm):
    """Read a CGNS .cgns file from `filename`.

    :arg comm: communicator to build the mesh on.
    """
    plex = PETSc.DMPlex().createCGNSFromFile(filename, comm=comm)

    # TODO: Add boundary IDs
    return plex


def _from_triangle(filename, dim, comm):
    """Read a set of triangle mesh files from `filename`.

    :arg dim: The embedding dimension.
    :arg comm: communicator to build the mesh on.
    """
    basename, ext = os.path.splitext(filename)

    comm = dup_comm(comm)
    if comm.rank == 0:
        try:
            facetfile = open(basename+".face")
            tdim = 3
        except:
            try:
                facetfile = open(basename+".edge")
                tdim = 2
            except:
                facetfile = None
                tdim = 1
        if dim is None:
            dim = tdim
        comm.bcast(tdim, root=0)

        with open(basename+".node") as nodefile:
            header = np.fromfile(nodefile, dtype=np.int32, count=2, sep=' ')
            nodecount = header[0]
            nodedim = header[1]
            assert nodedim == dim
            coordinates = np.loadtxt(nodefile, usecols=list(range(1, dim+1)), skiprows=1, dtype=np.double)
            assert nodecount == coordinates.shape[0]

        with open(basename+".ele") as elefile:
            header = np.fromfile(elefile, dtype=np.int32, count=2, sep=' ')
            elecount = header[0]
            eledim = header[1]
            eles = np.loadtxt(elefile, usecols=list(range(1, eledim+1)), dtype=np.int32, skiprows=1)
            assert elecount == eles.shape[0]

        cells = list(map(lambda c: c-1, eles))
    else:
        tdim = comm.bcast(None, root=0)
        cells = None
        coordinates = None
    plex = _from_cell_list(tdim, cells, coordinates, comm=comm)

    # Apply boundary IDs
    if comm.rank == 0:
        facets = None
        try:
            header = np.fromfile(facetfile, dtype=np.int32, count=2, sep=' ')
            edgecount = header[0]
            facets = np.loadtxt(facetfile, usecols=list(range(1, tdim+2)), dtype=np.int32, skiprows=0)
            assert edgecount == facets.shape[0]
        finally:
            facetfile.close()

        if facets is not None:
            vStart, vEnd = plex.getDepthStratum(0)   # vertices
            for facet in facets:
                bid = facet[-1]
                vertices = list(map(lambda v: v + vStart - 1, facet[:-1]))
                join = plex.getJoin(vertices)
                plex.setLabelValue("boundary_ids", join[0], bid)

    free_comm(comm)
    return plex


def _from_cell_list(dim, cells, coords, comm):
    """
    Create a DMPlex from a list of cells and coords.

    :arg dim: The topological dimension of the mesh
    :arg cells: The vertices of each cell
    :arg coords: The coordinates of each vertex
    :arg comm: communicator to build the mesh on.
    """
    comm = dup_comm(comm)
    # These types are /correct/, DMPlexCreateFromCellList wants int
    # and double (not PetscInt, PetscReal).
    if comm.rank == 0:
        cells = np.asarray(cells, dtype=np.int32)
        coords = np.asarray(coords, dtype=np.double)
        comm.bcast(cells.shape, root=0)
        comm.bcast(coords.shape, root=0)
        # Provide the actual data on rank 0.
        plex = PETSc.DMPlex().createFromCellList(dim, cells, coords, comm=comm)
    else:
        cell_shape = list(comm.bcast(None, root=0))
        coord_shape = list(comm.bcast(None, root=0))
        cell_shape[0] = 0
        coord_shape[0] = 0
        # Provide empty plex on other ranks
        # A subsequent call to plex.distribute() takes care of parallel partitioning
        plex = PETSc.DMPlex().createFromCellList(dim,
                                                 np.zeros(cell_shape, dtype=np.int32),
                                                 np.zeros(coord_shape, dtype=np.double),
                                                 comm=comm)
    free_comm(comm)
    return plex


class MeshTopology(object):
    """A representation of mesh topology."""

    @timed_function("CreateMesh")
    def __init__(self, plex, name, reorder, distribute):
        """Half-initialise a mesh topology.

        :arg plex: :class:`DMPlex` representing the mesh topology
        :arg name: name of the mesh
        :arg reorder: whether to reorder the mesh (bool)
        :arg distribute: whether to distribute the mesh to parallel processes
        """
        # Do some validation of the input mesh
        dmplex.validate_mesh(plex)
        utils._init()

        self._plex = plex
        self.name = name
        self.comm = dup_comm(plex.comm.tompi4py())

        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        # Cell subsets for integration over subregions
        self._subsets = {}
        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        label_boundary = (self.comm.size == 1) or distribute
        dmplex.label_facets(plex, label_boundary=label_boundary)

        # Distribute the dm to all ranks
        if self.comm.size > 1 and distribute:
            # We distribute with overlap zero, in case we're going to
            # refine this mesh in parallel.  Later, when we actually use
            # it, we grow the halo.
            partitioner = plex.getPartitioner()
            if IntType.itemsize == 8:
                # Default to Parmetis on 64bit ints (Chaco is 32 bit int only)
                partitioner.setType(partitioner.Type.PARMETIS)
            partitioner.setFromOptions()
            plex.distribute(overlap=0)

        dim = plex.getDimension()

        cStart, cEnd = plex.getHeightStratum(0)  # cells
        if cStart == cEnd:
            raise RuntimeError("Mesh must have at least one cell on every process")
        cell_nfacets = plex.getConeSize(cStart)

        self._grown_halos = False
        self._ufl_cell = ufl.Cell(_cells[dim][cell_nfacets])

        def callback(self):
            """Finish initialisation."""
            del self._callback
            if self.comm.size > 1:
                self._plex.distributeOverlap(1)
            self._grown_halos = True

            if reorder:
                with timed_region("Mesh: reorder"):
                    old_to_new = self._plex.getOrdering(PETSc.Mat.OrderingType.RCM).indices
                    reordering = np.empty_like(old_to_new)
                    reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
            else:
                # No reordering
                reordering = None
            self._did_reordering = bool(reorder)

            # Mark OP2 entities and derive the resulting Plex renumbering
            with timed_region("Mesh: numbering"):
                dmplex.mark_entity_classes(self._plex)
                self._entity_classes = dmplex.get_entity_classes(self._plex).astype(int)
                self._plex_renumbering = dmplex.plex_renumbering(self._plex,
                                                                 self._entity_classes,
                                                                 reordering)

                # Derive a cell numbering from the Plex renumbering
                entity_dofs = np.zeros(dim+1, dtype=IntType)
                entity_dofs[-1] = 1

                self._cell_numbering = self._plex.createSection([1], entity_dofs,
                                                                perm=self._plex_renumbering)
                entity_dofs[:] = 0
                entity_dofs[0] = 1
                self._vertex_numbering = self._plex.createSection([1], entity_dofs,
                                                                  perm=self._plex_renumbering)

                entity_dofs[:] = 0
                entity_dofs[-2] = 1
                facet_numbering = self._plex.createSection([1], entity_dofs,
                                                           perm=self._plex_renumbering)
                self._facet_ordering = dmplex.get_facet_ordering(self._plex, facet_numbering)
        self._callback = callback

    def mpi_comm(self):
        """The MPI communicator this mesh is built on (an mpi4py object)."""
        return self.comm

    @timed_function("CreateMesh")
    def init(self):
        """Finish the initialisation of the mesh."""
        if hasattr(self, '_callback'):
            self._callback(self)

    @property
    def topology(self):
        """The underlying mesh topology object."""
        return self

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self

    @property
    def layers(self):
        return None

    def ufl_cell(self):
        """The UFL :class:`~ufl.classes.Cell` associated with the mesh."""
        return self._ufl_cell

    @utils.cached_property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        plex = self._plex
        dim = plex.getDimension()

        # Cell numbering and global vertex numbering
        cell_numbering = self._cell_numbering
        vertex_numbering = self._vertex_numbering.createGlobalSection(plex.getPointSF())

        cell = self.ufl_cell()
        if cell.is_simplex():
            # Simplex mesh
            cStart, cEnd = plex.getHeightStratum(0)
            a_closure = plex.getTransitiveClosure(cStart)[0]

            entity_per_cell = np.zeros(dim + 1, dtype=IntType)
            for dim in range(dim + 1):
                start, end = plex.getDepthStratum(dim)
                entity_per_cell[dim] = sum(map(lambda idx: start <= idx < end,
                                               a_closure))

            return dmplex.closure_ordering(plex, vertex_numbering,
                                           cell_numbering, entity_per_cell)

        elif cell.cellname() == "quadrilateral":
            from firedrake.citations import Citations
            Citations().register("Homolya2016")
            Citations().register("McRae2016")
            # Quadrilateral mesh
            cell_ranks = dmplex.get_cell_remote_ranks(plex)

            facet_orientations = dmplex.quadrilateral_facet_orientations(
                plex, vertex_numbering, cell_ranks)

            cell_orientations = dmplex.orientations_facet2cell(
                plex, vertex_numbering, cell_ranks,
                facet_orientations, cell_numbering)

            dmplex.exchange_cell_orientations(plex,
                                              cell_numbering,
                                              cell_orientations)

            return dmplex.quadrilateral_closure_ordering(
                plex, vertex_numbering, cell_numbering, cell_orientations)

        else:
            raise NotImplementedError("Cell type '%s' not supported." % cell)

    @utils.cached_property
    def exterior_facets(self):
        # If there are no exterior facets on this process, everything
        # just falls through nicely, so no need to special case.

        # Compute the facet_numbering
        # Order exterior facets by OP2 entity class
        exterior_facets, exterior_facet_classes = \
            dmplex.get_facets_by_class(self._plex, b"exterior_facets",
                                       self._facet_ordering)

        # Derive attached boundary IDs
        if self._plex.hasLabel("boundary_ids"):
            boundary_ids = np.zeros(exterior_facets.size, dtype=IntType)
            for i, facet in enumerate(exterior_facets):
                boundary_ids[i] = self._plex.getLabelValue("boundary_ids", facet)

            # Determine union of boundary IDs.  All processes must
            # know the values of all ids, for collective reasons.
            comm = self.comm
            from mpi4py import MPI

            def merge_ids(x, y, datatype):
                return x.union(y)

            op = MPI.Op.Create(merge_ids, commute=True)

            local_ids = set(self._plex.getLabelIdIS("boundary_ids").indices)
            unique_ids = np.asarray(sorted(comm.allreduce(local_ids, op=op)),
                                    dtype=IntType)
            op.Free()
        else:
            boundary_ids = None
            unique_ids = None

        exterior_local_facet_number, exterior_facet_cell = \
            dmplex.facet_numbering(self._plex, "exterior",
                                   exterior_facets,
                                   self._cell_numbering,
                                   self.cell_closure)

        return _Facets(self, exterior_facet_classes, "exterior",
                       exterior_facet_cell, exterior_local_facet_number,
                       boundary_ids, unique_markers=unique_ids)

    @utils.cached_property
    def interior_facets(self):
        # If there are no interior facets on this process, everything
        # just falls through nicely, so no need to special case.

        # Compute the facet_numbering
        # Order interior facets by OP2 entity class
        interior_facets, interior_facet_classes = \
            dmplex.get_facets_by_class(self._plex, b"interior_facets",
                                       self._facet_ordering)

        label = "Face Sets"
        if self._plex.hasLabel(label):
            boundary_ids = np.zeros(interior_facets.size, dtype=np.int32)
            for i, facet in enumerate(interior_facets):
                boundary_ids[i] = self._plex.getLabelValue(label, facet)

            # Determine union of boundary IDs.  All processes must
            # know the values of all ids, for collective reasons.
            comm = self.comm
            from mpi4py import MPI

            def merge_ids(x, y, datatype):
                return x.union(y)

            op = MPI.Op.Create(merge_ids, commute=True)

            local_ids = set(self._plex.getLabelIdIS(label).indices)
            unique_ids = np.asarray(sorted(comm.allreduce(local_ids, op=op)),
                                    dtype=np.int32)
            op.Free()
            unique_ids = np.unique(boundary_ids)
        else:
            boundary_ids = None
            unique_ids = None
        interior_local_facet_number, interior_facet_cell = \
            dmplex.facet_numbering(self._plex, "interior",
                                   interior_facets,
                                   self._cell_numbering,
                                   self.cell_closure)

        return _Facets(self, interior_facet_classes, "interior",
                       interior_facet_cell, interior_local_facet_number, boundary_ids, unique_markers=unique_ids)

    @utils.cached_property
    def cell_to_facets(self):
        """Return a :class:`op2.Dat` that maps from a cell index to the local
        facet types on each cell.

        The i-th local facet is exterior if the value of this array is :data:`0`
        and interior if the value is :data:`1`.
        """
        cell_facets = dmplex.cell_facet_labeling(self._plex,
                                                 self._cell_numbering,
                                                 self.cell_closure)
        nfacet = cell_facets.shape[1]
        return op2.Dat(self.cell_set**nfacet, cell_facets, dtype=cell_facets.dtype,
                       name="cell-to-local-facet-dat")

    def make_cell_node_list(self, global_numbering, entity_dofs):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg entity_dofs: FInAT element entity DoFs
        """
        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     entity_dofs)

    def make_dofs_per_plex_entity(self, entity_dofs):
        """Returns the number of DoFs per plex entity for each stratum,
        i.e. [#dofs / plex vertices, #dofs / plex edges, ...].

        :arg entity_dofs: FInAT element entity DoFs
        """
        return [len(entity_dofs[d][0]) for d in sorted(entity_dofs)]

    def make_offset(self, entity_dofs, ndofs):
        """Returns None (only for extruded use)."""
        return None

    def _order_data_by_cell_index(self, column_list, cell_data):
        return cell_data[column_list]

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :func:`init_cell_orientations` on the mesh *geometry* to initialise."""
        if not hasattr(self, '_cell_orientations'):
            raise RuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

    def num_cells(self):
        cStart, cEnd = self._plex.getHeightStratum(0)
        return cEnd - cStart

    def num_facets(self):
        fStart, fEnd = self._plex.getHeightStratum(1)
        return fEnd - fStart

    def num_faces(self):
        fStart, fEnd = self._plex.getDepthStratum(2)
        return fEnd - fStart

    def num_edges(self):
        eStart, eEnd = self._plex.getDepthStratum(1)
        return eEnd - eStart

    def num_vertices(self):
        vStart, vEnd = self._plex.getDepthStratum(0)
        return vEnd - vStart

    def num_entities(self, d):
        eStart, eEnd = self._plex.getDepthStratum(d)
        return eEnd - eStart

    def size(self, d):
        return self.num_entities(d)

    def cell_dimension(self):
        """Returns the cell dimension."""
        return self.ufl_cell().topological_dimension()

    def facet_dimension(self):
        """Returns the facet dimension."""
        # Facets have co-dimension 1
        return self.ufl_cell().topological_dimension() - 1

    @utils.cached_property
    def cell_set(self):
        size = list(self._entity_classes[self.cell_dimension(), :])
        return op2.Set(size, "Cells", comm=self.comm)

    def cell_subset(self, subdomain_id, all_integer_subdomain_ids=None):
        """Return a subset over cells with the given subdomain_id.

        :arg subdomain_id: The subdomain of the mesh to iterate over.
             Either an integer, an iterable of integers or the special
             subdomains ``"everywhere"`` or ``"otherwise"``.
        :arg all_integer_subdomain_ids: Information to interpret the
             ``"otherwise"`` subdomain.  ``"otherwise"`` means all
             entities not explicitly enumerated by the integer
             subdomains provided here.  For example, if
             all_integer_subdomain_ids is empty, then ``"otherwise" ==
             "everywhere"``.  If it contains ``(1, 2)``, then
             ``"otherwise"`` is all entities except those marked by
             subdomains 1 and 2.

         :returns: A :class:`pyop2.Subset` for iteration.
        """
        if subdomain_id == "everywhere":
            return self.cell_set
        if subdomain_id == "otherwise":
            if all_integer_subdomain_ids is None:
                return self.cell_set
            key = ("otherwise", ) + all_integer_subdomain_ids
        else:
            key = subdomain_id
        try:
            return self._subsets[key]
        except KeyError:
            if subdomain_id == "otherwise":
                ids = tuple(dmplex.get_cell_markers(self._plex,
                                                    self._cell_numbering,
                                                    sid)
                            for sid in all_integer_subdomain_ids)
                to_remove = np.unique(np.concatenate(ids))
                indices = np.arange(self.cell_set.total_size, dtype=IntType)
                indices = np.delete(indices, to_remove)
            else:
                indices = dmplex.get_cell_markers(self._plex,
                                                  self._cell_numbering,
                                                  subdomain_id)
            return self._subsets.setdefault(key, op2.Subset(self.cell_set, indices))

    def measure_set(self, integral_type, subdomain_id,
                    all_integer_subdomain_ids=None):
        """Return an iteration set appropriate for the requested integral type.

        :arg integral_type: The type of the integral (should be a valid UFL measure).
        :arg subdomain_id: The subdomain of the mesh to iterate over.
             Either an integer, an iterable of integers or the special
             subdomains ``"everywhere"`` or ``"otherwise"``.
        :arg all_integer_subdomain_ids: Information to interpret the
             ``"otherwise"`` subdomain.  ``"otherwise"`` means all
             entities not explicitly enumerated by the integer
             subdomains provided here.  For example, if
             all_integer_subdomain_ids is empty, then ``"otherwise" ==
             "everywhere"``.  If it contains ``(1, 2)``, then
             ``"otherwise"`` is all entities except those marked by
             subdomains 1 and 2.  This should be a dict mapping
             ``integral_type`` to the explicitly enumerated subdomain ids.

         :returns: A :class:`pyop2.Subset` for iteration.
        """
        if all_integer_subdomain_ids is not None:
            all_integer_subdomain_ids = all_integer_subdomain_ids.get(integral_type, None)
        if integral_type == "cell":
            return self.cell_subset(subdomain_id, all_integer_subdomain_ids)
        elif integral_type in ("exterior_facet", "exterior_facet_vert",
                               "exterior_facet_top", "exterior_facet_bottom"):
            return self.exterior_facets.measure_set(integral_type, subdomain_id,
                                                    all_integer_subdomain_ids)
        elif integral_type in ("interior_facet", "interior_facet_vert",
                               "interior_facet_horiz"):
            return self.interior_facets.measure_set(integral_type, subdomain_id,
                                                    all_integer_subdomain_ids)
        else:
            raise ValueError("Unknown integral type '%s'" % integral_type)


class ExtrudedMeshTopology(MeshTopology):
    """Representation of an extruded mesh topology."""

    def __init__(self, mesh, layers):
        """Build an extruded mesh topology from an input mesh topology

        :arg mesh:           the unstructured base mesh topology
        :arg layers:         number of extruded cell layers in the "vertical"
                             direction.
        """
        from firedrake.citations import Citations
        Citations().register("McRae2016")
        Citations().register("Bercea2016")
        # A cache of shared function space data on this mesh
        self._shared_data_cache = defaultdict(dict)

        mesh.init()
        self._base_mesh = mesh
        if layers < 1:
            raise RuntimeError("Must have at least one layer of extruded cells (not %d)" % layers)
        self.comm = mesh.comm
        # All internal logic works with layers of base mesh (not layers of cells)
        self._layers = layers + 1
        self._ufl_cell = ufl.TensorProductCell(mesh.ufl_cell(), ufl.interval)

        # TODO: These attributes are copied so that FunctionSpaceBase can
        # access them directly.  Eventually we would want a better refactoring
        # of responsibilities between mesh and function space.
        self._plex = mesh._plex
        self._plex_renumbering = mesh._plex_renumbering
        self._cell_numbering = mesh._cell_numbering
        self._entity_classes = mesh._entity_classes
        self._subsets = {}

    @property
    def name(self):
        return self._base_mesh.name

    @property
    def cell_closure(self):
        """2D array of ordered cell closures

        Each row contains ordered cell entities for a cell, one row per cell.
        """
        return self._base_mesh.cell_closure

    @utils.cached_property
    def exterior_facets(self):
        exterior_facets = self._base_mesh.exterior_facets
        return _Facets(self, exterior_facets.classes,
                       "exterior",
                       exterior_facets.facet_cell,
                       exterior_facets.local_facet_number,
                       exterior_facets.markers,
                       unique_markers=exterior_facets.unique_markers)

    @utils.cached_property
    def interior_facets(self):
        interior_facets = self._base_mesh.interior_facets
        return _Facets(self, interior_facets.classes,
                       "interior",
                       interior_facets.facet_cell,
                       interior_facets.local_facet_number)

    def make_cell_node_list(self, global_numbering, entity_dofs):
        """Builds the DoF mapping.

        :arg global_numbering: Section describing the global DoF numbering
        :arg entity_dofs: FInAT element entity DoFs
        """
        flat_entity_dofs = {}
        for b, v in entity_dofs:
            # v in [0, 1].  Only look at the ones, then grab the data from zeros.
            if v == 0:
                continue
            flat_entity_dofs[b] = {}
            for i in entity_dofs[(b, v)]:
                # This line is fairly magic.
                # It works because an interval has two points.
                # We pick up the DoFs from the bottom point,
                # then the DoFs from the interior of the interval,
                # then finally the DoFs from the top point.
                flat_entity_dofs[b][i] = \
                    entity_dofs[(b, 0)][2*i] + entity_dofs[(b, 1)][i] + entity_dofs[(b, 0)][2*i+1]

        return dmplex.get_cell_nodes(global_numbering,
                                     self.cell_closure,
                                     flat_entity_dofs)

    def make_dofs_per_plex_entity(self, entity_dofs):
        """Returns the number of DoFs per plex entity for each stratum,
        i.e. [#dofs / plex vertices, #dofs / plex edges, ...].

        :arg entity_dofs: FInAT element entity DoFs
        """
        dofs_per_entity = [0] * (1 + self._base_mesh.cell_dimension())
        for (b, v), entities in entity_dofs.items():
            dofs_per_entity[b] += (self.layers - v) * len(entities[0])
        return dofs_per_entity

    def make_offset(self, entity_dofs, ndofs):
        """Returns the offset between the neighbouring cells of a
        column for each DoF.

        :arg entity_dofs: FInAT element entity DoFs
        :arg ndofs: number of DoFs in the FInAT element
        """
        entity_offset = [0] * (1 + self._base_mesh.cell_dimension())
        for (b, v), entities in entity_dofs.items():
            entity_offset[b] += len(entities[0])

        dof_offset = np.zeros(ndofs, dtype=IntType)
        for (b, v), entities in entity_dofs.items():
            for dof_indices in entities.values():
                for i in dof_indices:
                    dof_offset[i] = entity_offset[b]
        return dof_offset

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    def cell_dimension(self):
        """Returns the cell dimension."""
        return (self._base_mesh.cell_dimension(), 1)

    def facet_dimension(self):
        """Returns the facet dimension.

        .. note::

            This only returns the dimension of the "side" (vertical) facets,
            not the "top" or "bottom" (horizontal) facets.

        """
        return (self._base_mesh.facet_dimension(), 1)

    @utils.cached_property
    def cell_set(self):
        return op2.ExtrudedSet(self._base_mesh.cell_set, layers=self.layers)

    def _order_data_by_cell_index(self, column_list, cell_data):
        cell_list = []
        for col in column_list:
            cell_list += list(range(col, col + (self.layers - 1)))
        return cell_data[cell_list]


class MeshGeometry(ufl.Mesh):
    """A representation of mesh topology and geometry."""

    def __new__(cls, element):
        """Create mesh geometry object."""
        utils._init()
        mesh = super(MeshGeometry, cls).__new__(cls)
        mesh.uid = utils._new_uid()
        assert isinstance(element, ufl.FiniteElementBase)
        ufl.Mesh.__init__(mesh, element, ufl_id=mesh.uid)
        return mesh

    def __init__(self, coordinates):
        """Initialise a mesh geometry from coordinates.

        :arg coordinates: a coordinateless function containing the coordinates
        """
        # Direct link to topology
        self._topology = coordinates.function_space().mesh()

        # Cache mesh object on the coordinateless coordinates function
        coordinates._as_mesh_geometry = weakref.ref(self)

        self._coordinates = coordinates

    def init(self):
        """Finish the initialisation of the mesh.  Most of the time
        this is carried out automatically, however, in some cases (for
        example accessing a property of the mesh directly after
        constructing it) you need to call this manually."""
        if hasattr(self, '_callback'):
            self._callback(self)

    @property
    def topology(self):
        """The underlying mesh topology object."""
        return self._topology

    @property
    def topological(self):
        """Alias of topology.

        This is to ensure consistent naming for some multigrid codes."""
        return self._topology

    @utils.cached_property
    def _coordinates_function(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        import firedrake.functionspaceimpl as functionspaceimpl
        import firedrake.function as function
        self.init()

        coordinates_fs = self._coordinates.function_space()
        V = functionspaceimpl.WithGeometry(coordinates_fs, self)
        f = function.Function(V, val=self._coordinates)
        return f

    @property
    def coordinates(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        return self._coordinates_function

    @coordinates.setter
    def coordinates(self, value):
        message = """Cannot re-assign the coordinates.

You are free to change the coordinate values, but if you need a
different coordinate function space, use Mesh(f) to create a new mesh
object whose coordinates are f's values.  (This will not copy the
values from f.)"""

        raise AttributeError(message)

    def clear_spatial_index(self):
        """Reset the :attr:`spatial_index` on this mesh geometry.

        Use this if you move the mesh (for example by reassigning to
        the coordinate field)."""
        try:
            del self.spatial_index
        except AttributeError:
            pass

    @utils.cached_property
    def spatial_index(self):
        """Spatial index to quickly find which cell contains a given point."""

        from firedrake import function, functionspace
        from firedrake.parloops import par_loop, READ, RW

        gdim = self.ufl_cell().geometric_dimension()
        if gdim <= 1:
            info_red("libspatialindex does not support 1-dimension, falling back on brute force.")
            return None

        # Calculate the bounding boxes for all cells by running a kernel
        V = functionspace.VectorFunctionSpace(self, "DG", 0, dim=gdim)
        coords_min = function.Function(V)
        coords_max = function.Function(V)

        coords_min.dat.data.fill(np.inf)
        coords_max.dat.data.fill(-np.inf)

        kernel = """
    for (int d = 0; d < gdim; d++) {
        for (int i = 0; i < nodes_per_cell; i++) {
            f_min[0][d] = fmin(f_min[0][d], f[i][d]);
            f_max[0][d] = fmax(f_max[0][d], f[i][d]);
        }
    }
"""

        cell_node_list = self.coordinates.function_space().cell_node_list
        nodes_per_cell = len(cell_node_list[0])

        kernel = kernel.replace("gdim", str(gdim))
        kernel = kernel.replace("nodes_per_cell", str(nodes_per_cell))

        par_loop(kernel, ufl.dx, {'f': (self.coordinates, READ),
                                  'f_min': (coords_min, RW),
                                  'f_max': (coords_max, RW)})

        # Reorder bounding boxes according to the cell indices we use
        column_list = V.cell_node_list.reshape(-1)
        coords_min = self._order_data_by_cell_index(column_list, coords_min.dat.data_ro_with_halos)
        coords_max = self._order_data_by_cell_index(column_list, coords_max.dat.data_ro_with_halos)

        # Build spatial index
        return spatialindex.from_regions(coords_min, coords_max)

    def locate_cell(self, x, tolerance=None):
        """Locate cell containg given point.

        :arg x: point coordinates
        :kwarg tolerance: for checking if a point is in a cell.
        :returns: cell number (int), or None (if the point is not in the domain)
        """
        x = np.asarray(x, dtype=np.float)
        cell = self._c_locator(tolerance=tolerance)(self.coordinates._ctypes,
                                                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if cell == -1:
            return None
        else:
            return cell

    def _c_locator(self, tolerance=None):
        from pyop2 import compilation
        from pyop2.utils import get_petsc_dir
        import firedrake.function as function
        import firedrake.pointquery_utils as pq_utils

        cache = self.__dict__.setdefault("_c_locator_cache", {})
        try:
            return cache[tolerance]
        except KeyError:
            src = pq_utils.src_locate_cell(self, tolerance=tolerance)
            src += """
    int locator(struct Function *f, double *x)
    {
        struct ReferenceCoords reference_coords;
        return locate_cell(f, x, %(geometric_dimension)d, &to_reference_coords, &reference_coords);
    }
    """ % dict(geometric_dimension=self.geometric_dimension())

            locator = compilation.load(src, "c", "locator",
                                       cppargs=["-I%s" % os.path.dirname(__file__),
                                                "-I%s/include" % sys.prefix] +
                                       ["-I%s/include" % d for d in get_petsc_dir()],
                                       ldargs=["-L%s/lib" % sys.prefix,
                                               "-lspatialindex_c",
                                               "-Wl,-rpath,%s/lib" % sys.prefix])

            locator.argtypes = [ctypes.POINTER(function._CFunction),
                                ctypes.POINTER(ctypes.c_double)]
            locator.restype = ctypes.c_int
            return cache.setdefault(tolerance, locator)

    def init_cell_orientations(self, expr):
        """Compute and initialise :attr:`cell_orientations` relative to a specified orientation.

        :arg expr: an :class:`.Expression` evaluated to produce a
             reference normal direction.

        """
        import firedrake.function as function
        import firedrake.functionspace as functionspace

        if self.ufl_cell() not in (ufl.Cell('triangle', 3),
                                   ufl.Cell("quadrilateral", 3),
                                   ufl.TensorProductCell(ufl.Cell('interval'), ufl.Cell('interval'),
                                                         geometric_dimension=3)):
            raise NotImplementedError('Only implemented for triangles and quadrilaterals embedded in 3d')

        if hasattr(self.topology, '_cell_orientations'):
            raise RuntimeError("init_cell_orientations already called, did you mean to do so again?")

        if isinstance(expr, expression.Expression):
            if expr.value_shape()[0] != 3:
                raise NotImplementedError('Only implemented for 3-vectors')

            expr = interpolate(expr, functionspace.VectorFunctionSpace(self, 'DG', 0))
        elif isinstance(expr, ufl.classes.Expr):
            if expr.ufl_shape != (3,):
                raise NotImplementedError('Only implemented for 3-vectors')
        else:
            raise TypeError("UFL expression or Expression object expected!")

        fs = functionspace.FunctionSpace(self, 'DG', 0)
        x = ufl.SpatialCoordinate(self)
        f = function.Function(fs)
        f.interpolate(ufl.dot(expr, ufl.cross(ReferenceGrad(x)[:, 0], ReferenceGrad(x)[:, 1])))

        cell_orientations = function.Function(fs, name="cell_orientations", dtype=np.int32)
        cell_orientations.dat.data[:] = (f.dat.data_ro < 0)
        self.topology._cell_orientations = cell_orientations

    def __getattr__(self, name):
        return getattr(self._topology, name)


def make_mesh_from_coordinates(coordinates):
    """Given a coordinate field build a new mesh, using said coordinate field.

    :arg coordinates: A :class:`~.Function`.
    """
    if hasattr(coordinates, '_as_mesh_geometry'):
        mesh = coordinates._as_mesh_geometry()
        if mesh is not None:
            return mesh

    V = coordinates.function_space()
    element = coordinates.ufl_element()
    if V.rank != 1 or len(element.value_shape()) != 1:
        raise ValueError("Coordinates must be from a rank-1 FunctionSpace with rank-1 value_shape.")
    assert V.mesh().ufl_cell().topological_dimension() <= V.value_size
    # Build coordinate element
    element = coordinates.ufl_element()
    cell = element.cell().reconstruct(geometric_dimension=V.value_size)
    element = element.reconstruct(cell=cell)

    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh.__init__(coordinates)
    return mesh


@timed_function("CreateMesh")
def Mesh(meshfile, **kwargs):
    """Construct a mesh object.

    Meshes may either be created by reading from a mesh file, or by
    providing a PETSc DMPlex object defining the mesh topology.

    :param meshfile: Mesh file name (or DMPlex object) defining
           mesh topology.  See below for details on supported mesh
           formats.
    :param dim: optional specification of the geometric dimension
           of the mesh (ignored if not reading from mesh file).
           If not supplied the geometric dimension is deduced from
           the topological dimension of entities in the mesh.
    :param reorder: optional flag indicating whether to reorder
           meshes for better cache locality.  If not supplied the
           default value in ``parameters["reorder_meshes"]``
           is used.
    :param comm: the communicator to use when creating the mesh.  If
           not supplied, then the mesh will be created on COMM_WORLD.
           Ignored if ``meshfile`` is a DMPlex object (in which case
           the communicator will be taken from there).

    When the mesh is read from a file the following mesh formats
    are supported (determined, case insensitively, from the
    filename extension):

    * GMSH: with extension `.msh`
    * Exodus: with extension `.e`, `.exo`
    * CGNS: with extension `.cgns`
    * Triangle: with extension `.node`

    .. note::

        When the mesh is created directly from a DMPlex object,
        the ``dim`` parameter is ignored (the DMPlex already
        knows its geometric and topological dimensions).

    """
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    if isinstance(meshfile, function.Function):
        coordinates = meshfile.topological
    elif isinstance(meshfile, function.CoordinatelessFunction):
        coordinates = meshfile
    else:
        coordinates = None

    if coordinates is not None:
        return make_mesh_from_coordinates(coordinates)

    utils._init()

    geometric_dim = kwargs.get("dim", None)
    reorder = kwargs.get("reorder", None)
    if reorder is None:
        reorder = parameters["reorder_meshes"]
    distribute = kwargs.get("distribute", True)

    if isinstance(meshfile, PETSc.DMPlex):
        name = "plexmesh"
        plex = meshfile
    else:
        comm = kwargs.get("comm", COMM_WORLD)
        name = meshfile
        basename, ext = os.path.splitext(meshfile)

        if ext.lower() in ['.e', '.exo']:
            plex = _from_exodus(meshfile, comm)
        elif ext.lower() == '.cgns':
            plex = _from_cgns(meshfile, comm)
        elif ext.lower() == '.msh':
            plex = _from_gmsh(meshfile, comm)
        elif ext.lower() == '.node':
            plex = _from_triangle(meshfile, geometric_dim, comm)
        else:
            raise RuntimeError("Mesh file %s has unknown format '%s'."
                               % (meshfile, ext[1:]))

    # Create mesh topology
    topology = MeshTopology(plex, name=name, reorder=reorder, distribute=distribute)

    tcell = topology.ufl_cell()
    if geometric_dim is None:
        geometric_dim = tcell.topological_dimension()
    cell = tcell.reconstruct(geometric_dimension=geometric_dim)

    element = ufl.VectorElement("Lagrange", cell, 1)
    # Create mesh object
    mesh = MeshGeometry.__new__(MeshGeometry, element)
    mesh._topology = topology

    def callback(self):
        """Finish initialisation."""
        del self._callback
        # Finish the initialisation of mesh topology
        self.topology.init()

        coordinates_fs = functionspace.VectorFunctionSpace(self.topology, "Lagrange", 1,
                                                           dim=geometric_dim)

        coordinates_data = dmplex.reordered_coords(plex, coordinates_fs.dm.getDefaultSection(),
                                                   (self.num_vertices(), geometric_dim))

        coordinates = function.CoordinatelessFunction(coordinates_fs,
                                                      val=coordinates_data,
                                                      name="Coordinates")

        self.__init__(coordinates)

    mesh._callback = callback
    return mesh


@timed_function("CreateExtMesh")
def ExtrudedMesh(mesh, layers, layer_height=None, extrusion_type='uniform', kernel=None, gdim=None):
    """Build an extruded mesh from an input mesh

    :arg mesh:           the unstructured base mesh
    :arg layers:         number of extruded cell layers in the "vertical"
                         direction.
    :arg layer_height:   the layer height, assuming all layers are evenly
                         spaced. If this is omitted, the value defaults to
                         1/layers (i.e. the extruded mesh has total height 1.0)
                         unless a custom kernel is used.
    :arg extrusion_type: the algorithm to employ to calculate the extruded
                         coordinates. One of "uniform", "radial",
                         "radial_hedgehog" or "custom". See below.
    :arg kernel:         a :class:`pyop2.Kernel` to produce coordinates for
                         the extruded mesh. See :func:`~.make_extruded_coords`
                         for more details.
    :arg gdim:           number of spatial dimensions of the
                         resulting mesh (this is only used if a
                         custom kernel is provided)

    The various values of ``extrusion_type`` have the following meanings:

    ``"uniform"``
        the extruded mesh has an extra spatial
        dimension compared to the base mesh. The layers exist
        in this dimension only.

    ``"radial"``
        the extruded mesh has the same number of
        spatial dimensions as the base mesh; the cells are
        radially extruded outwards from the origin. This
        requires the base mesh to have topological dimension
        strictly smaller than geometric dimension.
    ``"radial_hedgehog"``
        similar to `radial`, but the cells
        are extruded in the direction of the outward-pointing
        cell normal (this produces a P1dgxP1 coordinate field).
        In this case, a radially extruded coordinate field
        (generated with ``extrusion_type="radial"``) is
        available in the :attr:`radial_coordinates` attribute.
    ``"custom"``
        use a custom kernel to generate the extruded coordinates

    For more details see the :doc:`manual section on extruded meshes <extruded-meshes>`.
    """
    import firedrake.functionspace as functionspace
    import firedrake.function as function

    mesh.init()
    topology = ExtrudedMeshTopology(mesh.topology, layers)

    if extrusion_type == "uniform":
        pass
    elif extrusion_type in ("radial", "radial_hedgehog"):
        # do not allow radial extrusion if tdim = gdim
        if mesh.ufl_cell().geometric_dimension() == mesh.ufl_cell().topological_dimension():
            raise RuntimeError("Cannot radially-extrude a mesh with equal geometric and topological dimension")
    else:
        # check for kernel
        if kernel is None:
            raise RuntimeError("If the custom extrusion_type is used, a kernel must be provided")
        # otherwise, use the gdim that was passed in
        if gdim is None:
            raise RuntimeError("The geometric dimension of the mesh must be specified if a custom extrusion kernel is used")

    # Compute Coordinates of the extruded mesh
    if layer_height is None:
        # Default to unit
        layer_height = 1 / layers

    if extrusion_type == 'radial_hedgehog':
        hfamily = "DG"
    else:
        hfamily = mesh._coordinates.ufl_element().family()
    hdegree = mesh._coordinates.ufl_element().degree()

    if gdim is None:
        gdim = mesh.ufl_cell().geometric_dimension() + (extrusion_type == "uniform")
    coordinates_fs = functionspace.VectorFunctionSpace(topology, hfamily, hdegree, dim=gdim,
                                                       vfamily="Lagrange", vdegree=1)

    coordinates = function.CoordinatelessFunction(coordinates_fs, name="Coordinates")

    eutils.make_extruded_coords(topology, mesh._coordinates, coordinates,
                                layer_height, extrusion_type=extrusion_type, kernel=kernel)

    self = make_mesh_from_coordinates(coordinates)
    self._base_mesh = mesh

    if extrusion_type == "radial_hedgehog":
        fs = functionspace.VectorFunctionSpace(self, "CG", hdegree, dim=gdim,
                                               vfamily="CG", vdegree=1)
        self.radial_coordinates = function.Function(fs)
        eutils.make_extruded_coords(topology, mesh._coordinates, self.radial_coordinates,
                                    layer_height, extrusion_type="radial", kernel=kernel)

    return self


def SubDomainData(geometric_expr):
    """Creates a subdomain data object from a boolean-valued UFL expression.

    The result can be attached as the subdomain_data field of a
    :class:`ufl.Measure`. For example:

        x = mesh.coordinates
        sd = SubDomainData(x[0] < 0.5)
        assemble(f*dx(subdomain_data=sd))

    """
    import firedrake.functionspace as functionspace
    import firedrake.projection as projection

    # Find domain from expression
    m = geometric_expr.ufl_domain()

    # Find selected cells
    fs = functionspace.FunctionSpace(m, 'DG', 0)
    f = projection.project(ufl.conditional(geometric_expr, 1, 0), fs)

    # Create cell subset
    indices, = np.nonzero(f.dat.data_ro_with_halos > 0.5)
    return op2.Subset(m.cell_set, indices)

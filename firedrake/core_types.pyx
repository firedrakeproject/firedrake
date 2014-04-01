# cython: embedsignature=True
# Re-implementation of state_types for firedrake.
import os
from mpi4py import MPI
import numpy as np
import pyop2.ir.ast_base as ast

from petsc import PETSc

cimport numpy as np
cimport cpython as py

import ufl
import FIAT

from pyop2 import op2
from pyop2.caching import Cached
from pyop2.utils import as_tuple

import utils
import dmplex
from dmplex import _from_cell_list
from collections import defaultdict

np.import_array()


__all__ = ['Mesh', 'ExtrudedMesh']


_cells = {
    1 : { 2 : "interval"},
    2 : { 3 : "triangle"},
    3 : { 3: "triangle", 4 : "tetrahedron"}
    }

_FIAT_cells = {
    "interval" : FIAT.reference_element.UFCInterval,
    "triangle" : FIAT.reference_element.UFCTriangle,
    "tetrahedron" : FIAT.reference_element.UFCTetrahedron
    }

# Start the uid from 1000 in a vain attempt to steer clear of any current uses.
_current_uid = 1000

valuetype = np.float64

def _new_uid():
    global _current_uid

    _current_uid += 1
    return _current_uid

def _init():
    """Cause :func:`pyop2.init` to be called in case the user has not done it
    for themselves. The result of this is that the user need only call
    :func:`pyop2.init` if she wants to set a non-default option, for example
    to switch the backend or the debug or log level."""
    if not op2.initialised():
        op2.init(log_level='INFO')
    # Work around circular dependency issue
    global types
    import types

def fiat_from_ufl_element(ufl_element):
    if isinstance(ufl_element, ufl.EnrichedElement):
        return FIAT.EnrichedElement(fiat_from_ufl_element(ufl_element._elements[0]), fiat_from_ufl_element(ufl_element._elements[1]))
    elif isinstance(ufl_element, ufl.HDiv):
        return FIAT.Hdiv(fiat_from_ufl_element(ufl_element._element))
    elif isinstance(ufl_element, ufl.HCurl):
        return FIAT.Hcurl(fiat_from_ufl_element(ufl_element._element))
    elif isinstance(ufl_element, (ufl.OuterProductElement, ufl.OuterProductVectorElement)):
        return FIAT.TensorFiniteElement(fiat_from_ufl_element(ufl_element._A), fiat_from_ufl_element(ufl_element._B))
    else:
        return FIAT.supported_elements[ufl_element.family()]\
            (_FIAT_cells[ufl_element.cell().cellname()](), ufl_element.degree())

# Functions related to the extruded case
def extract_offset(offset, facet_map, base_map):
    """Starting from existing mappings for base and facets extract
    the sub-offset corresponding to the facet map."""
    try:
        res = np.zeros(len(facet_map), np.int32)
    except TypeError:
        res = np.zeros(1, np.int32)
        facet_map = [facet_map]
    for i, facet_dof in enumerate(facet_map):
        for j, base_dof in enumerate(base_map):
            if base_dof == facet_dof:
                res[i] = offset[j]
                break
    return res

def compute_extruded_dofs(fiat_element, flat_dofs, layers):
    """Compute the number of dofs in a column"""
    size = len(flat_dofs)
    dofs_per_column = np.zeros(size, np.int32)
    for i in range(size):
        for j in range(2): #2 is due to the process of extrusion
            dofs_per_column[i] += (layers - j) * len(fiat_element.entity_dofs()[(i,j)][0])
    return dofs_per_column

def compute_vertical_offsets(ent_dofs, flat_dofs):
    """Compute the offset between corresponding dofs in layers.

    offsets[i] is the offset from the bottom of the stack to the
    corresponding dof in the ith layer.
    """
    size = len(flat_dofs)
    offsets_per_vertical = np.zeros(size, np.int32)
    for i in range(size):
        if len(flat_dofs[i][0]) > 0:
            offsets_per_vertical[i] = len(flat_dofs[i][0]) - len(ent_dofs[(i,0)][0])
    return offsets_per_vertical

def compute_offset(ent_dofs, flat_dofs, total_dofs):
    """Compute extruded offsets for flattened element.

    offsets[i] is the number of dofs in the vertical for the ith
    column of flattened mesh entities."""
    size = len(flat_dofs)
    res = np.zeros(total_dofs, np.int32)
    vert_dofs = compute_vertical_offsets(ent_dofs, flat_dofs)
    for i in range(size):
        elems = len(flat_dofs[i])
        dofs_per_elem = len(flat_dofs[i][0])
        for j in range(elems):
            for k in range(dofs_per_elem):
                res[flat_dofs[i][j][k]] = vert_dofs[i]
    return res

def total_num_dofs(flat_dofs):
    """Compute the total number of degrees of freedom in the extruded mesh"""
    size = len(flat_dofs)
    total = 0
    for i in range(size):
        total += len(flat_dofs[i]) * len(flat_dofs[i][0])
    return total

def make_flat_fiat_element(ufl_cell_element, ufl_cell, flattened_entity_dofs):
    """Create a modified FIAT-style element.
    Transform object from 3D-Extruded to 2D-flattened FIAT-style object."""
    # Create base element
    base_element = fiat_from_ufl_element(ufl_cell_element)

    # Alter base element
    base_element.dual.entity_ids = flattened_entity_dofs
    base_element.poly_set.num_members = total_num_dofs(flattened_entity_dofs)

    return base_element

def make_extruded_coords(extruded_mesh, layer_height,
                         extrusion_type='uniform', kernel=None):
    """
    Given either a kernel or a (fixed) layer_height, compute an
    extruded coordinate field for an extruded mesh.

    :arg extruded_mesh: an :class:`ExtrudedMesh` to extrude a
         coordinate field for.
    :arg layer_height: an equi-spaced height for each layer.
    :arg extrusion_type: the type of extrusion to use.  Predefined
         options are either "uniform" (creating equi-spaced layers by
         extruding in the (n+1)dth direction) or "radial" (creating
         equi-spaced layers by extruding in the outward direction from
         the origin).
    :arg kernel: an optional kernel to carry out coordinate extrusion.

    The kernel signature (if provided) is::

        void kernel(double **base_coords, double **ext_coords,
                    int **layer, double *layer_height)

    The kernel iterates over the cells of the mesh and receives as
    arguments the coordinates of the base cell (to read), the
    coordinates on the extruded cell (to write to), the layer number
    of each cell and the fixed layer height.
    """
    base_coords = extruded_mesh._old_mesh.coordinates
    ext_coords = extruded_mesh.coordinates
    vert_space = ext_coords.function_space().ufl_element()._B
    if kernel is None and not (vert_space.degree() == 1 and vert_space.family() == 'Lagrange'):
        raise RuntimeError('Extrusion of coordinates is only possible for P1 interval unless a custom kernel is provided')
    if kernel is not None:
        pass
    elif extrusion_type == 'uniform':
        kernel = op2.Kernel("""
        void uniform_extrusion_kernel(double **base_coords,
                    double **ext_coords,
                    int **layer,
                    double *layer_height) {
            for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                    ext_coords[2*d][c] = base_coords[d][c];
                    ext_coords[2*d+1][c] = base_coords[d][c];
                }
                ext_coords[2*d][%(base_coord_dim)d] = *layer_height * (layer[0][0]);
                ext_coords[2*d+1][%(base_coord_dim)d] = *layer_height * (layer[0][0] + 1);
            }
        }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                'base_coord_dim': base_coords.function_space().cdim},
                            "uniform_extrusion_kernel")
    elif extrusion_type == 'radial':
        kernel = op2.Kernel("""
        void radial_extrusion_kernel(double **base_coords,
                   double **ext_coords,
                   int **layer,
                   double *layer_height) {
            for ( int d = 0; d < %(base_map_arity)d; d++ ) {
                double norm = 0.0;
                for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                    norm += base_coords[d][c] * base_coords[d][c];
                }
                norm = sqrt(norm);
                for ( int c = 0; c < %(base_coord_dim)d; c++ ) {
                    ext_coords[2*d][c] = base_coords[d][c] * (1 + (*layer_height * layer[0][0])/norm);
                    ext_coords[2*d+1][c] = base_coords[d][c] * (1 + (*layer_height * (layer[0][0]+1))/norm);
                }
            }
        }""" % {'base_map_arity': base_coords.cell_node_map().arity,
                'base_coord_dim': base_coords.function_space().cdim},
                            "radial_extrusion_kernel")
    else:
        raise NotImplementedError('Unsupported extrusion type "%s"' % extrusion_type)

    # Dat to hold layer number
    layer_fs = types.FunctionSpace(extruded_mesh, 'DG', 0)
    layers = extruded_mesh.layers
    layer = op2.Dat(layer_fs.dof_dset,
                    np.repeat(np.arange(layers-1, dtype=np.int32),
                              extruded_mesh.cell_set.total_size).reshape(layers-1, extruded_mesh.cell_set.total_size).T.ravel(), dtype=np.int32)
    height = op2.Global(1, layer_height, dtype=float)
    op2.par_loop(kernel,
                 ext_coords.cell_set,
                 base_coords.dat(op2.READ, base_coords.cell_node_map()),
                 ext_coords.dat(op2.WRITE, ext_coords.cell_node_map()),
                 layer(op2.READ, layer_fs.cell_node_map()),
                 height(op2.READ))


class _Facets(object):
    """Wrapper class for facet interation information on a :class:`Mesh`"""
    def __init__(self, mesh, count, kind, facet_cell, local_facet_number, markers=None):

        self.mesh = mesh

        self.count = count

        self.kind = kind
        assert(kind in ["interior", "exterior"])
        if kind == "interior":
            self._rank = 2
        else:
            self._rank = 1

        self.facet_cell = facet_cell

        self.local_facet_number = local_facet_number

        self.markers = markers
        self._subsets = {}

    @utils.cached_property
    def set(self):
        # Currently no MPI parallel support
        size = self.count
        halo = None
        if isinstance(self.mesh, ExtrudedMesh):
            if self.kind == "interior":
                base = self.mesh._old_mesh.interior_facets.set
            else:
                base = self.mesh._old_mesh.exterior_facets.set
            return op2.ExtrudedSet(base, layers=self.mesh.layers)
        return op2.Set(size, "%s_%s_facets" % (self.mesh.name, self.kind), halo=halo)


    @property
    def bottom_set(self):
        '''Returns the bottom row of cells.'''
        return self.mesh.cell_set

    @utils.cached_property
    def _null_subset(self):
        '''Empty subset for the case in which there are no facets with
        a given marker value. This is required because not all
        markers need be represented on all processors.'''

        return op2.Subset(self.set,[])

    def measure_set(self, measure):
        '''Return the iteration set appropriate to measure. This will
        either be for all the interior or exterior (as appropriate)
        facets, or for a particular numbered subdomain.'''

        dom_id = measure.domain_id()
        dom_type = measure.domain_type()
        if dom_id in [measure.DOMAIN_ID_EVERYWHERE,
                      measure.DOMAIN_ID_OTHERWISE]:
            if dom_type == "exterior_facet_topbottom":
                return [(op2.ON_BOTTOM, self.bottom_set),
                        (op2.ON_TOP, self.bottom_set)]
            elif dom_type == "exterior_facet_bottom":
                return [(op2.ON_BOTTOM, self.bottom_set)]
            elif dom_type == "exterior_facet_top":
                return [(op2.ON_TOP, self.bottom_set)]
            elif dom_type == "interior_facet_horiz":
                return self.bottom_set
            else:
                return self.set
        else:
            return self.subset(measure.domain_id())

    def subset(self, markers):
        """Return the subset corresponding to a given marker value.

        :param markers: integer marker id or an iterable of marker ids"""
        if self.markers is None:
            return self._null_subset
        markers = as_tuple(markers, int)
        try:
            return self._subsets[markers]
        except KeyError:
            indices = np.concatenate([np.nonzero(self.markers==i)[0]
                                      for i in markers])
            self._subsets[markers] = op2.Subset(self.set, indices)
            return self._subsets[markers]

    @utils.cached_property
    def local_facet_dat(self):
        """Dat indicating which local facet of each adjacent
        cell corresponds to the current facet."""

        return op2.Dat(op2.DataSet(self.set, self._rank), self.local_facet_number,
                       np.uintc, "%s_%s_local_facet_number" % (self.mesh.name, self.kind))

class Mesh(object):
    """A representation of mesh topology and geometry."""
    def __init__(self, filename, dim=None, periodic_coords=None, plex=None):
        """
        :param filename: the mesh file to read.  Supported mesh formats
               are Gmsh (extension ``msh``) and triangle (extension
               ``node``).
        :param dim: optional dimension of the coordinates in the
               supplied mesh.  If not supplied, the coordinate
               dimension is determined from the type of topological
               entities in the mesh file.  In particular, you will
               need to supply a value for ``dim`` if the mesh is an
               immersed manifold (where the geometric and topological
               dimensions of entities are not the same).
        :param periodic_coords: optional numpy array of coordinates
               used to replace those read from the mesh file.  These
               are only supported in 1D and must have enough entries
               to be used as a DG1 field on the mesh.
        """

        _init()

        self.parent = None

        if dim is None:
            # Mesh reading in Fluidity level considers 0 to be None.
            dim = 0

        if plex is not None:
            self._from_dmplex(plex, geometric_dim=dim,
                              periodic_coords=periodic_coords)
        else:
            basename, ext = os.path.splitext(filename)

            if ext in ['.e', '.exo', '.E', '.EXO']:
                self._from_exodus(filename, dim)
            elif ext in ['.cgns', '.CGNS']:
                self._from_cgns(filename, dim)
            elif ext in ['.msh']:
                self._from_gmsh(filename, dim, periodic_coords)
            elif ext in ['.node']:
                self._from_triangle(filename, dim, periodic_coords)
            else:
                raise RuntimeError("Unknown mesh file format.")

        self._cell_orientations = op2.Dat(self.cell_set, dtype=np.int32,
                                          name="cell_orientations")
        # -1 is uninitialised.
        self._cell_orientations.data[:] = -1

    @property
    def coordinates(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        return self._coordinate_function

    @coordinates.setter
    def coordinates(self, value):
        self._coordinate_function = value

    def _from_dmplex(self, plex, geometric_dim=0, periodic_coords=None):
        """ Create mesh from DMPlex object """

        self._plex = plex
        self.uid = _new_uid()

        if geometric_dim == 0:
            geometric_dim = self._plex.getDimension()

        # Mark exterior and interior facets
        self._plex.markBoundaryFaces("exterior_facets")
        self._plex.createLabel("interior_facets")
        fStart, fEnd = self._plex.getHeightStratum(1)  # facets
        for face in range(fStart, fEnd):
            if self._plex.getLabelValue("exterior_facets", face) == -1:
                self._plex.setLabelValue("interior_facets", face, 1)

        # Distribute the dm to all ranks
        if op2.MPI.comm.size > 1:
            self.parallel_sf = self._plex.distribute(overlap=1)

        # Mark OP2 entities and derive the resulting Plex renumbering
        dmplex.mark_entity_classes(self._plex)
        self._plex_renumbering = dmplex.plex_renumbering(plex)

        cStart, cEnd = self._plex.getHeightStratum(0)  # cells
        cell_vertices = self._plex.getConeSize(cStart)
        self._ufl_cell = ufl.Cell(_cells[geometric_dim][cell_vertices],
                                  geometric_dimension = geometric_dim)

        dim = self._plex.getDimension()
        self._cells, self.cell_classes = dmplex.get_entities_by_class(self._plex, dim)

        # Derive a cell numbering from the Plex renumbering
        cell_entity_dofs = np.zeros(dim+1, dtype=np.int32)
        cell_entity_dofs[-1] = 1
        self._cell_numbering = self._plex.createSection(1, [1],
                                                        cell_entity_dofs,
                                                        perm=self._plex_renumbering)

        # Fenics facet and DoF numbering requires a universal vertex numbering
        self._vertex_numbering = None
        vertex_fs = types.FunctionSpace(self, "CG", 1)
        self._vertex_numbering = vertex_fs._universal_numbering

        # Exterior facets
        if self._plex.getStratumSize("exterior_facets", 1) > 0:

            # Order exterior facets by OP2 entity class
            ext_facet = lambda f: self._plex.getLabelValue("exterior_facets", f) == 1
            exterior_facets, exterior_facet_classes = \
                dmplex.get_entities_by_class(self._plex, dim-1, condition=ext_facet)

            # Derive attached boundary IDs
            if self._plex.hasLabel("boundary_ids"):
                boundary_ids = np.zeros(exterior_facets.size, dtype=np.int32)
                for i, facet in enumerate(exterior_facets):
                    boundary_ids[i] = self._plex.getLabelValue("boundary_ids", facet)
            else:
                boundary_ids = None

            exterior_facet_cell = []
            for f in exterior_facets:
                fcells = self._plex.getSupport(f)
                fcells_num = [np.array([self._cell_numbering.getOffset(c)]) for c in fcells]
                exterior_facet_cell.append(np.concatenate(fcells_num))
            exterior_facet_cell = np.array(exterior_facet_cell)

            get_f_no = lambda f: dmplex.facet_numbering(self._plex, self._vertex_numbering, f)
            exterior_local_facet_number = np.array([get_f_no(f) for f in exterior_facets], dtype=np.int32)
            # Note: To implement facets correctly in parallel
            # we need to pass exterior_facet_classes to _Facets()
            self.exterior_facets = _Facets(self, exterior_facets.size,
                                           "exterior",
                                           exterior_facet_cell,
                                           exterior_local_facet_number,
                                           boundary_ids)
        else:
            self.exterior_facets = _Facets(self, 0, "exterior", None, None)

        # Interior facets
        if self._plex.getStratumSize("interior_facets", 1) > 0:

            # Order interior facets by OP2 entity class
            int_facet = lambda f: self._plex.getLabelValue("interior_facets", f) == 1
            interior_facets, interior_facet_classes = \
                dmplex.get_entities_by_class(self._plex, dim-1, condition=int_facet)

            interior_facet_cell = []
            for f in interior_facets:
                fcells = self._plex.getSupport(f)
                fcells_num = [np.array([self._cell_numbering.getOffset(c)]) for c in fcells]
                interior_facet_cell.append(np.concatenate(fcells_num))
            interior_facet_cell = np.array(interior_facet_cell)
            get_f_no = lambda f: dmplex.facet_numbering(self._plex, self._vertex_numbering, f)
            interior_local_facet_number = np.array([get_f_no(f) for f in interior_facets])
            # Note: To implement facets correctly in parallel
            # we need to pass interior_facet_classes to _Facets()
            self.interior_facets = _Facets(self, interior_facets.size,
                                           "interior",
                                           interior_facet_cell,
                                           interior_local_facet_number)
        else:
            self.interior_facets = _Facets(self, 0, "interior", None, None)

        # Note that for bendy elements, this needs to change.
        if periodic_coords is not None:
            if self.ufl_cell().geometric_dimension() != 1:
                raise NotImplementedError("Periodic coordinates in more than 1D are unsupported")
            # We've been passed a periodic coordinate field, so use that.
            self._coordinate_fs = types.VectorFunctionSpace(self, "DG", 1)
            self.coordinates = types.Function(self._coordinate_fs,
                                                    val=periodic_coords,
                                                    name="Coordinates")
        else:
            self._coordinate_fs = types.VectorFunctionSpace(self, "Lagrange", 1)

            # Use the section permutation to re-order Plex coordinates
            plex_coords = self._plex.getCoordinatesLocal().getArray()
            plex_coords = np.reshape(plex_coords, (self.num_vertices(), geometric_dim))
            coordinates = np.empty(plex_coords.shape)
            vStart, vEnd = self._plex.getDepthStratum(0)
            for v in range(vStart, vEnd):
                offset = self._coordinate_fs._global_numbering.getOffset(v)
                coordinates[offset,:] = plex_coords[v-vStart,:]

            self.coordinates = types.Function(self._coordinate_fs,
                                                    val=coordinates,
                                                    name="Coordinates")
        self._dx = ufl.Measure('cell', domain_data=self.coordinates)
        self._ds = ufl.Measure('exterior_facet', domain_data=self.coordinates)
        self._dS = ufl.Measure('interior_facet', domain_data=self.coordinates)
        # Set the domain_data on all the default measures to this coordinate field.
        for measure in [ufl.dx, ufl.ds, ufl.dS]:
            measure._domain_data = self.coordinates

    def _from_gmsh(self, filename, dim=0, periodic_coords=None):
        """Read a Gmsh .msh file from `filename`"""
        basename, ext = os.path.splitext(filename)
        self.name = filename

        # Create a read-only PETSc.Viewer
        gmsh_viewer = PETSc.Viewer().create()
        gmsh_viewer.setType("ascii")
        gmsh_viewer.setFileMode("r")
        gmsh_viewer.setFileName(filename)
        gmsh_plex = PETSc.DMPlex().createGmsh(gmsh_viewer)

        if gmsh_plex.hasLabel("Face Sets"):
            boundary_ids = gmsh_plex.getLabelIdIS("Face Sets").getIndices()
            gmsh_plex.createLabel("boundary_ids")
            for bid in boundary_ids:
                faces = gmsh_plex.getStratumIS("Face Sets", bid).getIndices()
                for f in faces:
                    gmsh_plex.setLabelValue("boundary_ids", f, bid)

        self._from_dmplex(gmsh_plex, dim, periodic_coords)

    def _from_exodus(self, filename, dim=0):
        self.name = filename
        dmplex = PETSc.DMPlex().createExodusFromFile(filename)

        boundary_ids = dmplex.getLabelIdIS("Face Sets").getIndices()
        dmplex.createLabel("boundary_ids")
        for bid in boundary_ids:
            faces = dmplex.getStratumIS("Face Sets", bid).getIndices()
            for f in faces:
                dmplex.setLabelValue("boundary_ids", f, bid)

        self._from_dmplex(dmplex)

    def _from_cgns(self, filename, dim=0):
        self.name = filename
        dmplex = PETSc.DMPlex().createCGNSFromFile(filename)

        #TODO: Add boundary IDs
        self._from_dmplex(dmplex)

    def _from_triangle(self, filename, dim=0, periodic_coords=None):
        """Read a set of triangle mesh files from `filename`"""
        self.name = filename
        basename, ext = os.path.splitext(filename)

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
        if dim == 0:
            dim = tdim

        with open(basename+".node") as nodefile:
            header = np.fromfile(nodefile, dtype=np.int32, count=2, sep=' ')
            nodecount = header[0]
            nodedim = header[1]
            coordinates = np.loadtxt(nodefile, usecols=range(1,dim+1), skiprows=1, delimiter=' ')
            assert nodecount == coordinates.shape[0]

        with open(basename+".ele") as elefile:
            header = np.fromfile(elefile, dtype=np.int32, count=2, sep=' ')
            elecount = header[0]
            eledim = header[1]
            eles = np.loadtxt(elefile, usecols=range(1,eledim+1), dtype=np.int32, skiprows=1, delimiter=' ')
            assert elecount == eles.shape[0]

        cells = map(lambda c: c-1, eles)
        dmplex = _from_cell_list(tdim, cells, coordinates, comm=op2.MPI.comm)

        # Apply boundary IDs
        facets = None
        try:
            header = np.fromfile(facetfile, dtype=np.int32, count=2, sep=' ')
            edgecount = header[0]
            edgedim = header[1]
            facets = np.loadtxt(facetfile, usecols=range(1,tdim+2), dtype=np.int32, skiprows=0, delimiter=' ')
        finally:
            facetfile.close()

        if facets is not None:
            vStart, vEnd = dmplex.getDepthStratum(0)   # vertices
            for facet in facets:
                bid = facet[-1]
                vertices = map(lambda v: v + vStart - 1, facet[:-1])
                join = dmplex.getJoin(vertices)
                dmplex.setLabelValue("boundary_ids", join[0], bid)

        self._from_dmplex(dmplex, dim, periodic_coords)

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :func:`init_cell_orientations` to initialise this data."""
        return self._cell_orientations.data_ro

    def init_cell_orientations(self, expr):
        """Compute and initialise `cell_orientations` relative to a specified orientation.

        :arg expr: an :class:`.Expression` evaluated to produce a
             reference normal direction.

        """
        if expr.shape()[0] != 3:
            raise NotImplementedError('Only implemented for 3-vectors')
        if self.ufl_cell() != ufl.Cell('triangle', 3):
            raise NotImplementedError('Only implemented for triangles embedded in 3d')

        v0 = lambda x: ast.Symbol("v0", (x,))
        v1 = lambda x: ast.Symbol("v1", (x,))
        n = lambda x: ast.Symbol("n", (x,))
        x = lambda x: ast.Symbol("x", (x,))
        coords = lambda x, y: ast.Symbol("coords", (x, y))

        body = []
        body += [ast.Decl("double", v(3)) for v in [v0, v1, n, x]]
        body.append(ast.Decl("double", "dot"))
        body.append(ast.Assign("dot", 0.0))
        body.append(ast.Decl("int", "i"))
        body.append(ast.For(ast.Assign("i", 0), ast.Less("i", 3), ast.Incr("i", 1),
                            [ast.Assign(v0("i"), ast.Sub(coords(1, "i"), coords(0, "i"))),
                             ast.Assign(v1("i"), ast.Sub(coords(2, "i"), coords(0, "i"))),
                             ast.Assign(x("i"), 0.0)]))
        # n = v0 x v1
        body.append(ast.Assign(n(0), ast.Sub(ast.Prod(v0(1), v1(2)), ast.Prod(v0(2), v1(1)))))
        body.append(ast.Assign(n(1), ast.Sub(ast.Prod(v0(2), v1(0)), ast.Prod(v0(0), v1(2)))))
        body.append(ast.Assign(n(2), ast.Sub(ast.Prod(v0(0), v1(1)), ast.Prod(v0(1), v1(0)))))

        body.append(ast.For(ast.Assign("i", 0), ast.Less("i", 3), ast.Incr("i", 1),
                            [ast.Incr(x(j), coords("i", j)) for j in range(3)]))

        body.extend([ast.FlatBlock("dot += (%(x)s) * n[%(i)d];\n" % {"x": x, "i": i})
                     for i, x in enumerate(expr.code)])
        body.append(ast.Assign("*orientation", ast.Ternary(ast.Less("dot", 0), 1, 0)))

        kernel = op2.Kernel(ast.FunDecl("void", "cell_orientations",
                                        [ast.Decl("int*", "orientation"),
                                         ast.Decl("double**", "coords")],
                                        ast.Block(body)),
                            "cell_orientations")

        op2.par_loop(kernel, self.cell_set,
                     self._cell_orientations(op2.WRITE),
                     self.coordinates.dat(op2.READ, self.coordinates.cell_node_map()))
        self._cell_orientations._force_evaluation(read=True, write=False)

    def cells(self):
        return self._cells

    def ufl_cell(self):
        return self._ufl_cell

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

    @utils.cached_property
    def cell_set(self):
        size = self.cell_classes
        return self.parent.cell_set if self.parent else \
            op2.Set(size, "%s_cells" % self.name)

    def compute_boundaries(self):
        '''Currently a no-op for flop.py compatibility.'''
        pass

class ExtrudedMesh(Mesh):
    """Build an extruded mesh from a 2D input mesh

    :arg mesh:           2D unstructured mesh
    :arg layers:         number of extruded cell layers in the "vertical"
                         direction.
    :arg kernel:         a :class:`pyop2.Kernel` to produce coordinates for the extruded
                         mesh see :func:`make_extruded_coords` for more details.
    :arg layer_height:   the height between layers when all layers are
                         evenly spaced.  If no layer_height is
                         provided and a preset extrusion_type is
                         given, the value defaults to 1/layers
                         (i.e. the extruded mesh has unit extent in
                         the extruded direction).
    :arg extrusion_type: refers to how the coordinates are computed for the
                         evenly spaced layers:
                         `uniform`: layers are computed in the extra dimension
                         generated by the extrusion process
                         `radial`: radially extrudes the mesh points in the
                         outwards direction from the origin.

    If a kernel for extrusion is passed in, this overrides both the
    layer_height and extrusion_type options (should they have also
    been specified)."""
    def __init__(self, mesh, layers, kernel=None, layer_height=None, extrusion_type='uniform'):
        if kernel is None and extrusion_type is None:
            raise RuntimeError("Please provide a kernel or a preset extrusion_type ('uniform' or 'radial') for extruding the mesh")
        self._old_mesh = mesh
        if layers < 1:
            raise RuntimeError("Must have at least one layer of extruded cells (not %d)" % layers)
        # All internal logic works with layers of base mesh (not layers of cells)
        self._layers = layers + 1
        self._cells = mesh._cells
        self.parent = mesh.parent
        self.uid = mesh.uid
        self._vertex_numbering = mesh._vertex_numbering
        self.name = mesh.name
        self._plex = mesh._plex
        self._plex_renumbering = mesh._plex_renumbering
        self._cell_numbering = mesh._cell_numbering

        interior_f = self._old_mesh.interior_facets
        self._interior_facets = _Facets(self, interior_f.count,
                                       "interior",
                                       interior_f.facet_cell,
                                       interior_f.local_facet_number)
        exterior_f = self._old_mesh.exterior_facets
        self._exterior_facets = _Facets(self, exterior_f.count,
                                           "exterior",
                                           exterior_f.facet_cell,
                                           exterior_f.local_facet_number,
                                           exterior_f.markers)

        self.ufl_cell_element = ufl.FiniteElement("Lagrange",
                                               domain = mesh._ufl_cell,
                                               degree = 1)
        self.ufl_interval_element = ufl.FiniteElement("Lagrange",
                                               domain = ufl.Cell("interval",1),
                                               degree = 1)

        self.fiat_base_element = fiat_from_ufl_element(self.ufl_cell_element)
        self.fiat_vert_element = fiat_from_ufl_element(self.ufl_interval_element)

        fiat_element = FIAT.tensor_finite_element.TensorFiniteElement(self.fiat_base_element, self.fiat_vert_element)

        self._ufl_cell = ufl.OuterProductCell(mesh._ufl_cell, ufl.Cell("interval",1))

        flat_temp = fiat_element.flattened_element()

        # Calculated dofs_per_column from flattened_element and layers.
        # The mirrored elements have to be counted only once.
        # Then multiply by layers and layers - 1 accordingly.
        self.dofs_per_column = compute_extruded_dofs(fiat_element, flat_temp.entity_dofs(), layers)

        #Compute Coordinates of the extruded mesh
        if layer_height is None:
            # Default to unit
            layer_height = 1.0 / layers

        self._coordinate_fs = types.VectorFunctionSpace(self, mesh._coordinate_fs.ufl_element().family(),
                                                        mesh._coordinate_fs.ufl_element().degree(),
                                                        vfamily="CG",
                                                        vdegree=1)

        self.coordinates = types.Function(self._coordinate_fs)
        make_extruded_coords(self, layer_height, extrusion_type=extrusion_type,
                             kernel=kernel)
        self._coordinates = self.coordinates.dat.data_ro_with_halos

        self._dx = ufl.Measure('cell', domain_data=self.coordinates)
        self._ds = ufl.Measure('exterior_facet', domain_data=self.coordinates)
        self._dS = ufl.Measure('interior_facet', domain_data=self.coordinates)
        # Set the domain_data on all the default measures to this coordinate field.
        for measure in [ufl.ds, ufl.dS, ufl.dx, ufl.ds_t, ufl.ds_b, ufl.ds_v, ufl.dS_h, ufl.dS_v]:
            measure._domain_data = self.coordinates

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    @utils.cached_property
    def cell_set(self):
        return self.parent.cell_set if self.parent else \
            op2.ExtrudedSet(self._old_mesh.cell_set, layers=self._layers)

    @property
    def exterior_facets(self):
        return self._exterior_facets

    @property
    def interior_facets(self):
        return self._interior_facets

    @property
    def geometric_dimension(self):
        return self._ufl_cell.geometric_dimension()


class Halo(object):
    """Build a Halo associated with the appropriate FunctionSpace.

    The Halo is derived from a PetscSF object and builds the global
    to universal numbering map from the respective PetscSections."""

    def __init__(self, petscsf, global_numbering, universal_numbering):
        self._tag = _new_uid()
        self._comm = op2.MPI.comm
        self._nprocs = self.comm.size
        self._sends = defaultdict(list)
        self._receives = defaultdict(list)
        self._gnn2unn = None
        remote_sends = defaultdict(list)

        if op2.MPI.comm.size <= 1:
            return

        # Sort the SF by local indices
        nroots, nleaves, local, remote = petscsf.getGraph()
        local_new, remote_new = (list(x) for x in zip(*sorted(zip(local, remote), key=lambda x: x[0])))
        petscsf.setGraph(nroots, nleaves, local_new, remote_new)

        # Derive local receives and according remote sends
        nroots, nleaves, local, remote = petscsf.getGraph()
        for local, (rank, index) in zip(local, remote):
            if rank != self.comm.rank:
                self._receives[rank].append(local)
                remote_sends[rank].append(index)

        # Propagate remote send lists to the actual sender
        send_reqs = []
        for p in range(self._nprocs):
            # send sizes
            if p != self._comm.rank:
                s = np.array(len(remote_sends[p]), dtype=np.int32)
                send_reqs.append(self.comm.Isend(s, dest=p, tag=self.tag))

        recv_reqs = []
        sizes = [np.empty(1, dtype=np.int32) for _ in range(self._nprocs)]
        for p in range(self._nprocs):
            # receive sizes
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(sizes[p], source=p, tag=self.tag))

        MPI.Request.Waitall(recv_reqs)
        MPI.Request.Waitall(send_reqs)

        for p in range(self._nprocs):
            # allocate buffers
            if p != self._comm.rank:
                self._sends[p] = np.empty(sizes[p], dtype=np.int32)

        send_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                send_buf = np.array(remote_sends[p], dtype=np.int32)
                send_reqs.append(self.comm.Isend(send_buf, dest=p, tag=self.tag))

        recv_reqs = []
        for p in range(self._nprocs):
            if p != self._comm.rank:
                recv_reqs.append(self.comm.Irecv(self._sends[p], source=p, tag=self.tag))

        MPI.Request.Waitall(send_reqs)
        MPI.Request.Waitall(recv_reqs)

        # Build Global-To-Universal mapping
        pStart, pEnd = global_numbering.getChart()
        self._gnn2unn = np.zeros(global_numbering.getStorageSize(), dtype=np.int32)
        for p in range(pStart, pEnd):
            dof = global_numbering.getDof(p)
            goff = global_numbering.getOffset(p)
            uoff = universal_numbering.getOffset(p)
            if uoff < 0:
                uoff = (-1*uoff)-1
            for c in range(dof):
                self._gnn2unn[goff+c] = uoff+c

    @utils.cached_property
    def op2_halo(self):
        if not self.sends and not self.receives:
            return None
        return op2.Halo(self.sends, self.receives,
                        comm=self.comm, gnn2unn=self.gnn2unn)

    @property
    def comm(self):
        return self._comm

    @property
    def tag(self):
        return self._tag

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def sends(self):
        return self._sends

    @property
    def receives(self):
        return self._receives

    @property
    def gnn2unn(self):
        return self._gnn2unn


class FunctionSpaceBase(Cached):
    """Base class for :class:`.FunctionSpace`, :class:`.VectorFunctionSpace` and
    :class:`.MixedFunctionSpace`.

    .. note ::

        Users should not directly create objects of this class, but one of its
        derived types.
    """

    _cache = {}

    def __init__(self, mesh, element, name=None, dim=1, rank=0):
        """
        :param mesh: :class:`Mesh` to build this space on
        :param element: :class:`ufl.FiniteElementBase` to build this space from
        :param name: user-defined name for this space
        :param dim: vector space dimension of a :class:`.VectorFunctionSpace`
        :param rank: rank of the space, not the value rank
        """

        self._ufl_element = element

        # Compute the FIAT version of the UFL element above
        self.fiat_element = fiat_from_ufl_element(element)

        if isinstance(mesh, ExtrudedMesh):
            # Set up some extrusion-specific things
            # The bottom layer maps will come from element_dof_list
            # dof_count is the total number of dofs in the extruded mesh

            # Get the flattened version of the FIAT element
            self.flattened_element = self.fiat_element.flattened_element()
            entity_dofs = self.flattened_element.entity_dofs()
            self._dofs_per_cell = [len(entity)*len(entity[0]) for d, entity in entity_dofs.iteritems()]

            # Compute the number of DoFs per dimension on top/bottom and sides
            entity_dofs = self.fiat_element.entity_dofs()
            top_dim = mesh._plex.getDimension()
            self._xtr_hdofs = [len(entity_dofs[(d,0)][0]) for d in range(top_dim+1)]
            self._xtr_vdofs = [len(entity_dofs[(d,1)][0]) for d in range(top_dim+1)]

            # Compute the dofs per column
            self.dofs_per_column = compute_extruded_dofs(self.fiat_element,
                                                         self.flattened_element.entity_dofs(),
                                                         mesh._layers)

            # Compute the offset for the extrusion process
            self.offset = compute_offset(self.fiat_element.entity_dofs(),
                                         self.flattened_element.entity_dofs(),
                                         self.fiat_element.space_dimension())

            # Compute the top and bottom masks to identify boundary dofs
            b_mask = self.fiat_element.get_lower_mask()
            t_mask = self.fiat_element.get_upper_mask()

            self.bt_masks = (b_mask, t_mask)

            self.extruded = True

            self._dofs_per_entity = self.dofs_per_column
        else:
            # If not extruded specific, set things to None/False, etc.
            self.offset = None
            self.bt_masks = None
            self.dofs_per_column = np.zeros(1, np.int32)
            self.extruded = False

            entity_dofs = self.fiat_element.entity_dofs()
            self._dofs_per_entity = [len(entity[0]) for d, entity in entity_dofs.iteritems()]
            self._dofs_per_cell = [len(entity)*len(entity[0]) for d, entity in entity_dofs.iteritems()]

        self.name = name
        self._dim = dim
        self._mesh = mesh
        self._index = None

        # Create the PetscSection mapping topological entities to DoFs
        self._global_numbering = mesh._plex.createSection(1, [1], self._dofs_per_entity,
                                                          perm=mesh._plex_renumbering)
        mesh._plex.setDefaultSection(self._global_numbering)
        self._universal_numbering = mesh._plex.getDefaultGlobalSection()

        # Re-initialise the DefaultSF with the numbering for this FS
        mesh._plex.createDefaultSF(self._global_numbering,
                                   self._universal_numbering)

        # Derive the Halo from the DefaultSF
        self._halo = Halo(mesh._plex.getDefaultSF(),
                          self._global_numbering,
                          self._universal_numbering)

        # Compute entity class offsets
        self.dof_classes = [0, 0, 0, 0]
        for d in range(mesh._plex.getDimension()+1):
            ncore = mesh._plex.getStratumSize("op2_core", d)
            nowned = mesh._plex.getStratumSize("op2_non_core", d)
            nhalo = mesh._plex.getStratumSize("op2_exec_halo", d)
            ndofs = self._dofs_per_entity[d]
            self.dof_classes[0] += ndofs * ncore
            self.dof_classes[1] += ndofs * (ncore + nowned)
            self.dof_classes[2] += ndofs * (ncore + nowned + nhalo)
            self.dof_classes[3] += ndofs * (ncore + nowned + nhalo)

        self._node_count = self._global_numbering.getStorageSize()
        self.cell_node_list = np.array([self._get_cell_nodes(c) for c in mesh.cells()])

        if mesh._plex.getStratumSize("interior_facets", 1) > 0:
            dim = mesh._plex.getDimension()
            int_facet = lambda f: mesh._plex.getLabelValue("interior_facets", f) == 1
            interior_facets = dmplex.get_entities_by_class(mesh._plex, dim-1, condition=int_facet)[0]

            interior_facet_eles = []
            for f in interior_facets:
                fcells = self._mesh._plex.getSupport(f)
                fcells_num = [np.array([mesh._cell_numbering.getOffset(c)]) for c in fcells]
                interior_facet_eles.append(np.concatenate(fcells_num))
            self.interior_facet_node_list = np.array([np.concatenate([self.cell_node_list[e] for e in eles]) for eles in interior_facet_eles])
        else:
            self.interior_facet_node_list = None

        if mesh._plex.getStratumSize("exterior_facets", 1) > 0:
            dim = mesh._plex.getDimension()
            ext_facet = lambda f: mesh._plex.getLabelValue("exterior_facets", f) == 1
            exterior_facets = dmplex.get_entities_by_class(mesh._plex, dim-1, condition=ext_facet)[0]

            exterior_facet_eles = []
            for f in exterior_facets:
                fcells = self._mesh._plex.getSupport(f)
                fcells_num = [np.array([mesh._cell_numbering.getOffset(c)]) for c in fcells]
                exterior_facet_eles.append(np.concatenate(fcells_num))
            self.exterior_facet_node_list = np.array([np.concatenate([self.cell_node_list[e] for e in eles]) for eles in exterior_facet_eles])
        else:
            self.exterior_facet_node_list = None

        # Note: this is the function space rank. The value rank may be different.
        self.rank = rank

        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        self._cell_node_map_cache = {}
        self._exterior_facet_map_cache = {}
        self._interior_facet_map_cache = {}

    def _get_cell_nodes(self, cell):
        plex = self._mesh._plex
        closure = plex.getTransitiveClosure(cell)[0]
        if self._dofs_per_entity[0] > 0:
            vertex_numbering = self._universal_numbering
        else:
            vertex_numbering = self._mesh._vertex_numbering
        numbering = dmplex.closure_numbering(plex, vertex_numbering, closure,
                                             self._dofs_per_entity)

        offset = 0
        cell_nodes = np.empty(sum(self._dofs_per_cell), dtype=np.int32)
        if isinstance(self._mesh, ExtrudedMesh):
            # Instead of using the numbering directly, we step through
            # all points and build the numbering for each entity
            # according to the extrusion rules.
            dim = plex.getDimension()
            flat_entity_dofs = self.flattened_element.entity_dofs()
            hdofs = self._xtr_hdofs
            vdofs = self._xtr_vdofs

            for d in range(dim+1):
                pStart, pEnd = plex.getDepthStratum(d)
                points = filter(lambda x: pStart<=x and x<pEnd, numbering)
                for i in range(len(points)):
                    p = points[i]
                    if self._global_numbering.getDof(p) > 0:
                        glbl = self._global_numbering.getOffset(p)

                        # For extruded entities the numberings are:
                        # Global: [bottom[:], top[:], side[:]]
                        # Local:  [bottom[i], top[i], side[i] for i in bottom[:]]
                        #
                        # eg. extruded P3 facet:
                        #       Local            Global
                        #  --1---6---11--   --12---13---14--
                        #  | 4   9   14 |   |  5    8   11 |
                        #  | 3   8   13 |   |  4    7   10 |
                        #  | 2   7   12 |   |  3    6    9 |
                        #  --0---5---10--   ---0----1----2--
                        #
                        # cell_nodes = [0,12,3,4,5,1,13,6,7,8,2,14,9,10,11]

                        lcl_dofs = flat_entity_dofs[d][i]
                        glbl_dofs = np.zeros(len(lcl_dofs), dtype=np.int32)
                        glbl_dofs[:hdofs[d]] = range(glbl,glbl+hdofs[d])
                        glbl_sides = glbl + hdofs[d]
                        glbl_dofs[hdofs[d]:hdofs[d]+vdofs[d]] = range(glbl_sides, glbl_sides + vdofs[d])
                        glbl_top = glbl + hdofs[d] + vdofs[d]
                        glbl_dofs[vdofs[d]+hdofs[d]:vdofs[d]+2*hdofs[d]] = range(glbl_top, glbl_top+hdofs[d])
                        for l, g in zip(lcl_dofs, glbl_dofs):
                            cell_nodes[l] = g

                        offset += 2*hdofs[d] + vdofs[d]

        else:
            for n in numbering:
                dof = self._global_numbering.getDof(n)
                off = self._global_numbering.getOffset(n)
                for i in range(dof):
                    cell_nodes[offset+i] = off+i
                offset += dof
        return cell_nodes

    @property
    def index(self):
        """Position of this :class:`FunctionSpaceBase` in the
        :class:`.MixedFunctionSpace` it was extracted from."""
        return self._index

    @property
    def node_count(self):
        """The number of global nodes in the function space. For a
        plain :class:`.FunctionSpace` this is equal to
        :attr:`dof_count`, however for a :class:`.VectorFunctionSpace`,
        the :attr:`dof_count`, is :attr:`dim` times the
        :attr:`node_count`."""

        return self._node_count

    @property
    def dof_count(self):
        """The number of global degrees of freedom in the function
        space. Cf. :attr:`node_count`."""

        return self._node_count*self._dim

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`.FunctionSpace`. One or (for
        :class:`.VectorFunctionSpace`\s) more degrees of freedom are
        stored at each node.
        """

        name = "%s_nodes" % self.name
        if self._halo:
            s = op2.Set(self.dof_classes, name,
                        halo=self._halo.op2_halo)
            if self.extruded:
                return op2.ExtrudedSet(s, layers=self._mesh.layers)
            return s
        else:
            s = op2.Set(self.node_count, name)
            if self.extruded:
                return op2.ExtrudedSet(s, layers=self._mesh.layers)
            return s

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`.FunctionSpace`."""
        return op2.DataSet(self.node_set, self.dim)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.Dat` defined on the
        :attr:`dof_dset` of this :class:`.Function`."""
        return op2.Dat(self.dof_dset, val, valuetype, name, uid=uid)


    def cell_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.cell_node_map()
        else:
            parent = None

        return self._map_cache(self._cell_node_map_cache,
                               self._mesh.cell_set,
                               self.cell_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               "cell_node",
                               self.offset,
                               parent)

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.interior_facet_node_map()
        else:
            parent = None

        offset = self.cell_node_map().offset
        return self._map_cache(self._interior_facet_map_cache,
                               self._mesh.interior_facets.set,
                               self.interior_facet_node_list,
                               2*self.fiat_element.space_dimension(),
                               bcs,
                               "interior_facet_node",
                               offset=np.append(offset, offset),
                               parent=parent)

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.exterior_facet_node_map()
        else:
            parent = None

        facet_set = self._mesh.exterior_facets.set
        if isinstance(self._mesh, ExtrudedMesh):
            name = "extruded_exterior_facet_node"
            offset = self.offset
        else:
            name = "exterior_facet_node"
            offset = None
        return self._map_cache(self._exterior_facet_map_cache,
                               facet_set,
                               self.exterior_facet_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               name,
                               parent=parent,
                               offset=offset)

    def bottom_nodes(self):
        """Return a list of the bottom boundary nodes of the extruded mesh.
        The bottom mask is applied to every bottom layer cell to get the
        dof ids."""
        return np.unique(self.cell_node_list[:, self.bt_masks[0]])

    def top_nodes(self):
        """Return a list of the top boundary nodes of the extruded mesh.
        The top mask is applied to every top layer cell to get the dof ids."""
        voffs = self.offset.take(self.bt_masks[1])*(self._mesh.layers-2)
        return np.unique(self.cell_node_list[:, self.bt_masks[1]] + voffs)

    def _map_cache(self, cache, entity_set, entity_node_list, map_arity, bcs, name,
                   offset=None, parent=None):
        if bcs is None:
            lbcs = None
        else:
            if not all(bc.function_space() == self for bc in bcs):
              raise RuntimeError("DirichletBC defined on a different FunctionSpace!")
            # Ensure bcs is a tuple in a canonical order for the hash key.
            lbcs = tuple(sorted(bcs, key=lambda bc: bc.__hash__()))
        try:
            # Cache hit
            return cache[lbcs]
        except KeyError:
            # Cache miss.
            if not lbcs:
                new_entity_node_list = entity_node_list
            elif offset is not None:
                l = [bc.nodes for bc in bcs if bc.sub_domain not in ['top', 'bottom']]
                if l:
                    bcids = reduce(np.union1d, l)
                    nl = entity_node_list.ravel()
                    new_entity_node_list = np.where(np.in1d(nl, bcids), -10000000, nl)
                else:
                    new_entity_node_list = entity_node_list
            else:
                bcids = reduce(np.union1d, [bc.nodes for bc in bcs])
                nl = entity_node_list.ravel()
                new_entity_node_list = np.where(np.in1d(nl, bcids), -1, nl)

            cache[lbcs] = op2.Map(entity_set, self.node_set,
                                  map_arity,
                                  new_entity_node_list,
                                  ("%s_"+name) % (self.name),
                                  offset,
                                  parent,
                                  self.bt_masks)

            return cache[lbcs]

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''

        el = self.fiat_element

        if isinstance(self._mesh, ExtrudedMesh):
            # The facet is indexed by (base-ele-codim 1, 1) for
            # extruded meshes.
            # e.g. for the two supported options of
            # triangle x interval interval x interval it's (1, 1) and
            # (0, 1) respectively.
            if self._mesh.geometric_dimension == 3:
                dim = (1,1)
            elif self._mesh.geometric_dimension == 2:
                dim = (0,1)
            else:
                raise RuntimeError("Dimension computation for other than 2D or 3D extruded meshes not supported.")
        else:
            # Facets have co-dimension 1
            dim = len(el.get_reference_element().topology)-1
            dim = dim - 1

        nodes_per_facet = \
            len(self.fiat_element.entity_closure_dofs()[dim][0])

        facet_set = self._mesh.exterior_facets.set

        fs_dat = op2.Dat(facet_set**el.space_dimension(),
                         data=self.exterior_facet_node_map().values_with_halo)

        facet_dat = op2.Dat(facet_set**nodes_per_facet,
                            dtype=np.int32)

        local_facet_nodes = np.array(
            [dofs for e, dofs in el.entity_closure_dofs()[dim].iteritems()])

        # Helper function to turn the inner index of an array into c
        # array literals.
        c_array = lambda xs: "{"+", ".join(map(str, xs))+"}"

        body = ast.Block([ast.Decl("int", ast.Symbol("l_nodes", (len(el.get_reference_element().topology[dim]),
                                                                 nodes_per_facet)),
                                   init=ast.ArrayInit(c_array(map(c_array, local_facet_nodes))),
                                   qualifiers=["const"]),
                          ast.For(ast.Decl("int", "n", 0),
                                  ast.Less("n", nodes_per_facet),
                                  ast.Incr("n", 1),
                                  ast.Assign(ast.Symbol("facet_nodes", ("n",)),
                                             ast.Symbol("cell_nodes", ("l_nodes[facet[0]][n]",))))
                          ])

        kernel = op2.Kernel(ast.FunDecl("void", "create_bc_node_map",
                                        [ast.Decl("int*", "cell_nodes"),
                                         ast.Decl("int*", "facet_nodes"),
                                         ast.Decl("unsigned int*", "facet")],
                                        body),
                            "create_bc_node_map")

        local_facet_dat = self._mesh.exterior_facets.local_facet_dat
        op2.par_loop(kernel, facet_set,
                     fs_dat(op2.READ),
                     facet_dat(op2.WRITE),
                     local_facet_dat(op2.READ))

        if isinstance(self._mesh, ExtrudedMesh):
            offset = extract_offset(self.offset,
                                    facet_dat.data_ro_with_halos[0],
                                    self.cell_node_map().values[0])
        else:
            offset = None
        return op2.Map(facet_set, self.node_set,
                       nodes_per_facet,
                       facet_dat.data_ro_with_halos,
                       name="exterior_facet_boundary_node",
                       offset=offset)


    @property
    def dim(self):
        """The vector dimension of the :class:`.FunctionSpace`. For a
        :class:`.FunctionSpace` this is always one. For a
        :class:`.VectorFunctionSpace` it is the value given to the
        constructor, and defaults to the geometric dimension of the :class:`Mesh`. """
        return self._dim

    @property
    def cdim(self):
        """The sum of the vector dimensions of the :class:`.FunctionSpace`. For a
        :class:`.FunctionSpace` this is always one. For a
        :class:`.VectorFunctionSpace` it is the value given to the
        constructor, and defaults to the geometric dimension of the :class:`Mesh`. """
        return self._dim

    def ufl_element(self):
        """The :class:`ufl.FiniteElement` used to construct this
        :class:`FunctionSpace`."""
        return self._ufl_element

    def mesh(self):
        """The :class:`Mesh` used to construct this :class:`.FunctionSpace`."""
        return self._mesh

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __getitem__(self, i):
        """Return ``self`` if ``i`` is 0 or raise an exception."""
        if i != 0:
            raise IndexError("Only index 0 supported on a FunctionSpace")
        return self

    def __mul__(self, other):
        """Create a :class:`.MixedFunctionSpace` composed of this
        :class:`.FunctionSpace` and other"""
        return types.MixedFunctionSpace((self, other))

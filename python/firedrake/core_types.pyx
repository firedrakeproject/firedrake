# cython: embedsignature=True
# Re-implementation of state_types for firedrake.
import os
from mpi4py import MPI
import numpy as np
import cgen

import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc

from libc.stdlib cimport malloc, free
# Necessary so that we can use memoryviews
from cython cimport view
cimport numpy as np
cimport cpython as py

import ufl
from ufl import *
import FIAT

from pyop2 import op2
from pyop2.caching import Cached
from pyop2.utils import as_tuple

import utils

cimport fluidity_types as ft

cdef extern from "capsulethunk.h": pass

np.import_array()

cdef char *_function_space_name = "function_space"
cdef char *_vector_field_name = "vector_field"
cdef char *_mesh_name = "mesh_type"
cdef char *_halo_name = "halo_type"
ft.python_cache = False

"""Mapping from mesh file extension to file format"""
_file_extensions = {
    ".node" : "triangle",
    ".msh"  : "gmsh"
    # Note: need to include the file extension for exodus.
    }

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
    base_coords = extruded_mesh._old_mesh._coordinate_field
    ext_coords = extruded_mesh._coordinate_field
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


# C utility functions, not seen in python module
cdef void function_space_destructor(object capsule):
    cdef void *fs = py.PyCapsule_GetPointer(capsule, _function_space_name)
    ft.function_space_destructor_f(fs)

cdef void vector_field_destructor(object capsule):
    cdef void *vf = py.PyCapsule_GetPointer(capsule, _vector_field_name)
    ft.vector_field_destructor_f(vf)

cdef ft.element_t as_element(object fiat_element):
    """Convert a FIAT element into a Fluidity element_t struct.

    This is used to build Fluidity element_types from FIAT element
    descriptions.
    """
    cdef ft.element_t element
    if isinstance(fiat_element.get_reference_element(), FIAT.reference_element.two_product_cell):
        # Use the reference element of the horizontal element.
        f_element = fiat_element.flattened_element()
    else:
        f_element = fiat_element

    element.dimension = len(f_element.get_reference_element().topology) - 1
    element.vertices = len(f_element.get_reference_element().vertices)
    element.ndof = f_element.space_dimension()
    element.degree = f_element.degree()

    # Need to free these when you're done
    element.dofs_per = <int *>malloc((element.dimension + 1) * sizeof(int))
    element.entity_dofs = <int *>malloc(3 * element.ndof * sizeof(int))

    entity_dofs = f_element.entity_dofs()
    cdef int d
    cdef int e
    cdef int dof
    cdef int dof_pos = 0
    for d, dim_entities in entity_dofs.iteritems():
        element.dofs_per[d] = 0
        for e, this_entity_dofs in dim_entities.iteritems():
            # This overwrites the same data a few times, but all of
            # this_entity_dofs are the same length for a given
            # entity_dimension
            element.dofs_per[d] = len(this_entity_dofs)
            for dof in this_entity_dofs:
                element.entity_dofs[dof_pos] = d
                element.entity_dofs[dof_pos+1] = e
                element.entity_dofs[dof_pos+2] = dof
                dof_pos += 3

    return element

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

        if measure.domain_id() in [measure.DOMAIN_ID_EVERYWHERE,
                                   measure.DOMAIN_ID_OTHERWISE]:
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

        self._plex = plex

        if self._plex:
            self._from_dmplex(self._plex, periodic_coords)
        else:
            self._from_file(filename, dim, periodic_coords)

        self._cell_orientations = op2.Dat(self.cell_set, dtype=np.int32,
                                          name="cell_orientations")
        # -1 is uninitialised.
        self._cell_orientations.data[:] = -1

    def _from_dmplex(self, plex, periodic_coords=None):
        """ Create mesh from DMPlex object """

        self._plex = plex
        self._dim  = plex.getDimension()

        cStart, cEnd = self._plex.getHeightStratum(0)  # cells
        fStart, fEnd = self._plex.getHeightStratum(1)  # facets
        vStart, vEnd = self._plex.getDepthStratum(0)   # vertices
        self._entities = np.zeros(self._dim+1, dtype=np.int)
        self._entities[:] = -1 # Ensure that 3d edges get an out of band value.
        self._entities[0]  = vEnd - vStart  # vertex count
        self._entities[-1] = cEnd - cStart  # cell count
        self._entities[-2] = fEnd - fStart  # facet count
        self.uid = _new_uid()

        self._ufl_cell = ufl.Cell(_cells[self._dim][self._dim+1], self._dim)
        self._cells = np.array([self._plex.getCone(c) for c in range(cStart, cEnd)])

        # TODO: Add facet and boundary ID support
        boundary_ids = None
        self.exterior_facets = _Facets(self, 0, "exterior", None, None)
        self.interior_facets = _Facets(self, 0, "interior", None, None)

        # Note that for bendy elements, this needs to change.
        if periodic_coords is not None:
            if self.ufl_cell().geometric_dimension() != 1:
                raise NotImplementedError("Periodic coordinates in more than 1D are unsupported")
            # We've been passed a periodic coordinate field, so use that.
            self._coordinate_fs = types.VectorFunctionSpace(self, "DG", 1)
            self._coordinate_field = types.Function(self._coordinate_fs,
                                                    val=periodic_coords,
                                                    name="Coordinates")
        else:
            self._coordinate_fs = types.VectorFunctionSpace(self, "Lagrange", 1)

            plex_coords = self._plex.getCoordinates().getArray()
            self._coordinates = np.reshape(plex_coords, (vEnd - vStart, self._dim))

            self._coordinate_field = types.Function(self._coordinate_fs,
                                                    val=self._coordinates,
                                                    name="Coordinates")
        self._dx = Measure('cell', domain_data=self._coordinate_field)
        self._ds = Measure('exterior_facet', domain_data=self._coordinate_field)
        self._dS = Measure('interior_facet', domain_data=self._coordinate_field)
        # Set the domain_data on all the default measures to this coordinate field.
        for measure in [ufl.dx, ufl.ds, ufl.dS]:
            measure._domain_data = self._coordinate_field

        # TODO: Add region ID support
        self.region_ids = None

    def _from_file(self, filename, dim=0, periodic_coords=None):
        """Read a mesh from `filename`

        The extension of the filename determines the mesh type."""
        basename, ext = os.path.splitext(filename)

        # Retrieve mesh struct from Fluidity
        cdef ft.mesh_t mesh = ft.read_mesh_f(basename, _file_extensions[ext], dim)
        self.name = filename

        # Create DMPlex object from Fluidity struct
        dim = mesh.topological_dimension
        coords = np.array(<double[:mesh.vertex_count, :mesh.geometric_dimension:1]>mesh.coordinates)
        cells = np.array(<int[:mesh.cell_count, :mesh.cell_vertices:1]>mesh.element_vertex_list)
        cells = np.array([ c-1 for c in cells ])  # Non-Fortran numbering...
        dmplex = PETSc.DMPlex().createFromCellList(dim, cells, coords)

        self._from_dmplex(dmplex, periodic_coords)

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

        body = cgen.Block()
        body.extend([cgen.ArrayOf(v, 3) for v in [cgen.Value("double", "v0"),
                                                 cgen.Value("double", "v1"),
                                                 cgen.Value("double", "n"),
                                                 cgen.Value("double", "x")]])
        body.append(cgen.Initializer(cgen.Value("double", "dot"), "0.0"))
        body.append(cgen.Value("int", "i"))
        body.append(cgen.For("i = 0", "i < 3", "i++",
                             cgen.Block([cgen.Assign("v0[i]", "coords[1][i] - coords[0][i]"),
                                         cgen.Assign("v1[i]", "coords[2][i] - coords[0][i]"),
                                         cgen.Assign("x[i]", "0.0")])))
        body.append(cgen.Assign("n[0]", "v0[1]*v1[2] - v0[2]*v1[1]"))
        body.append(cgen.Assign("n[1]", "v0[2]*v1[0] - v0[0]*v1[2]"))
        body.append(cgen.Assign("n[2]", "v0[0]*v1[1] - v0[1]*v1[0]"))

        body.append(cgen.For("i = 0", "i < 3", "i++",
                             cgen.Block([cgen.Line("x[0] += coords[i][0];"),
                                         cgen.Line("x[1] += coords[i][1];"),
                                         cgen.Line("x[2] += coords[i][2];")])))
        body.extend([cgen.Line("dot += (%(x)s) * n[%(i)d];" % {"x": x, "i": i})
                     for i, x in enumerate(expr.code)])
        body.append(cgen.Assign("*orientation", "dot < 0 ? 1 : 0"))

        fdecl = cgen.FunctionDeclaration(cgen.Value("void", "cell_orientations"),
                                         [cgen.Pointer(cgen.Value("int", "orientation")),
                                          cgen.Pointer(cgen.Pointer(cgen.Value("double", "coords")))])

        fn = cgen.FunctionBody(fdecl, body)
        kernel = op2.Kernel(str(fn), "cell_orientations")
        op2.par_loop(kernel, self.cell_set,
                     self._cell_orientations(op2.WRITE),
                     self._coordinate_field.dat(op2.READ, self._coordinate_field.cell_node_map()))
        self._cell_orientations._force_evaluation(read=True, write=False)

    def cells(self):
        return self._cells

    def ufl_cell(self):
        return self._ufl_cell

    def num_cells(self):
        return self._entities[-1]

    def num_facets(self):
        return self._entities[-2]

    def num_faces(self):
        return self._entities[2]

    def num_edges(self):
        return self._entities[1]

    def num_vertices(self):
        return self._entities[0]

    def num_entities(self, d):
        return self._entities[d]

    def size(self, d):
        return self._entities[d]

    @utils.cached_property
    def cell_set(self):
        size = self.num_cells()
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
        self._entities = mesh._entities
        self.parent = mesh.parent
        self.uid = mesh.uid
        self.region_ids = mesh.region_ids
        self._coordinates = mesh._coordinates
        self.name = mesh.name

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
        # Now we need to produce the extruded mesh using
        # techqniues employed when computing the
        # function space.
        cdef ft.element_t element_f = as_element(fiat_element)
        cdef void *fluidity_mesh = py.PyCapsule_GetPointer(mesh._fluidity_mesh, _mesh_name)
        if fluidity_mesh == NULL:
            raise RuntimeError("Didn't find fluidity mesh pointer in mesh %s" % mesh)

        cdef int *dofs_per_column = <int *>np.PyArray_DATA(self.dofs_per_column)

        extruded_mesh = ft.extruded_mesh_f(fluidity_mesh, &element_f, dofs_per_column)

        #Assign the newly computed extruded mesh as the fluidity mesh
        self._fluidity_mesh = py.PyCapsule_New(extruded_mesh.fluidity_mesh,
                                                         _mesh_name,
                                                         NULL)

        self._coordinate_fs = types.VectorFunctionSpace(self, mesh._coordinate_fs.ufl_element().family(),
                                                        mesh._coordinate_fs.ufl_element().degree(),
                                                        vfamily="CG",
                                                        vdegree=1)

        self._coordinate_field = types.Function(self._coordinate_fs)
        make_extruded_coords(self, layer_height, extrusion_type=extrusion_type,
                             kernel=kernel)
        self._coordinates = self._coordinate_field.dat.data_ro_with_halos

        self._dx = Measure('cell', domain_data=self._coordinate_field)
        self._ds = Measure('exterior_facet', domain_data=self._coordinate_field)
        self._dS = Measure('interior_facet', domain_data=self._coordinate_field)
        # Set the domain_data on all the default measures to this coordinate field.
        for measure in [ufl.dx, ufl.ds, ufl.dS]:
            measure._domain_data = self._coordinate_field

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
    """Fluidity Halo type"""
    # Enum corresponding to halo entity types
    _ENTITY_TYPES = {
        'vertex' : 0,
        'cell' : 1
    }

    def __init__(self, mesh_capsule, entity_type='vertex', entity_classes=None):
        """Build a Halo associated with the appropriate Fluidity mesh.

        `mesh_capsule` should be a PyCapsule containing the Fluidity
        mesh to pull the halos out of.  The two halo types are
        `vertex` and `cell`.  The appropriate mesh `entity_classes`
        must be passed in to correct set up global to universal
        numbering maps.
        """
        assert type(entity_type) is str, 'Entity type should be a string'
        entity_type = entity_type.lower()

        # Convert from char * to str.
        capsule_name = <bytes>py.PyCapsule_GetName(mesh_capsule)

        if capsule_name not in [_function_space_name, _mesh_name]:
            raise RuntimeError("Passed a capsule that didn't contain a mesh pointer")

        cdef void *fluidity_mesh = py.PyCapsule_GetPointer(mesh_capsule, capsule_name)
        cdef ft.halo_t halo = ft.halo_f(fluidity_mesh, Halo._ENTITY_TYPES[entity_type])

        if halo.nprocs == -1:
            # Build empty halo, this is probably right
            self._nprocs = 1
            self._sends = {}
            self._receives = {}
            self._entity_type = entity_type
            self._nowned_nodes = entity_classes[1]
            self._fluidity_halo = None
            self._universal_offset = 0
            self._comm = None
            self._global_to_universal_number = None
            return
        self._nprocs = halo.nprocs
        self._entity_type = entity_type

        self._sends = {}
        self._receives = {}

        cdef int i
        # These come in with Fortran numbering, but we want C
        # numbering, so fix them up on creation
        for i in range(halo.nprocs):
            dim = halo.nsends[i]
            if dim > 0:
                self._sends[i] = np.array(<int[:dim]>halo.sends[i]) - 1
            dim = halo.nreceives[i]
            if dim > 0:
                self._receives[i] = np.array(<int[:dim]>halo.receives[i]) - 1
        # These were allocated in Python_Interface_f.F90, but only used to
        # pass size information, and we don't have a destructor, so free
        # them here.
        free(halo.nsends)
        free(halo.nreceives)
        self._nowned_nodes = halo.nowned_nodes
        # No destructor, since Fluidity owns a reference.
        self._fluidity_halo = py.PyCapsule_New(halo.fluidity_halo, _halo_name, NULL)
        self._universal_offset = halo.universal_offset
        self._comm = MPI.Comm.f2py(halo.comm)
        assert self._comm.size == self._nprocs, \
            "Communicator size does not match specified number of processes for halo"

        assert self._comm.rank not in self._sends.keys(), "Halos should contain no self-sends"
        assert self._comm.rank not in self._receives.keys(), "Halos should contain no self-receives"

        # Fluidity gives us the global to universal mapping for dofs
        # in the halo, but for matrix assembly we need to mapping for
        # all dofs.  Fortunately, for owned dofs we just need to add
        # the appropriate offset
        tmp = np.arange(0, entity_classes[3], dtype=np.int32)
        dim = halo.receives_global_to_universal_len
        if dim > 0:
            # Can't make a memoryview of a zero-length array hence the
            # check
            recv_map = np.array(<int[:dim]>halo.receives_global_to_universal)
            tmp[entity_classes[1]:entity_classes[2]] = recv_map[tmp[entity_classes[1]:entity_classes[2]] \
                                                                - entity_classes[1]] - 1
        tmp[:entity_classes[1]] = tmp[:entity_classes[1]] + self._universal_offset
        tmp[entity_classes[2]:] = -1
        self._global_to_universal_number = tmp

    @utils.cached_property
    def op2_halo(self):
        if not self.sends and not self.receives:
            return None
        return op2.Halo(self.sends, self.receives,
                        comm=self.comm, gnn2unn=self.global_to_universal_number)

    @property
    def comm(self):
        return self._comm

    @property
    def nprocs(self):
        return self._nprocs

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def sends(self):
        return self._sends

    @property
    def receives(self):
        return self._receives

    @property
    def nowned_nodes(self):
        return self._nowned_nodes

    @property
    def universal_offset(self):
        return self._universal_offset

    @property
    def global_to_universal_number(self):
        return self._global_to_universal_number


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
            flattened_element = self.fiat_element.flattened_element()

            # Compute the dofs per column
            self.dofs_per_column = compute_extruded_dofs(self.fiat_element,
                                                         flattened_element.entity_dofs(),
                                                         mesh._layers)

            # Compute the offset for the extrusion process
            self.offset = compute_offset(self.fiat_element.entity_dofs(),
                                         flattened_element.entity_dofs(),
                                         self.fiat_element.space_dimension())

            # Compute the top and bottom masks to identify boundary dofs
            b_mask = self.fiat_element.get_lower_mask()
            t_mask = self.fiat_element.get_upper_mask()

            self.bt_masks = (b_mask, t_mask)

            self.extruded = True
        else:
            # If not extruded specific, set things to None/False, etc.
            self.offset = None
            self.bt_masks = None
            self.dofs_per_column = np.zeros(1, np.int32)
            self.extruded = False

        self.name = name
        self._dim = dim
        self._mesh = mesh
        self._index = None

        # TODO: DoF classes need to be derived properly
        self.dof_classes = mesh.vertex_classes

        # TODO: Derive Halo from DMPlex
        self._halo = None

        self._node_count = mesh.vertex_count
        cStart,cEnd = mesh._plex.getHeightStratum(0)  # cells
        self.cell_node_list = np.array([self._get_cell_nodes(c) for c in range(cStart, cEnd)])

        self.interior_facet_node_list = None
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
        vStart,vEnd = self._mesh._plex.getDepthStratum(0)   # vertices
        closure = self._mesh._plex.getTransitiveClosure(cell)[0]
        closure_vertices = filter(lambda x: x>=vStart and x<vEnd, closure)
        return map(lambda x: x-vStart, closure_vertices)

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

        return self._map_cache(self._interior_facet_map_cache,
                               self._mesh.interior_facets.set,
                               self.interior_facet_node_list,
                               2*self.fiat_element.space_dimension(),
                               bcs,
                               "interior_facet_node",
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
        c_array = lambda xs : "{"+", ".join(map(str, xs))+"}"

        kernel = op2.Kernel(str(cgen.FunctionBody(
                    cgen.FunctionDeclaration(
                        cgen.Value("void", "create_bc_node_map"),
                        [cgen.Value("int", "*cell_nodes"),
                         cgen.Value("int", "*facet_nodes"),
                         cgen.Value("unsigned int", "*facet")
                         ]
                        ),
                    cgen.Block(
                        [cgen.ArrayInitializer(
                                cgen.Const(
                                    cgen.ArrayOf(
                                        cgen.ArrayOf(
                                            cgen.Value("int", "l_nodes"),
                                            str(len(el.get_reference_element().topology[dim]))),
                                        str(nodes_per_facet)),
                                    ),
                                map(c_array, local_facet_nodes)
                                ),
                         cgen.Value("int", "n"),
                         cgen.For("n=0", "n < %d" % nodes_per_facet, "++n",
                                  cgen.Assign("facet_nodes[n]",
                                              "cell_nodes[l_nodes[facet[0]][n]]")
                                  )
                         ]
                        )
                    )), "create_bc_node_map")

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

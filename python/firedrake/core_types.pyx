# cython: embedsignature=True
# Re-implementation of state_types for firedrake.
import os
import ufl
from ufl import *
from mpi4py import MPI
import FIAT
import numpy as np
import utils
import pyop2 as op2
from pyop2.utils import flatten
import assemble_expressions
from vector import Vector
import cgen

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
# Necessary so that we can use memoryviews
from cython cimport view
cimport numpy as np
cimport cpython as py
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
    3 : { 4 : "tetrahedron"}
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
    """Cause op2.init() to be called in case the user has not done it
    for themselves. The result of this is that the user need only call
    op2.init if (s)he wants to set a non-default option, for example
    to switch the backend or the debug or log level."""
    if not op2.initialised():
        op2.init()

def fiat_from_ufl_element(ufl_element):
    if isinstance(ufl_element, ufl.OuterProductElement):
        a = FIAT.supported_elements[ufl_element._A.family()]\
            (_FIAT_cells[ufl_element._A.cell().cellname()](), ufl_element._A.degree())

        b = FIAT.supported_elements[ufl_element._B.family()]\
            (_FIAT_cells[ufl_element._B.cell().cellname()](), ufl_element._B.degree())

        return FIAT.TensorFiniteElement(a, b)
    else:
        return FIAT.supported_elements[ufl_element.family()]\
            (_FIAT_cells[ufl_element.cell().cellname()](), ufl_element.degree())

# Functions related to the extruded case
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

def make_extruded_coords(mesh, layers, kernel=None, layer_height=None):
    """Given a kernel or height between layers, use it to generate the
    extruded coordinates.

    :arg mesh: the 2d mesh to extrude
    :arg layers: the number of layers in the extruded mesh
    :arg kernel: :class:`pyop2.Kernel` which produces the extruded coordinates
    :arg layer_height: if provided it creates coordinates for evenly
                       spaced layers

    Either the kernel or the layer_height must be provided. Should
    both be provided then the kernel takes precendence.
    Its calling signature is:

    .. c::

        void extrusion_kernel(double *extruded_coords[],
                              double *two_d_coords[],
                              int *layer_number[])

    So for example to build an evenly-spaced extruded mesh with eleven
    layers and height 1 in the vertical you would write:

    .. c::

       void extrusion_kernel(double *extruded_coords[],
                             double *two_d_coords[],
                             int *layer_number[]) {
           extruded_coords[0][0] = two_d_coords[0][0]; // X
           extruded_coords[0][1] = two_d_coords[0][1]; // Y
           extruded_coords[0][2] = 0.1 * layer_number[0][0]; // Z
       }
    """

    if kernel is None and layer_height is not None:
        kernel = op2.Kernel("""
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Only the Z-coord is increased, the others stay the same
    xtr[0][0] = x[0][0];
    xtr[0][1] = x[0][1];
    xtr[0][2] = %(height)s*j[0][0];
}""" % {"height" : str(layer_height)} , "extrusion_kernel")

    coords_dim = len(mesh._coordinates[0])
    coords_xtr_dim = 3 #dimension
    # BIG TRICK HERE:
    # We need the +1 in order to include the entire column of vertices.
    # Extrusion is meant to iterate over the 3D cells which are layer - 1 in number.
    # The +1 correction helps in the case of iteration over vertices which need
    # one extra layer.
    iterset = op2.Set(mesh.num_vertices(), "verts1", layers=(layers+1))
    vnodes = op2.DataSet(iterset, coords_dim)
    lnodes = op2.DataSet(iterset, 1)
    nodes_xtr = op2.Set(mesh.num_vertices()*layers, "verts_xtr")
    d_nodes_xtr = op2.DataSet(nodes_xtr, coords_xtr_dim)
    d_lnodes_xtr = op2.DataSet(nodes_xtr, 1)

    # Create an op2.Dat with the base mesh coordinates
    coords_vec = mesh._coordinates.flatten()
    coords = op2.Dat(vnodes, coords_vec, np.float64, "dat1")

    # Create an op2.Dat with slots for the extruded coordinates
    coords_new = np.empty(layers * mesh.num_vertices() * coords_xtr_dim, dtype=np.float64)
    coords_xtr = op2.Dat(d_nodes_xtr, coords_new, np.float64, "dat_xtr")

    # Creat an op2.Dat to hold the layer number
    layer_vec = np.tile(np.arange(0, layers), mesh.num_vertices())
    layer = op2.Dat(d_lnodes_xtr, layer_vec, np.int32, "dat_layer")

    # Map a map for the bottom of the mesh.
    vertex_to_coords = [ i for i in range(0, mesh.num_vertices()) ]
    v2coords_offset = np.zeros(1, np.int32)
    map_2d = op2.Map(iterset, iterset, 1, vertex_to_coords, "v2coords", v2coords_offset)

    # Create Map for extruded vertices
    vertex_to_xtr_coords = [ layers * i for i in range(0, mesh.num_vertices()) ]
    v2xtr_coords_offset = np.array([1], np.int32)
    map_xtr = op2.Map(iterset, nodes_xtr, 1, vertex_to_xtr_coords, "v2xtr_coords", v2xtr_coords_offset)

    # Create Map for layer number
    v2xtr_layer_offset = np.array([1], np.int32)
    layer_xtr = op2.Map(iterset, nodes_xtr, 1, vertex_to_xtr_coords, "v2xtr_layer", v2xtr_layer_offset)

    op2.par_loop(kernel, iterset,
                 coords_xtr(op2.INC, map_xtr),
                 coords(op2.READ, map_2d),
                 layer(op2.READ, layer_xtr))

    return coords_xtr.data

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
    if isinstance(fiat_element, FIAT.tensor_finite_element.TensorFiniteElement):
        # Use the reference element of the horizontal element.
        f_element = fiat_element.flattened_element()
    else:
        f_element = fiat_element

    element.dimension = len(f_element.ref_el.topology) - 1
    element.vertices = len(f_element.ref_el.vertices)
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
    """Wrapper class for facet interation information on a Mesh"""
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

        if measure.domain_id() == measure.DOMAIN_ID_EVERYWHERE:
            return self.set
        else:
            return self.subset(measure.domain_id().subdomain_ids()[0])

    def subset(self, i):
        """Return the subset corresponding to a marker value of i."""

        if self.markers is not None and not self._subsets:
            # Generate the subsets. One subset is created for each unique marker value.
            self._subsets = dict(((i,op2.Subset(self.set, np.nonzero(self.markers==i)[0]))
                                  for i in np.unique(self.markers)))
        try:
            return self._subsets[i]
        except KeyError:
            return self._null_subset


    @utils.cached_property
    def local_facet_dat(self):
        """Dat indicating which local facet of each adjacent
        cell corresponds to the current facet."""

        return op2.Dat(op2.DataSet(self.set, self._rank), self.local_facet_number,
                       np.uintc, "%s_%s_local_facet_number" % (self.mesh.name, self.kind))

class Mesh(object):
    """Note that this is the mesh topology and geometry,
    it is NOT a FunctionSpace."""
    def __init__(self, *args):

        _init()

        self._layers = 1

        self.cell_halo = None
        self.vertex_halo = None
        self.parent = None

        if len(args)==0:
            return

        if isinstance(args[0], str):
            self._from_file(args[0])

        else:
            raise NotImplementedError(
                "Unknown argument types for Mesh constructor")

    def _from_file(self, filename):
        """Read a mesh from `filename`

        The extension of the filename determines the mesh type."""
        basename, ext = os.path.splitext(filename)

        # Retrieve mesh struct from Fluidity
        cdef ft.mesh_t mesh = ft.read_mesh_f(basename, _file_extensions[ext])

        self.name = filename

        self._cells = np.array(<int[:mesh.cell_count, :mesh.cell_vertices:1]>mesh.element_vertex_list)
        self._ufl_cell = ufl.Cell(
            _cells[mesh.geometric_dimension][mesh.cell_vertices],
            mesh.topological_dimension)

        self._entities = np.zeros(mesh.topological_dimension + 1, dtype=np.int)
        self._entities[:] = -1 # Ensure that 3d edges get an out of band value.
        self._entities[0]  = mesh.vertex_count
        self._entities[-1] = mesh.cell_count
        self._entities[-2] = mesh.interior_facet_count + mesh.exterior_facet_count
        self.uid = mesh.uid

        if mesh.interior_facet_count > 0:
            interior_facet_cell = \
                np.array(<int[:mesh.interior_facet_count, :2]>mesh.interior_facet_cell)
            interior_local_facet_number = \
                np.array(<int[:mesh.interior_facet_count, :2]>mesh.interior_local_facet_number)
            self.interior_facets = _Facets(self, mesh.interior_facet_count,
                                           "interior",
                                           interior_facet_cell,
                                           interior_local_facet_number)
        else:
            self.interior_facets = _Facets(self, 0, "interior", None, None)

        if mesh.exterior_facet_count > 0:
            if mesh.boundary_ids != NULL:
                boundary_ids = np.array(<int[:mesh.exterior_facet_count]>mesh.boundary_ids)
            else:
                boundary_ids = None

            exterior_facet_cell = \
                np.array(<int[:mesh.exterior_facet_count, :1]>mesh.exterior_facet_cell)
            exterior_local_facet_number = \
                np.array(<int[:mesh.exterior_facet_count, :1]>mesh.exterior_local_facet_number)
            self.exterior_facets = _Facets(self, mesh.exterior_facet_count,
                                           "exterior",
                                           exterior_facet_cell,
                                           exterior_local_facet_number,
                                           boundary_ids)
        else:
            self.exterior_facets = _Facets(self, 0, "exterior", None, None)

        if mesh.region_ids != NULL:
            self.region_ids = np.array(<int[:mesh.cell_count]>mesh.region_ids)
        else:
            self.region_ids = None

        # Build these from the Fluidity data, then need to convert
        # them from np.int32 to python int type which is what PyOP2 expects
        self.cell_classes = np.array(<int[:4]>mesh.cell_classes)
        self.cell_classes = self.cell_classes.astype(int)
        self.vertex_classes = np.array(<int[:4]>mesh.vertex_classes)
        self.vertex_classes = self.vertex_classes.astype(int)

        self._coordinates = np.array(<double[:mesh.vertex_count, :mesh.geometric_dimension:1]>
                                     mesh.coordinates)

        self._fluidity_coordinate = py.PyCapsule_New(mesh.fluidity_coordinate,
                                                     _vector_field_name,
                                                     &vector_field_destructor)
        # No destructor for the mesh, because we don't have a separate
        # Fluidity reference for it.
        self._fluidity_mesh = py.PyCapsule_New(mesh.fluidity_mesh,
                                               _mesh_name,
                                               NULL)

        self.cell_halo = Halo(self._fluidity_mesh, 'cell', self.cell_classes)
        self.vertex_halo = Halo(self._fluidity_mesh, 'vertex', self.vertex_classes)
        # Note that for bendy elements, this needs to change.
        self._coordinate_fs = VectorFunctionSpace(self, "Lagrange", 1)

        self._coordinate_field = Function(self._coordinate_fs,
                                          val = self._coordinates)

        # Set the domain_data on all the default measures to this coordinate field.
        for measure in [ufl.dx, ufl.ds, ufl.dS]:
            measure._domain_data = self._coordinate_field

    @property
    def layers(self):
        return self._layers

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
        if self.cell_halo:
            size = self.cell_classes
            halo = self.cell_halo.op2_halo
        else:
            size = self.num_cells()
            halo = None
        return self.parent.cell_set if self.parent else \
            op2.Set(size, "%s_cells" % self.name, halo=halo)

    def compute_boundaries(self):
        '''Currently a no-op for flop.py compatibility.'''
        pass

class ExtrudedMesh(Mesh):
    """Build an extruded mesh from a 2D input mesh

    :arg mesh:         2D unstructured mesh
    :arg layers:       number of structured layers in the "vertical"
                       direction
    :arg kernel:       pyop2 Kernel to produce 3D coordinates for the extruded
                       mesh see :func:`make_extruded_coords` for more details.
    :arg layer_height: the height between two layers when all layers are
                       evenly spaced."""
    def __init__(self, mesh, layers, kernel=None, layer_height=None):
        if kernel is None and layer_height is None:
            raise RuntimeError("Please provide a kernel or a fixed layer height")
        self._old_mesh = mesh
        self._layers = layers
        self._cells = mesh._cells
        self.cell_halo = mesh.cell_halo
        self._entities = mesh._entities
        self.parent = mesh.parent
        self.uid = mesh.uid
        self.region_ids = mesh.region_ids
        self.cell_classes = mesh.cell_classes
        self._coordinates = mesh._coordinates
        self.name = mesh.name

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
        self._coordinates = make_extruded_coords(mesh, layers, kernel, layer_height)

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

        self._coordinate_fs = VectorFunctionSpace(self, "Lagrange", 1)

        self._coordinate_field = Function(self._coordinate_fs,
                                          val = self._coordinates)


        # Set the domain_data on all the default measures to this coordinate field.
        for measure in [ufl.dx, ufl.ds, ufl.dS]:
            measure._domain_data = self._coordinate_field

    @utils.cached_property
    def cell_set(self):
        if self.cell_halo:
            size = self.cell_classes
            halo = self.cell_halo.op2_halo
        else:
            size = self.num_cells()
            halo = None
        return self.parent.cell_set if self.parent else \
            op2.Set(size, "%s_elements" % self.name, halo=halo, layers=self._layers)

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


class FunctionSpace(object):
    """Create a function space

    :arg mesh: mesh to build the function space on
    :arg family: string describing function space family, or a
        :class:`ufl.OuterProductElement`
    :arg degree: degree of the function space
    :arg name: (optional) name of the function space
    :arg vfamily: family of function space in vertical dimension
        (:class:`ExtrudedMesh`\es only)
    :arg vdegree: degree of function space in vertical dimension
        (:class:`ExtrudedMesh`\es only)

    If the mesh is an :class:`ExtrudedMesh`, and the `family` argument
    is a :class:`ufl.OuterProductElement`, `degree`, `vfamily` and
    `vdegree` are ignored, since the `family` provides all necessary
    information, otherwise a :class:`ufl.OuterProductElement` is built
    from the (`family`, `degree`) and (`vfamily`, `vdegree`) pair.  If
    the `vfamily` and `vdegree` are not provided, the vertical element
    will be the same as the provided (`family`, `degree`) pair.

    If the mesh is not an :class:`ExtrudedMesh`, the `family` must be
    a string describing the finite element family to use, and the
    `degree` must be provided, `vfamily` and `vdegree` are ignored in
    this case.
    """

    def __init__(self, mesh, family, degree=None, name=None, vfamily=None, vdegree=None):
        if isinstance(mesh, ExtrudedMesh):
            # The function space must adapt to the extrusion
            # accordingly.
            # The bottom layer maps  will come from lement_dof_list
            # dof_count is the total number of dofs in the extruded mesh

            if isinstance(family, ufl.OuterProductElement):

                a = family._A
                b = family._B

                la = ufl.FiniteElement(a.family(),
                                       domain=mesh._old_mesh._ufl_cell,
                                       degree=a.degree())

                lb = ufl.FiniteElement(b.family(),
                                       domain=ufl.Cell("interval",1),
                                       degree=b.degree())

            else:
                la = ufl.FiniteElement(family,
                                       domain=mesh._old_mesh._ufl_cell,
                                       degree=degree)
                # FIAT version of the extrusion
                if vfamily is None or vdegree is None:
                    lb = ufl.FiniteElement(family,
                                           domain=ufl.Cell("interval",1),
                                           degree=degree)
                else:
                    lb = ufl.FiniteElement(vfamily,
                                           domain=ufl.Cell("interval",1),
                                           degree=vdegree)

            # Create the Function Space element
            self._ufl_element = ufl.OuterProductElement(la, lb)

            # Compute the FIAT version of the UFL element above
            self.fiat_element = fiat_from_ufl_element(self._ufl_element)

            # Get the flattened version of the 3D FIAT element
            flat_temp = self.fiat_element.flattened_element()

            # Compute the dofs per column
            self.dofs_per_column = compute_extruded_dofs(self.fiat_element,
                                                         flat_temp.entity_dofs(),
                                                         mesh._layers)

            # Compute the offset for the extrusion process
            self.offset = compute_offset(self.fiat_element.entity_dofs(),
                                         flat_temp.entity_dofs(),
                                         self.fiat_element.space_dimension())
        else:
            if isinstance(family, ufl.OuterProductElement):
                raise RuntimeError("You can't build an extruded element on an unextruded Mesh")
            if degree is None:
                raise RuntimeError("The function space requires a degree")

            self.offset = None
            self.dofs_per_column = np.zeros(1, np.int32)
            self.extruded = False

            self._ufl_element = ufl.FiniteElement(family,
                                                  domain=mesh._ufl_cell,
                                                  degree=degree)

            self.fiat_element = fiat_from_ufl_element(self._ufl_element)

        # Create the extruded function space
        cdef ft.element_t element_f = as_element(self.fiat_element)

        cdef void *fluidity_mesh = py.PyCapsule_GetPointer(mesh._fluidity_mesh, _mesh_name)
        if fluidity_mesh == NULL:
            raise RuntimeError("Didn't find fluidity mesh pointer in mesh %s" % mesh)

        cdef int *dofs_per_column = <int *>np.PyArray_DATA(self.dofs_per_column)

        if isinstance(mesh, ExtrudedMesh):
            function_space = ft.extruded_mesh_f(fluidity_mesh, &element_f, dofs_per_column)
        else:
            function_space = ft.function_space_f(fluidity_mesh, &element_f)

        free(element_f.dofs_per)
        free(element_f.entity_dofs)

        self._fluidity_function_space = py.PyCapsule_New(function_space.fluidity_mesh,
                                                         _function_space_name,
                                                         &function_space_destructor)

        self._node_count = function_space.dof_count
        self.cell_node_list = np.array(<int[:function_space.element_count, :element_f.ndof:1]>
                                      function_space.element_dof_list)-1

        self.dof_classes = np.array(<int[:4]>function_space.dof_classes)
        self.dof_classes = self.dof_classes.astype(int)
        self._mesh = mesh

        self._halo = Halo(self._fluidity_function_space, 'vertex', self.dof_classes)

        self.name = name

        self._dim = 1

        if not isinstance(self._mesh, ExtrudedMesh):
            if self._mesh.interior_facets.count > 0:
                self.interior_facet_node_list = \
                    np.array(<int[:self._mesh.interior_facets.count,:2*element_f.ndof]>
                             function_space.interior_facet_node_list)
            else:
                self.interior_facet_node_list = None

            self.exterior_facet_node_list = \
                np.array(<int[:self._mesh.exterior_facets.count,:element_f.ndof]>
                         function_space.exterior_facet_node_list)

        # Note that this is the function space rank and is therefore
        # always 0. The value rank may be different.
        self.rank = 0

        # Empty map caches. This is a sui generis cache
        # implementation because of the need to support boundary
        # conditions.
        self._cell_node_map_cache = {}
        self._exterior_facet_map_cache = {}
        self._interior_facet_map_cache = {}

    @property
    def node_count(self):
        """The number of global nodes in the function space. For a
        plain :class:`FunctionSpace` this is equal to
        :attr:`dof_count`, however for a :class:`VectorFunctionSpace`,
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
        :class:`FunctionSpace`. One or (for VectorFunctionSpaces) more degrees
        of freedom are stored at each node."""

        name = "%s_nodes" % self.name
        if self._halo:
            return op2.Set(self.dof_classes, name,
                           halo=self._halo.op2_halo, layers=self._mesh.layers)
        else:
            return op2.Set(self.node_count, name, layers=self._mesh.layers)

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.Set` containing the degrees of freedom of
        this :class:`FunctionSpace`."""
        return op2.DataSet(self.node_set, self.dim)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.Dat` defined on the
        :attr:`dof.dset` of this :class:`Function`."""

        return op2.Dat(self.dof_dset, val, valuetype, name, uid=uid)


    def cell_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`DirichletBC`\s. In this case, the facet_node_map will return
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
        :class:`DirichletBC`\s. In this case, the facet_node_map will return
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
        :class:`DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""

        if bcs:
            parent = self.exterior_facet_node_map()
        else:
            parent = None

        return self._map_cache(self._exterior_facet_map_cache,
                               self._mesh.exterior_facets.set,
                               self.exterior_facet_node_list,
                               self.fiat_element.space_dimension(),
                               bcs,
                               "exterior_facet_node",
                               parent=parent)

    def _map_cache(self, cache, entity_set, entity_node_list, map_arity, bcs, name,
                   offset=None, parent=None):
        if bcs is None:
            lbcs = None
        else:
            # Ensure bcs is a tuple in a canonical order for the hash key.
            lbcs = tuple(sorted(bcs, key=lambda bc: bc.__hash__()))

        try:
            # Cache hit
            return cache[lbcs]
        except KeyError:
            # Cache miss.
            if not lbcs:
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
                                  parent)

            return cache[lbcs]

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :method:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''

        el = self.fiat_element
        dim = len(el.ref_el.topology)-1
        nodes_per_facet = \
            len(self.fiat_element.entity_closure_dofs()[dim-1][0])

        facet_set = self._mesh.exterior_facets.set

        fs_dat = op2.Dat(facet_set**el.space_dimension(),
                         data=self.exterior_facet_node_map().values_with_halo)

        facet_dat = op2.Dat(facet_set**nodes_per_facet,
                            dtype=np.int32)

        local_facet_nodes = np.array(
            [dofs for e, dofs in el.entity_closure_dofs()[dim-1].iteritems()])

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
                                            str(len(el.ref_el.topology[dim-1]))),
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

        op2.par_loop(kernel, facet_set,
                     fs_dat(op2.READ),
                     facet_dat(op2.WRITE),
                     self._mesh.exterior_facets.local_facet_dat(op2.READ))

        return op2.Map(facet_set, self.node_set, nodes_per_facet, 
                       facet_dat.data_ro_with_halos, name="exterior_facet_boundary_node")


    @property
    def dim(self):
        """The vector dimension of the :class:`FunctionSpace`. For a
        :class:`FunctionSpace` this is always one. For a
        :class:`VectorFunctionSpace` it is the value given to the
        constructor, and defaults to the geometric dimension of the :class:`Mesh`. """
        return self._dim

    def ufl_element(self):
        """The :class:`ufl.FiniteElement` used to construct this
        :class:`FunctionSpace`."""
        return self._ufl_element

    def mesh(self):
        """The :class:`Mesh` used to construct this :class:`FunctionSpace`."""
        return self._mesh

    def __iter__(self):
        yield self

    def __mul__(self, other):
        """Create a :class:`MixedFunctionSpace` composed of this
        :class:`FunctionSpace` and other"""
        return MixedFunctionSpace((self, other))


class VectorFunctionSpace(FunctionSpace):
    def __init__(self, mesh, family, degree, dim=None, name=None, vfamily=None, vdegree=None):
        super(VectorFunctionSpace, self).__init__(mesh, family, degree, name, vfamily=vfamily, vdegree=vdegree)
        self.rank = 1
        if dim is None:
            # VectorFunctionSpace dimension defaults to the geometric dimension of the mesh.
            self._dim = self._mesh.ufl_cell().geometric_dimension()
        else:
            self._dim = dim

        if isinstance(mesh, ExtrudedMesh):
            if isinstance(family, ufl.OuterProductElement):
                raise NotImplementedError("Not yet implemented")
            la = ufl.FiniteElement(family,
                                   domain=mesh._old_mesh._ufl_cell,
                                   degree=degree)
            if vfamily is None or vdegree is None:
                lb = ufl.FiniteElement(family,
                                       domain=ufl.Cell("interval", 1),
                                       degree=degree)
            else:
                lb = ufl.FiniteElement(vfamily,
                                       domain=ufl.Cell("interval", 1),
                                       degree=vdegree)
            self._ufl_element = ufl.OuterProductVectorElement(la, lb)
        else:
            self._ufl_element = VectorElement(family,
                                              domain=mesh._ufl_cell,
                                              degree=degree,
                                              dim=self._dim)


class MixedFunctionSpace(FunctionSpace):
    """A mixed finite element :class:`FunctionSpace`."""

    def __init__(self, spaces, name=None):
        """
        :param spaces: a list (or tuple) of :class:`FunctionSpace`s

        The function space may be created as ::

            V = MixedFunctionSpace(spaces)

        ``spaces`` may consist of multiple occurances of the same space: ::

            P1  = FunctionSpace(mesh, "CG", 1)
            P2v = VectorFunctionSpace(mesh, "Lagrange", 2)

            ME  = MixedFunctionSpace([P2v, P1, P1, P1])
        """

        self._spaces = list(flatten(spaces))
        self._mesh = self._spaces[0]._mesh
        self._ufl_element = ufl.MixedElement(*[fs.ufl_element() for fs in self._spaces])
        self.name = '_'.join(str(s.name) for s in self._spaces)
        self.rank = 1

    def split(self):
        """A list of :class:`FunctionSpace`\s this :class:`MixedFunctionSpace`
        is composed of"""
        return self._spaces

    def sub(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self[i]

    def num_sub_spaces(self):
        """Return the number of :class:`FunctionSpace`\s this
        :class:`MixedFunctionSpace` is composed of."""
        return len(self)

    def __len__(self):
        """Return the number of :class:`FunctionSpace`\s this
        :class:`MixedFunctionSpace` is composed of."""
        return len(self._spaces)

    def __getitem__(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self._spaces[i]

    def __iter__(self):
        for s in self._spaces:
            yield s

    @property
    def dim(self):
        """Return a tuple of the :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return tuple(fs.dim for fs in self._spaces)

    @property
    def node_count(self):
        """Return a tuple of the :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return tuple(fs.node_count for fs in self._spaces)

    @property
    def dof_count(self):
        """Return a tuple of the :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return tuple(fs.dof_count for fs in self._spaces)

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.MixedSet` containing the nodes of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.node_set`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of One or (for VectorFunctionSpaces) more degrees of freedom
        are stored at each node."""
        return op2.MixedSet(s.node_set for s in self._spaces)

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.MixedSet` containing the degrees of freedom of
        this :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.dof_dset`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is composed
        of."""
        return op2.MixedDataSet(s.dof_dset for s in self._spaces)

    def cell_node_map(self, bcs=None):
        """A :class:`pyop2.MixedMap` from the :attr:`Mesh.cell_set` of the
        underlying mesh to the :attr:`node_set` of this
        :class:MixedFunctionSpace. This is composed of the
        :attr:`FunctionSpace.cell_node_map`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is composed
        of."""
        # FIXME: these want caching of sorts
        return op2.MixedMap(s.cell_node_map(bcs) for s in self._spaces)

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.MixedMap` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        return op2.MixedMap(s.interior_facet_node_map(bcs) for s in self._spaces)

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        return op2.MixedMap(s.exterior_facet_node_map(bcs) for s in self._spaces)

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.MixedMap` from exterior facets to the nodes on
        those facets. Note that this differs from
        :method:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return op2.MixedMap(s.exterior_facet_boundary_node_map for s in self._spaces)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.MixedDat` defined on the
        :attr:`dof.dset` of this :class:`MixedFunctionSpace`."""
        if val is not None:
            assert len(val) == len(self)
        else:
            val = [None for _ in self]
        return op2.MixedDat(s.make_dat(v, valuetype, name, _new_uid())
                            for s, v in zip(self._spaces, val))


class Function(ufl.Coefficient):
    """A :class:`Function` represents a discretised field over the
    domain defined by the underlying :class:`Mesh`. Functions are
    represented as sums of basis functions:

    .. math::

            f = \\sum_i f_i \phi_i(x)

    The :class:`Function` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`FunctionSpace` object
    which provides the basis functions :math:`\\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Function` is vector-valued then this is specified in
    the :class:`FunctionSpace`.
    """

    def __init__(self, function_space, val = None, name = None):

        if isinstance(function_space, Function):
            self._function_space = function_space._function_space
        elif isinstance(function_space, FunctionSpace):
            self._function_space = function_space
        else:
            raise NotImplementedError("Can't make a Function defined on a "
                                      +str(type(function_space)))

        ufl.Coefficient.__init__(self, self._function_space.ufl_element())

        self._label = "a function"
        self.uid = _new_uid()
        self._name = name or 'function_%d' % self.uid

        self.dat = self._function_space.make_dat(val, valuetype, self._name,
                                                 uid=self.uid)

        self._repr = None

        if isinstance(function_space, Function):
          self.assign(function_space)

    @property
    def cell_set(self):
        """The :class:`pyop2.Set` of cells for the mesh on which this
        :class:`Function` is defined."""
        return self._function_space._mesh.cell_set

    @property
    def node_set(self):
        return self._function_space.node_set

    @property
    def dof_dset(self):
        return self._function_space.dof_dset

    def cell_node_map(self, bcs=None):
        return self._function_space.cell_node_map(bcs)

    def interior_facet_node_map(self, bcs=None):
        return self._function_space.interior_facet_node_map(bcs)

    def exterior_facet_node_map(self, bcs=None):
        return self._function_space.exterior_facet_node_map(bcs)

    def vector(self):
        """Return a :class:`Vector` wrapping the data in this :class:`Function`"""
        return Vector(self.dat)

    def function_space(self):
        return self._function_space

    def name(self):
        """Return the name of this :class:`Function`"""
        return self._name

    def label(self):
        """Return the label (a description) of this :class:`Function`"""
        return self._label

    def rename(self, name=None, label=None):
        """Set the name and or label of this :class:`Function`

        :arg name: The new name of the `Function` (if not `None`)
        :arg label: The new label for the `Function` (if not `None`)
        """
        if name is not None:
            self._name = name
        if label is not None:
            self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Function, self).__str__()

    def interpolate(self, expression):
        """Interpolate expression onto the :class:`Function`."""

        coords = self.function_space().mesh()._coordinate_field

        coords_space = coords.function_space()

        coords_element = coords_space.fiat_element

        to_element = self.function_space().fiat_element

        to_pts = []

        if expression.rank() != self.function_space().rank:
            raise RuntimeError('Rank mismatch between Expression and FunctionSpace')

        for dual in to_element.dual_basis():
            if not isinstance(dual, FIAT.functional.PointEvaluation):
                raise NotImplementedError("Can only interpolate onto point evaluation operators. Try projecting instead")

            to_pts.append(dual.pt_dict.keys()[0])

        X=coords_element.tabulate(0, to_pts).values()[0]

        # Produce C array notation of X.
        X_str = "{{"+"},\n{".join([ ",".join(map(str,x)) for x in X.T])+"}}"

        assign_expression = ";\n".join(["A[%(i)d] = %(code)s" % { 'i': i, 'code': code } for i, code in enumerate(expression.code)])
        _expression_template = """
void expression_kernel(double A[%(rank)d], double **x_, int k)
{
  const double X[%(ndof)d][%(xndof)d] = %(x_array)s;

  double x[%(dim)d];
  const double pi = 3.141592653589793;

  for (unsigned int d=0; d < %(dim)d; d++) {
    x[d] = 0;
    for (unsigned int i=0; i < %(xndof)d; i++) {
      x[d] += X[k][i] * x_[i][d];
    };
  };

  %(assign_expression)s;
}
"""
        kernel = op2.Kernel(_expression_template % { "x_array" : X_str,
                                                     "dim" : coords_space.dim,
                                                     "xndof" : coords_element.space_dimension(),
                                                     "ndof" : to_element.space_dimension(),
                                                     "assign_expression" : assign_expression,
                                                     "rank" : expression.rank() },
                            "expression_kernel")

        op2.par_loop(kernel, self.cell_set,
                     self.dat(op2.WRITE, self.cell_node_map()[op2.i[0]]),
                     coords.dat(op2.READ, coords.cell_node_map())
                     )


    def assign(self, expr, subset=None):
        """Set the :class:`Function` value to the pointwise value of
        expr. expr may only contain Functions on the same
        :class:`FunctionSpace` as the :class:`Function` being assigned to.

        Similar functionality is available for the augmented assignment
        operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
        both Functions on the same :class:`FunctionSpace` then::

          f += 2 * g

        will add twice `g` to `f`.

        If present, subset must be an :class:`pyop2.Subset` of
        :attr:`self.node_set`. The expression will then only be assigned
        to the nodes on that subset.
        """

        assemble_expressions.evaluate_expression(
            assemble_expressions.Assign(self, expr), subset)

        return self

    def __iadd__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.IAdd(self, expr))

        return self

    def __isub__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.ISub(self, expr))

        return self

    def __imul__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.IMul(self, expr))

        return self

    def __idiv__(self, expr):

        assemble_expressions.evaluate_expression(
            assemble_expressions.IDiv(self, expr))

        return self


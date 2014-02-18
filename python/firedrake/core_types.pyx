# cython: embedsignature=True
# Re-implementation of state_types for firedrake.
import os
from mpi4py import MPI
import numpy as np
import cgen

from libc.stdlib cimport malloc, free
# Necessary so that we can use memoryviews
from cython cimport view
cimport numpy as np
cimport cpython as py

import ufl
from ufl import *
import FIAT

from pyop2 import op2
from pyop2.utils import as_tuple, flatten
from pyop2.ir.ast_base import *

import assemble_expressions
from expression import Expression
import utils
from vector import Vector

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

def make_extruded_coords(mesh, layers, kernel=None, layer_height=None, extrusion_type='uniform'):
    """Given a kernel or height between layers, use it to generate the
    extruded coordinates.

    :arg mesh: the base nD (1D, 2D, etc) :class:`Mesh` to extrude
    :arg layers: the number of layers in the extruded mesh
    :arg kernel: :class:`pyop2.Kernel` which produces the extruded coordinates
    :arg layer_height: if provided it creates coordinates for evenly
                       spaced layers, the default value is 1/(layers-1)
    :arg extrusion_type: refers to how the coodinate field is computed in the
                         extrusion process.
                         `uniform`: create equidistant layers in the (n+1)-direction
                         `radial`: create equidistant layers in the outward direction
                         from the origin. For each extruded vertex the layer height is
                         multiplied by the corresponding unit direction vector and
                         then added to the position vector of the vertex

    Either the kernel or the layer_height must be provided. Should
    both be provided then the kernel takes precendence.
    Its calling signature is::

        void extrusion_kernel(double *extruded_coords[],
                              double *two_d_coords[],
                              int *layer_number[])

    So for example to build an evenly-spaced extruded mesh with eleven
    layers and height 1 in the vertical you would write::

       void extrusion_kernel(double *extruded_coords[],
                             double *two_d_coords[],
                             int *layer_number[]) {
           extruded_coords[0][0] = two_d_coords[0][0]; // X
           extruded_coords[0][1] = two_d_coords[0][1]; // Y
           extruded_coords[0][2] = 0.1 * layer_number[0][0]; // Z
       }
    """
    # The dimension of the space the coordinates are in
    coords_dim = mesh.ufl_cell().geometric_dimension()

    # Start code generation
    _height = str(layer_height)+" * j[0][0];"
    _extruded_direction = ""
    _norm = ""

    if extrusion_type == 'uniform':
        coords_xtr_dim = coords_dim + 1
        kernel_template = """
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Only the appropriate (n+1)th coordinate is increased
    %(init)s
    %(extruded_direction)s
}
        """
        _init = "\n".join(["xtr[0][%(i)s] = x[0][%(i)s];" % {"i": str(i)}
                           for i in range(coords_dim)]) + "\n"
        _extruded_direction = "xtr[0][%(i)s] = %(height)s" % \
                          {"i": str(coords_xtr_dim - 1),
                           "height": _height}
    elif extrusion_type == 'radial':
        coords_xtr_dim = coords_dim
        kernel_template = """
void extrusion_kernel(double *xtr[], double *x[], int* j[])
{
    //Use the position vector of the current coordinate as a
    //base for outward extrusion.
    double norm = sqrt(%(norm)s);
    %(init)s
}
        """
        _norm = " + ".join(["x[0][%(i)d]*x[0][%(i)d]" %
                            {"i": i} for i in range(coords_dim)])
        _init = "\n".join(["xtr[0][%(i)s] = x[0][%(i)s] + (x[0][%(i)s] / norm) * %(height)s;" %
                           {"i": str(i),
                            "height": _height}
                           for i in range(coords_dim)]) + "\n"
    else:
        raise NotImplementedError("Unsupported extrusion type.")

    kernel_template = kernel_template % {"init": _init,
                                         "extruded_direction": _extruded_direction,
                                         "height": _height,
                                         "norm": _norm}

    kernel = op2.Kernel(kernel_template, "extrusion_kernel")

    #dimension
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
    def __init__(self, mesh, count, kind, facet_cell, local_facet_number, markers=None, layers=1):

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
        self._layers = layers

    @utils.cached_property
    def set(self):
        # Currently no MPI parallel support
        size = self.count
        halo = None
        return op2.Set(size, "%s_%s_facets" % (self.mesh.name, self.kind), halo=halo, layers=self._layers)

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

    @property
    def layers(self):
        """Returns the number of layers in the mesh."""
        return self._layers

    @utils.cached_property
    def local_facet_dat(self):
        """Dat indicating which local facet of each adjacent
        cell corresponds to the current facet."""

        return op2.Dat(op2.DataSet(self.set, self._rank), self.local_facet_number,
                       np.uintc, "%s_%s_local_facet_number" % (self.mesh.name, self.kind))


class Mesh(object):
    """A representation of mesh topology and geometry."""
    def __init__(self, filename, dim=None):
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
        """

        _init()

        self._layers = 1

        self.cell_halo = None
        self.vertex_halo = None
        self.parent = None

        if dim is None:
            # Mesh reading in Fluidity level considers 0 to be None.
            dim = 0

        self._from_file(filename, dim)

        self._cell_orientations = op2.Dat(self.cell_set, dtype=np.int32,
                                          name="cell_orientations")
        # -1 is uninitialised.
        self._cell_orientations.data[:] = -1

    def _from_file(self, filename, dim=0):
        """Read a mesh from `filename`

        The extension of the filename determines the mesh type."""
        basename, ext = os.path.splitext(filename)

        # Retrieve mesh struct from Fluidity
        cdef ft.mesh_t mesh = ft.read_mesh_f(basename, _file_extensions[ext], dim)
        self.name = filename

        self._cells = np.array(<int[:mesh.cell_count, :mesh.cell_vertices:1]>mesh.element_vertex_list)
        self._ufl_cell = ufl.Cell(
            _cells[mesh.geometric_dimension][mesh.cell_vertices],
            geometric_dimension=mesh.geometric_dimension)

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

    :arg mesh:           2D unstructured mesh
    :arg layers:         number of layers in the "vertical"
                         direction representing the multiplicity of the
                         base mesh
    :arg kernel:         a :class:`pyop2.Kernel` to produce coordinates for the extruded
                         mesh see :func:`make_extruded_coords` for more details.
    :arg layer_height:   the height between layers when all layers are
                         evenly spaced.
    :arg extrusion_type: refers to how the coordinates are computed for the
                         evenly spaced layers:
                         `uniform`: layers are computed in the extra dimension
                         generated by the extrusion process
                         `radial`: radially extrudes the mesh points in the
                         outwards direction from the origin."""
    def __init__(self, mesh, layers, kernel=None, layer_height=None, extrusion_type='uniform'):
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

        interior_f = self._old_mesh.interior_facets
        self._interior_facets = _Facets(self, interior_f.count,
                                       "interior",
                                       interior_f.facet_cell,
                                       interior_f.local_facet_number,
                                       layers=layers)
        exterior_f = self._old_mesh.exterior_facets
        self._exterior_facets = _Facets(self, exterior_f.count,
                                           "exterior",
                                           exterior_f.facet_cell,
                                           exterior_f.local_facet_number,
                                           exterior_f.markers,
                                           layers=layers)

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
            layer_height = 1.0 / (layers - 1)
        self._coordinates = make_extruded_coords(mesh, layers, kernel, layer_height,
                                                 extrusion_type=extrusion_type)

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


class FunctionSpaceBase(object):
    """Base class for :class:`FunctionSpace`, :class:`VectorFunctionSpace` and
    :class:`MixedFunctionSpace`.

    .. note ::

        Users should not directly create objects of this class, but one of its
        derived types.
    """

    def __init__(self, mesh, element, name=None, dim=1, rank=0):
        """
        :param mesh: :class:`Mesh` to build this space on
        :param element: :class:`ufl.FiniteElementBase` to build this space from
        :param name: user-defined name for this space
        :param dim: vector space dimension of a :class:`VectorFunctionSpace`
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

        self._dim = dim
        self._index = None

        if mesh.interior_facets.count > 0:
            self.interior_facet_node_list = \
                np.array(<int[:mesh.interior_facets.count,:2*element_f.ndof]>
                         function_space.interior_facet_node_list)
        else:
            self.interior_facet_node_list = None

        if mesh.exterior_facets.count > 0:
            self.exterior_facet_node_list = \
                np.array(<int[:mesh.exterior_facets.count,:element_f.ndof]>
                         function_space.exterior_facet_node_list)
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

    @property
    def index(self):
        """Position of this :class:`FunctionSpaceBase` in the
        :class:`MixedFunctionSpace` it was extracted from."""
        return self._index

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
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`FunctionSpace`."""
        return op2.DataSet(self.node_set, self.dim)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.Dat` defined on the
        :attr:`dof_dset` of this :class:`Function`."""
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
        """The vector dimension of the :class:`FunctionSpace`. For a
        :class:`FunctionSpace` this is always one. For a
        :class:`VectorFunctionSpace` it is the value given to the
        constructor, and defaults to the geometric dimension of the :class:`Mesh`. """
        return self._dim

    @property
    def cdim(self):
        """The sum of the vector dimensions of the :class:`FunctionSpace`. For a
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
        """Create a :class:`MixedFunctionSpace` composed of this
        :class:`FunctionSpace` and other"""
        return MixedFunctionSpace((self, other))


class FunctionSpace(FunctionSpaceBase):
    """Create a function space

    :arg mesh: :class:`Mesh` to build the function space on
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
        # Two choices:
        # 1) pass in mesh, family, degree to generate a simple function space
        # 2) set up the function space using FiniteElement, EnrichedElement,
        #       OuterProductElement and so on
        if isinstance(family, ufl.FiniteElementBase):
            # Second case...
            element = family
        else:
            # First case...
            if isinstance(mesh, ExtrudedMesh):
                # if extruded mesh, make the OPE
                la = ufl.FiniteElement(family,
                                       domain=mesh._old_mesh._ufl_cell,
                                       degree=degree)
                if vfamily is None or vdegree is None:
                    # if second element was not passed in, assume same as first
                    # (only makes sense for CG or DG)
                    lb = ufl.FiniteElement(family,
                                           domain=ufl.Cell("interval",1),
                                           degree=degree)
                else:
                    # if second element was passed in, use in
                    lb = ufl.FiniteElement(vfamily,
                                           domain=ufl.Cell("interval",1),
                                           degree=vdegree)
                # now make the OPE
                element = ufl.OuterProductElement(la, lb)
            else:
                # if not an extruded mesh, just make the element
                element = ufl.FiniteElement(family,
                                            domain=mesh._ufl_cell,
                                            degree=degree)

        super(FunctionSpace, self).__init__(mesh, element, name, dim=1)


class VectorFunctionSpace(FunctionSpaceBase):
    """A vector finite element :class:`FunctionSpace`."""

    def __init__(self, mesh, family, degree, dim=None, name=None, vfamily=None, vdegree=None):
        # VectorFunctionSpace dimension defaults to the geometric dimension of the mesh.
        dim = dim or mesh.ufl_cell().geometric_dimension()

        if isinstance(mesh, ExtrudedMesh):
            if isinstance(family, ufl.OuterProductElement):
                raise NotImplementedError("Not yet implemented")
            la = ufl.FiniteElement(family,
                                   domain=mesh._old_mesh._ufl_cell,
                                   degree=degree)
            if vfamily is None or vdegree is None:
                lb = ufl.FiniteElement(family, domain=ufl.Cell("interval", 1),
                                       degree=degree)
            else:
                lb = ufl.FiniteElement(vfamily, domain=ufl.Cell("interval", 1),
                                       degree=vdegree)
            element = ufl.OuterProductVectorElement(la, lb)
        else:
            element = VectorElement(family, domain=mesh.ufl_cell(),
                                    degree=degree, dim=dim)
        super(VectorFunctionSpace, self).__init__(mesh, element, name, dim=dim, rank=1)


class MixedFunctionSpace(FunctionSpaceBase):
    """A mixed finite element :class:`FunctionSpace`."""

    def __init__(self, spaces, name=None):
        """
        :param spaces: a list (or tuple) of :class:`FunctionSpace`\s

        The function space may be created as ::

            V = MixedFunctionSpace(spaces)

        ``spaces`` may consist of multiple occurances of the same space: ::

            P1  = FunctionSpace(mesh, "CG", 1)
            P2v = VectorFunctionSpace(mesh, "Lagrange", 2)

            ME  = MixedFunctionSpace([P2v, P1, P1, P1])
        """

        self._spaces = [IndexedFunctionSpace(s, i, self)
                        for i, s in enumerate(flatten(spaces))]
        self._mesh = self._spaces[0].mesh()
        self._ufl_element = ufl.MixedElement(*[fs.ufl_element() for fs in self._spaces])
        self.name = '_'.join(str(s.name) for s in self._spaces)
        self.rank = 1
        self._index = None

    def split(self):
        """The list of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return self._spaces

    def sub(self, i):
        """Return the `i`th :class:`FunctionSpace` in this
        :class:`MixedFunctionSpace`."""
        return self[i]

    def num_sub_spaces(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
        return len(self)

    def __len__(self):
        """Return the number of :class:`FunctionSpace`\s of which this
        :class:`MixedFunctionSpace` is composed."""
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
        """Return a tuple of :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dim for fs in self._spaces)

    @property
    def cdim(self):
        """Return the sum of the :attr:`FunctionSpace.dim`\s of the
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of."""
        return sum(fs.dim for fs in self._spaces)

    @property
    def node_count(self):
        """Return a tuple of :attr:`FunctionSpace.node_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.node_count for fs in self._spaces)

    @property
    def dof_count(self):
        """Return a tuple of :attr:`FunctionSpace.dof_count`\s of the
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return tuple(fs.dof_count for fs in self._spaces)

    @utils.cached_property
    def node_set(self):
        """A :class:`pyop2.MixedSet` containing the nodes of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.node_set`\s of the underlying
        :class:`FunctionSpace`\s this :class:`MixedFunctionSpace` is
        composed of one or (for VectorFunctionSpaces) more degrees of freedom
        are stored at each node."""
        return op2.MixedSet(s.node_set for s in self._spaces)

    @utils.cached_property
    def dof_dset(self):
        """A :class:`pyop2.MixedDataSet` containing the degrees of freedom of
        this :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.dof_dset`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        return op2.MixedDataSet(s.dof_dset for s in self._spaces)

    def cell_node_map(self, bcs=None):
        """A :class:`pyop2.MixedMap` from the :attr:`Mesh.cell_set` of the
        underlying mesh to the :attr:`node_set` of this
        :class:`MixedFunctionSpace`. This is composed of the
        :attr:`FunctionSpace.cell_node_map`\s of the underlying
        :class:`FunctionSpace`\s of which this :class:`MixedFunctionSpace` is
        composed."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.cell_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def interior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.MixedMap` from interior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.interior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    def exterior_facet_node_map(self, bcs=None):
        """Return the :class:`pyop2.Map` from exterior facets to
        function space nodes. If present, bcs must be a tuple of
        :class:`.DirichletBC`\s. In this case, the facet_node_map will return
        negative node indices where boundary conditions should be
        applied. Where a PETSc matrix is employed, this will cause the
        corresponding values to be discarded during matrix assembly."""
        # FIXME: these want caching of sorts
        bc_list = [[] for _ in self]
        if bcs:
            for bc in bcs:
                bc_list[bc.function_space().index].append(bc)
        return op2.MixedMap(s.exterior_facet_node_map(bc_list[i])
                            for i, s in enumerate(self._spaces))

    @utils.cached_property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.MixedMap` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return op2.MixedMap(s.exterior_facet_boundary_node_map for s in self._spaces)

    def make_dat(self, val=None, valuetype=None, name=None, uid=None):
        """Return a newly allocated :class:`pyop2.MixedDat` defined on the
        :attr:`dof_dset` of this :class:`MixedFunctionSpace`."""
        if val is not None:
            assert len(val) == len(self)
        else:
            val = [None for _ in self]
        return op2.MixedDat(s.make_dat(v, valuetype, name, _new_uid())
                            for s, v in zip(self._spaces, val))


class IndexedFunctionSpace(FunctionSpaceBase):
    """A :class:`FunctionSpaceBase` with an index to indicate which position
    it has as part of a :class:`MixedFunctionSpace`."""

    def __init__(self, fs, index, parent):
        """
        :param fs: the :class:`FunctionSpaceBase` that was extracted
        :param index: the position in the parent :class:`MixedFunctionSpace`
        :param parent: the parent :class:`MixedFunctionSpace`
        """
        # If the function space was extracted from a mixed function space,
        # extract the underlying component space
        if isinstance(fs, IndexedFunctionSpace):
            fs = fs._fs
        # Override the __class__ to make instance checks on the type of the
        # wrapped function space work as expected
        self.__class__ = type(fs.__class__.__name__,
                              (self.__class__, fs.__class__), {})
        self._fs = fs
        self._index = index
        self._parent = parent

    def __getattr__(self, name):
        return getattr(self._fs, name)

    def __repr__(self):
        return "<IndexFunctionSpace: %r at %d>" % (FunctionSpaceBase.__repr__(self._fs), self._index)

    @property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`FunctionSpace`. One or (for VectorFunctionSpaces) more degrees
        of freedom are stored at each node."""
        return self._fs.node_set

    @property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`FunctionSpace`."""
        return self._fs.dof_dset

    @property
    def exterior_facet_boundary_node_map(self):
        '''The :class:`pyop2.Map` from exterior facets to the nodes on
        those facets. Note that this differs from
        :meth:`exterior_facet_node_map` in that only surface nodes
        are referenced, not all nodes in cells touching the surface.'''
        return self._fs.exterior_facet_boundary_node_map


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

    def __init__(self, function_space, val=None, name=None):
        """
        :param function_space: the :class:`FunctionSpaceBase` or another
            :class:`Function` to build this :class:`Function` on
        :param val: NumPy array-like with initial values or a :class:`op2.Dat`
            (optional)
        :param name: user-defined name of this :class:`Function` (optional)
        """

        if isinstance(function_space, Function):
            self._function_space = function_space._function_space
        elif isinstance(function_space, FunctionSpaceBase):
            self._function_space = function_space
        else:
            raise NotImplementedError("Can't make a Function defined on a "
                                      +str(type(function_space)))

        ufl.Coefficient.__init__(self, self._function_space.ufl_element())

        self._label = "a function"
        self.uid = _new_uid()
        self._name = name or 'function_%d' % self.uid

        if isinstance(val, op2.Dat):
            self.dat = val
        else:
            self.dat = self._function_space.make_dat(val, valuetype,
                                                     self._name, uid=self.uid)

        self._repr = None

        if isinstance(function_space, Function):
          self.assign(function_space)

    def split(self):
        """Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`FunctionSpace`."""
        return tuple(Function(fs, dat)
                     for fs, dat in zip(self._function_space, self.dat))

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

    def interpolate(self, expression, subset=None):
        """Interpolate an expression onto this :class:`Function`.

        :param expression: :class:`.Expression` to interpolate
        :returns: this :class:`Function` object"""

        # Make sure we have an expression of the right length i.e. a value for
        # each component in the value shape of each function space
        dims = [np.prod(fs.ufl_element().value_shape(), dtype=int)
                for fs in self.function_space()]
        if len(expression.code) != sum(dims):
            raise RuntimeError('Expression of length %d required, got length %d'
                               % (sum(dims), len(expression.code)))

        # Splice the expression and pass in the right number of values for
        # each component function space of this function
        d = 0
        for fs, dat, dim in zip(self.function_space(), self.dat, dims):
            idx = d if fs.rank == 0 else slice(d, d+dim)
            self._interpolate(fs, dat, Expression(expression.code[idx]), subset)
            d += dim
        return self

    def _interpolate(self, fs, dat, expression, subset):
        """Interpolate expression onto a :class:`FunctionSpace`.

        :param fs: :class:`FunctionSpace`
        :param dat: :class:`pyop2.Dat`
        :param expression: :class:`.Expression`
        """
        to_element = fs.fiat_element
        to_pts = []

        for dual in to_element.dual_basis():
            if not isinstance(dual, FIAT.functional.PointEvaluation):
                raise NotImplementedError("Can only interpolate onto point \
                    evaluation operators. Try projecting instead")
            to_pts.append(dual.pt_dict.keys()[0])

        if expression.rank() != len(fs.ufl_element().value_shape()):
            raise RuntimeError('Rank mismatch: Expression rank %d, FunctionSpace rank %d'
                               % (expression.rank(), len(fs.ufl_element().value_shape())))

        if expression.shape() != fs.ufl_element().value_shape():
            raise RuntimeError('Shape mismatch: Expression shape %r, FunctionSpace shape %r'
                               % (expression.shape(), fs.ufl_element().value_shape()))

        coords = fs.mesh()._coordinate_field
        coords_space = coords.function_space()
        coords_element = coords_space.fiat_element

        X=coords_element.tabulate(0, to_pts).values()[0]

        # Produce C array notation of X.
        X_str = "{{"+"},\n{".join([ ",".join(map(str,x)) for x in X.T])+"}}"

        ass_exp = [Assign(Symbol("A", ("k",), ((len(expression.code), i),)),
                          FlatBlock("%s" % code)) for i, code in enumerate(expression.code)]
        vals = {
            "x_array" : X_str,
            "dim" : coords_space.dim,
            "xndof" : coords_element.space_dimension(),
            "ndof" : to_element.space_dimension(),
            "assign_dim" : np.prod(expression.shape(), dtype=int) 
        }
        init = FlatBlock("""
const double X[%(ndof)d][%(xndof)d] = %(x_array)s;

double x[%(dim)d];
const double pi = 3.141592653589793;

""" % vals)
        block = FlatBlock("""
for (unsigned int d=0; d < %(dim)d; d++) {
  x[d] = 0;
  for (unsigned int i=0; i < %(xndof)d; i++) {
    x[d] += X[k][i] * x_[i][d];
  };
};

""" % vals)
        loop = c_for("k", "%(ndof)d" % vals, Block([block] + ass_exp, open_scope=True))
        kernel_code = FunDecl("void", "expression_kernel", \
                        [Decl("double", Symbol("A", (int("%(ndof)d" % vals),))), \
                         Decl("double**", c_sym("x_"))], \
                        Block([init, loop], open_scope=False))
        kernel = op2.Kernel(kernel_code, "expression_kernel")

        op2.par_loop(kernel, subset or self.cell_set,
                     dat(op2.WRITE, fs.cell_node_map()[op2.i[0]]),
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
        :attr:`node_set`. The expression will then only be assigned
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


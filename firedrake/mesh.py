import numpy as np
import tempfile
import os
import FIAT
import ufl
from shutil import rmtree

from pyop2 import op2
from pyop2.coffee import ast_base as ast
from pyop2.mpi import MPI
from pyop2.profiling import timed_function
from pyop2.utils import as_tuple

import dmplex
import extrusion_utils as eutils
import fiat_utils
import function
import functionspace
import utils
from parameters import parameters
from petsc import PETSc


__all__ = ['Mesh', 'ExtrudedMesh',
           'UnitIntervalMesh', 'UnitSquareMesh', 'UnitCircleMesh',
           'IntervalMesh', 'PeriodicIntervalMesh', 'PeriodicUnitIntervalMesh',
           'UnitTetrahedronMesh', 'UnitTriangleMesh', 'UnitCubeMesh',
           'IcosahedralSphereMesh', 'UnitIcosahedralSphereMesh']


_cachedir = os.path.join(tempfile.gettempdir(),
                         'firedrake-mesh-cache-uid%d' % os.getuid())


def _ensure_cachedir():
    if MPI.comm.rank == 0 and not os.path.exists(_cachedir):
        os.makedirs(_cachedir)

_ensure_cachedir()


def _clear_cachedir():
    if MPI.comm.rank == 0 and os.path.exists(_cachedir):
        rmtree(_cachedir, ignore_errors=True)
        _ensure_cachedir()

_exts = [".node", ".ele"]
_2dexts = [".edge"]
_3dexts = [".face"]
_pexts = [".halo"]


def _msh_exists(name):
    f = os.path.join(_cachedir, name)
    return os.path.exists(f + '.msh')


def _build_msh_file(input, output, dimension):
    try:
        # Must occur after mpi4py import due to:
        # 1) MPI initialisation issues
        # 2) LD_PRELOAD issues
        import gmshpy
        gmshpy.Msg.SetVerbosity(-1)
        # We've got the gmsh python interface available, so
        # use that, rather than spawning the gmsh binary.
        m = gmshpy.GModel()
        m.readGEO(input)
        m.mesh(dimension)
        m.writeMSH(output + ".msh")
        return
    except ImportError:
        raise RuntimeError('Creation of gmsh meshes requires gmshpy')


def _get_msh_file(source, name, dimension, meshed=False):
    """Given a source code, name and dimension  of the mesh,
    returns the name of the file that contains necessary information to build
    a mesh class. The mesh class would call _from_file method on this file
    to contruct itself.
    """

    if MPI.comm.rank == 0:
        input = os.path.join(_cachedir, name + '.geo')
        if not meshed:
            if not os.path.exists(input):
                with open(input, 'w') as f:
                    f.write(source)

        output = os.path.join(_cachedir, name)

        if not _msh_exists(name):
            if meshed:
                with file(output + '.msh', 'w') as f:
                    f.write(source)
            else:
                _build_msh_file(input, output, dimension)
        MPI.comm.bcast(output, root=0)
    else:
        output = MPI.comm.bcast(None, root=0)
    return output + '.msh'


class _Facets(object):
    """Wrapper class for facet interation information on a :class:`Mesh`

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

        self.markers = markers
        self.unique_markers = [] if unique_markers is None else unique_markers
        self._subsets = {}

    @utils.cached_property
    def set(self):
        size = self.classes
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

        return op2.Subset(self.set, [])

    def measure_set(self, integral_type, subdomain_id):
        '''Return the iteration set appropriate to measure. This will
        either be for all the interior or exterior (as appropriate)
        facets, or for a particular numbered subdomain.'''

        # ufl.Measure doesn't have enums for these any more :(
        if subdomain_id in ["everywhere", "otherwise"]:
            if integral_type == "exterior_facet_topbottom":
                return [(op2.ON_BOTTOM, self.bottom_set),
                        (op2.ON_TOP, self.bottom_set)]
            elif integral_type == "exterior_facet_bottom":
                return [(op2.ON_BOTTOM, self.bottom_set)]
            elif integral_type == "exterior_facet_top":
                return [(op2.ON_TOP, self.bottom_set)]
            elif integral_type == "interior_facet_horiz":
                return self.bottom_set
            else:
                return self.set
        else:
            return self.subset(subdomain_id)

    def subset(self, markers):
        """Return the subset corresponding to a given marker value.

        :param markers: integer marker id or an iterable of marker ids"""
        if self.markers is None:
            return self._null_subset
        markers = as_tuple(markers, int)
        try:
            return self._subsets[markers]
        except KeyError:
            indices = np.concatenate([np.nonzero(self.markers == i)[0]
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

    @timed_function("Build mesh")
    def __init__(self, filename, dim=None, periodic_coords=None, plex=None, reorder=None):
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
        :param reorder: optional flag indicating whether to reorder
               meshes for better cache locality.  If not supplied the
               default value in :py:data:`parameters["reorder_meshes"]`
               is used.
        """

        utils._init()

        # A cache of function spaces that have been built on this mesh
        self._cache = {}
        self.parent = None

        if dim is None:
            # Mesh reading in Fluidity level considers 0 to be None.
            dim = 0

        if reorder is None:
            reorder = parameters["reorder_meshes"]
        if plex is not None:
            self._from_dmplex(plex, geometric_dim=dim,
                              periodic_coords=periodic_coords,
                              reorder=reorder)
            self.name = filename
        else:
            basename, ext = os.path.splitext(filename)

            if ext in ['.e', '.exo', '.E', '.EXO']:
                self._from_exodus(filename, dim, reorder=reorder)
            elif ext in ['.cgns', '.CGNS']:
                self._from_cgns(filename, dim)
            elif ext in ['.msh']:
                self._from_gmsh(filename, dim, periodic_coords, reorder=reorder)
            elif ext in ['.node']:
                self._from_triangle(filename, dim, periodic_coords, reorder=reorder)
            else:
                raise RuntimeError("Unknown mesh file format.")

    @property
    def coordinates(self):
        """The :class:`.Function` containing the coordinates of this mesh."""
        return self._coordinate_function

    @coordinates.setter
    def coordinates(self, value):
        self._coordinate_function = value

    @timed_function("Build mesh from DMPlex")
    def _from_dmplex(self, plex, geometric_dim=0,
                     periodic_coords=None, reorder=None):
        """ Create mesh from DMPlex object """

        self._plex = plex
        self.uid = utils._new_uid()

        # Mark exterior and interior facets
        # Note.  This must come before distribution, because otherwise
        # DMPlex will consider facets on the domain boundary to be
        # exterior, which is wrong.
        dmplex.label_facets(self._plex)

        if geometric_dim == 0:
            geometric_dim = plex.getDimension()

        # Distribute the dm to all ranks
        if op2.MPI.comm.size > 1:
            self.parallel_sf = plex.distribute(overlap=1)

        self._plex = plex

        if reorder:
            old_to_new = self._plex.getOrdering(PETSc.Mat.OrderingType.RCM).indices
            reordering = np.empty_like(old_to_new)
            reordering[old_to_new] = np.arange(old_to_new.size, dtype=old_to_new.dtype)
        else:
            # No reordering
            reordering = None

        # Mark OP2 entities and derive the resulting Plex renumbering
        dmplex.mark_entity_classes(self._plex)
        self._plex_renumbering = dmplex.plex_renumbering(self._plex, reordering)

        cStart, cEnd = self._plex.getHeightStratum(0)  # cells
        cell_vertices = self._plex.getConeSize(cStart)
        self._ufl_cell = ufl.Cell(fiat_utils._cells[geometric_dim][cell_vertices],
                                  geometric_dimension=geometric_dim)

        self._ufl_domain = ufl.Domain(self.ufl_cell(), data=self)
        dim = self._plex.getDimension()
        self._cells, self.cell_classes = dmplex.get_cells_by_class(self._plex)

        # Derive a cell numbering from the Plex renumbering
        cell_entity_dofs = np.zeros(dim+1, dtype=np.int32)
        cell_entity_dofs[-1] = 1
        self._cell_numbering = self._plex.createSection(1, [1],
                                                        cell_entity_dofs,
                                                        perm=self._plex_renumbering)

        self._cell_closure = None
        self.interior_facets = None
        self.exterior_facets = None

        # Note that for bendy elements, this needs to change.
        if periodic_coords is not None:
            if self.ufl_cell().geometric_dimension() != 1:
                raise NotImplementedError("Periodic coordinates in more than 1D are unsupported")
            # We've been passed a periodic coordinate field, so use that.
            self._coordinate_fs = functionspace.VectorFunctionSpace(self, "DG", 1)
            self.coordinates = function.Function(self._coordinate_fs,
                                                 val=periodic_coords,
                                                 name="Coordinates")
        else:
            self._coordinate_fs = functionspace.VectorFunctionSpace(self, "Lagrange", 1)

            coordinates = dmplex.reordered_coords(self._plex, self._coordinate_fs._global_numbering,
                                                  (self.num_vertices(), geometric_dim))
            self.coordinates = function.Function(self._coordinate_fs,
                                                 val=coordinates,
                                                 name="Coordinates")
        self._ufl_domain = ufl.Domain(self.coordinates)
        # Build a new ufl element for this function space with the
        # correct domain.  This is necessary since this function space
        # is in the cache and will be picked up by later
        # VectorFunctionSpace construction.
        self._coordinate_fs._ufl_element = self._coordinate_fs.ufl_element().reconstruct(domain=self.ufl_domain())
        # HACK alert!
        # Replace coordinate Function by one that has a real domain on it (but don't copy values)
        self.coordinates = function.Function(self._coordinate_fs, val=self.coordinates.dat)
        # Add domain and subdomain_data to the measure objects we store with the mesh.
        self._dx = ufl.Measure('cell', domain=self, subdomain_data=self.coordinates)
        self._ds = ufl.Measure('exterior_facet', domain=self, subdomain_data=self.coordinates)
        self._dS = ufl.Measure('interior_facet', domain=self, subdomain_data=self.coordinates)
        # Set the subdomain_data on all the default measures to this
        # coordinate field.  Also set the domain on the measure.
        for measure in [ufl.dx, ufl.ds, ufl.dS]:
            measure._subdomain_data = self.coordinates
            measure._domain = self.ufl_domain()

    @timed_function("Build mesh from Gmsh")
    def _from_gmsh(self, filename, dim=0, periodic_coords=None, reorder=None):
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

        self._from_dmplex(gmsh_plex, dim, periodic_coords, reorder=reorder)

    @timed_function("Build mesh from Exodus")
    def _from_exodus(self, filename, dim=0, reorder=None):
        self.name = filename
        plex = PETSc.DMPlex().createExodusFromFile(filename)

        boundary_ids = dmplex.getLabelIdIS("Face Sets").getIndices()
        plex.createLabel("boundary_ids")
        for bid in boundary_ids:
            faces = plex.getStratumIS("Face Sets", bid).getIndices()
            for f in faces:
                plex.setLabelValue("boundary_ids", f, bid)

        self._from_dmplex(plex, reorder=reorder)

    @timed_function("Build mesh from CGNS")
    def _from_cgns(self, filename, dim=0, reorder=None):
        self.name = filename
        plex = PETSc.DMPlex().createCGNSFromFile(filename)

        #TODO: Add boundary IDs
        self._from_dmplex(plex, reorder=reorder)

    @timed_function("Build mesh from triangle")
    def _from_triangle(self, filename, dim=0, periodic_coords=None, reorder=None):
        """Read a set of triangle mesh files from `filename`"""
        self.name = filename
        basename, ext = os.path.splitext(filename)

        if op2.MPI.comm.rank == 0:
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
            op2.MPI.comm.bcast(tdim, root=0)

            with open(basename+".node") as nodefile:
                header = np.fromfile(nodefile, dtype=np.int32, count=2, sep=' ')
                nodecount = header[0]
                nodedim = header[1]
                assert nodedim == dim
                coordinates = np.loadtxt(nodefile, usecols=range(1, dim+1), skiprows=1, delimiter=' ')
                assert nodecount == coordinates.shape[0]

            with open(basename+".ele") as elefile:
                header = np.fromfile(elefile, dtype=np.int32, count=2, sep=' ')
                elecount = header[0]
                eledim = header[1]
                eles = np.loadtxt(elefile, usecols=range(1, eledim+1), dtype=np.int32, skiprows=1, delimiter=' ')
                assert elecount == eles.shape[0]

            cells = map(lambda c: c-1, eles)
        else:
            tdim = op2.MPI.comm.bcast(None, root=0)
            cells = None
            coordinates = None
        plex = dmplex._from_cell_list(tdim, cells, coordinates, comm=op2.MPI.comm)

        # Apply boundary IDs
        if op2.MPI.comm.rank == 0:
            facets = None
            try:
                header = np.fromfile(facetfile, dtype=np.int32, count=2, sep=' ')
                edgecount = header[0]
                facets = np.loadtxt(facetfile, usecols=range(1, tdim+2), dtype=np.int32, skiprows=0, delimiter=' ')
                assert edgecount == facets.shape[0]
            finally:
                facetfile.close()

            if facets is not None:
                vStart, vEnd = plex.getDepthStratum(0)   # vertices
                for facet in facets:
                    bid = facet[-1]
                    vertices = map(lambda v: v + vStart - 1, facet[:-1])
                    join = plex.getJoin(vertices)
                    plex.setLabelValue("boundary_ids", join[0], bid)

        self._from_dmplex(plex, dim, periodic_coords, reorder=None)

    @property
    def layers(self):
        """Return the number of layers of the extruded mesh
        represented by the number of occurences of the base mesh."""
        return self._layers

    def cell_orientations(self):
        """Return the orientation of each cell in the mesh.

        Use :func:`init_cell_orientations` to initialise this data."""
        if not hasattr(self, '_cell_orientations'):
            raise RuntimeError("No cell orientations found, did you forget to call init_cell_orientations?")
        return self._cell_orientations

    def init_cell_orientations(self, expr):
        """Compute and initialise `cell_orientations` relative to a specified orientation.

        :arg expr: an :class:`.Expression` evaluated to produce a
             reference normal direction.

        """
        if expr.shape()[0] != 3:
            raise NotImplementedError('Only implemented for 3-vectors')
        if self.ufl_cell() != ufl.Cell('triangle', 3):
            raise NotImplementedError('Only implemented for triangles embedded in 3d')

        if hasattr(self, '_cell_orientations'):
            raise RuntimeError("init_cell_orientations already called, did you mean to do so again?")

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

        body.extend([ast.FlatBlock("dot += (%(x)s) * n[%(i)d];\n" % {"x": x_, "i": i})
                     for i, x_ in enumerate(expr.code)])
        body.append(ast.Assign("orientation[0][0]", ast.Ternary(ast.Less("dot", 0), 1, 0)))

        kernel = op2.Kernel(ast.FunDecl("void", "cell_orientations",
                                        [ast.Decl("int**", "orientation"),
                                         ast.Decl("double**", "coords")],
                                        ast.Block(body)),
                            "cell_orientations")

        # Build the cell orientations as a DG0 field (so that we can
        # pass it in for facet integrals and the like)
        fs = functionspace.FunctionSpace(self, 'DG', 0)
        cell_orientations = function.Function(fs, name="cell_orientations", dtype=np.int32)
        op2.par_loop(kernel, self.cell_set,
                     cell_orientations.dat(op2.WRITE, cell_orientations.cell_node_map()),
                     self.coordinates.dat(op2.READ, self.coordinates.cell_node_map()))
        self._cell_orientations = cell_orientations

    def cells(self):
        return self._cells

    def ufl_id(self):
        return id(self)

    def ufl_domain(self):
        return self._ufl_domain

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

    @timed_function("Build extruded mesh")
    def __init__(self, mesh, layers, kernel=None, layer_height=None, extrusion_type='uniform'):
        # A cache of function spaces that have been built on this mesh
        self._cache = {}
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
        self.name = mesh.name
        self._plex = mesh._plex
        self._plex_renumbering = mesh._plex_renumbering
        self._cell_numbering = mesh._cell_numbering
        self._cell_closure = mesh._cell_closure

        interior_f = self._old_mesh.interior_facets
        self._interior_facets = _Facets(self, interior_f.classes,
                                        "interior",
                                        interior_f.facet_cell,
                                        interior_f.local_facet_number)
        exterior_f = self._old_mesh.exterior_facets
        self._exterior_facets = _Facets(self, exterior_f.classes,
                                        "exterior",
                                        exterior_f.facet_cell,
                                        exterior_f.local_facet_number,
                                        exterior_f.markers)

        self.ufl_cell_element = ufl.FiniteElement("Lagrange",
                                                  domain=mesh._ufl_cell,
                                                  degree=1)
        self.ufl_interval_element = ufl.FiniteElement("Lagrange",
                                                      domain=ufl.Cell("interval", 1),
                                                      degree=1)

        self.fiat_base_element = fiat_utils.fiat_from_ufl_element(self.ufl_cell_element)
        self.fiat_vert_element = fiat_utils.fiat_from_ufl_element(self.ufl_interval_element)

        fiat_element = FIAT.tensor_finite_element.TensorFiniteElement(self.fiat_base_element, self.fiat_vert_element)

        self._ufl_cell = ufl.OuterProductCell(mesh._ufl_cell, ufl.Cell("interval", 1))

        self._ufl_domain = ufl.Domain(self.ufl_cell(), data=self)
        flat_temp = fiat_element.flattened_element()

        # Calculated dofs_per_column from flattened_element and layers.
        # The mirrored elements have to be counted only once.
        # Then multiply by layers and layers - 1 accordingly.
        self.dofs_per_column = eutils.compute_extruded_dofs(fiat_element, flat_temp.entity_dofs(),
                                                            layers)

        #Compute Coordinates of the extruded mesh
        if layer_height is None:
            # Default to unit
            layer_height = 1.0 / layers

        self._coordinate_fs = functionspace.VectorFunctionSpace(self, mesh._coordinate_fs.ufl_element().family(),
                                                                mesh._coordinate_fs.ufl_element().degree(),
                                                                vfamily="CG",
                                                                vdegree=1)

        self.coordinates = function.Function(self._coordinate_fs)
        self._ufl_domain = ufl.Domain(self.coordinates)
        eutils.make_extruded_coords(self, layer_height, extrusion_type=extrusion_type,
                                    kernel=kernel)
        # Build a new ufl element for this function space with the
        # correct domain.  This is necessary since this function space
        # is in the cache and will be picked up by later
        # VectorFunctionSpace construction.
        self._coordinate_fs._ufl_element = self._coordinate_fs.ufl_element().reconstruct(domain=self.ufl_domain())
        # HACK alert!
        # Replace coordinate Function by one that has a real domain on it (but don't copy values)
        self.coordinates = function.Function(self._coordinate_fs, val=self.coordinates.dat)
        self._dx = ufl.Measure('cell', domain=self, subdomain_data=self.coordinates)
        self._ds = ufl.Measure('exterior_facet', domain=self, subdomain_data=self.coordinates)
        self._dS = ufl.Measure('interior_facet', domain=self, subdomain_data=self.coordinates)
        self._ds_t = ufl.Measure('exterior_facet_top', domain=self, subdomain_data=self.coordinates)
        self._ds_b = ufl.Measure('exterior_facet_bottom', domain=self, subdomain_data=self.coordinates)
        self._ds_v = ufl.Measure('exterior_facet_vert', domain=self, subdomain_data=self.coordinates)
        self._dS_h = ufl.Measure('interior_facet_horiz', domain=self, subdomain_data=self.coordinates)
        self._dS_v = ufl.Measure('interior_facet_vert', domain=self, subdomain_data=self.coordinates)
        # Set the subdomain_data on all the default measures to this coordinate field.
        for measure in [ufl.ds, ufl.dS, ufl.dx, ufl.ds_t, ufl.ds_b, ufl.ds_v, ufl.dS_h, ufl.dS_v]:
            measure._subdomain_data = self.coordinates
            measure._domain = self.ufl_domain()

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


class UnitSquareMesh(Mesh):

    """Class that represents a structured triangular mesh of a 2D square whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.
    :arg reorder: Should the mesh be reordered?

    The number of the elements in a mesh can be computed from 2 * nx * ny,
    and the number of vertices from (nx+1) * (ny+1).

    The boundary edges are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    """

    def __init__(self, nx, ny, reorder=None):
        self.name = "unitsquare_%d_%d" % (nx, ny)

        # Create mesh from DMPlex
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(1)
        boundary.createSquareBoundary([0., 0.], [1., 1.], [nx, ny])
        plex = PETSc.DMPlex().generate(boundary)

        # Apply boundary IDs
        plex.createLabel("boundary_ids")
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if face_coords[0] == 0. and face_coords[2] == 0.:
                    plex.setLabelValue("boundary_ids", face, 1)
                if face_coords[0] == 1. and face_coords[2] == 1.:
                    plex.setLabelValue("boundary_ids", face, 2)
                if face_coords[1] == 0. and face_coords[3] == 0.:
                    plex.setLabelValue("boundary_ids", face, 3)
                if face_coords[1] == 1. and face_coords[3] == 1.:
                    plex.setLabelValue("boundary_ids", face, 4)

        super(UnitSquareMesh, self).__init__(self.name, plex=plex, reorder=reorder)


class UnitCubeMesh(Mesh):

    """Class that represents a structured tetrahedron mesh of a 3D cube whose
    edge is a unit length.

    :arg nx: The number of the cells in the x direction.
    :arg ny: The number of the cells in the y direction.
    :arg nx: The number of the cells in the z direction.
    :arg reorder: Should the mesh be reordered?

    The number of the elements in a mesh can be computed from 6 * nx * ny * nz,
    and the number of the vertices from (nx+1) * (ny+1) * (nz+1).

    The boundary surface are numbered as follows:

    * 1: plane x == 0
    * 2: plane x == 1
    * 3: plane y == 0
    * 4: plane y == 1
    * 5: plane z == 0
    * 6: plane z == 1
    """

    def __init__(self, nx, ny, nz, reorder=None):
        self.name = "unitcube_%d_%d_%d" % (nx, ny, nz)

        # Create mesh from DMPlex
        boundary = PETSc.DMPlex().create(MPI.comm)
        boundary.setDimension(2)
        boundary.createCubeBoundary([0., 0., 0.], [1., 1., 1.], [nx, ny, nz])
        plex = PETSc.DMPlex().generate(boundary)

        # Apply boundary IDs
        plex.createLabel("boundary_ids")
        plex.markBoundaryFaces("boundary_faces")
        coords = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        if plex.getStratumSize("boundary_faces", 1) > 0:
            boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if face_coords[0] == 0. and face_coords[3] == 0. and face_coords[6] == 0.:
                    plex.setLabelValue("boundary_ids", face, 1)
                if face_coords[0] == 1. and face_coords[3] == 1. and face_coords[6] == 1.:
                    plex.setLabelValue("boundary_ids", face, 2)
                if face_coords[1] == 0. and face_coords[4] == 0. and face_coords[7] == 0.:
                    plex.setLabelValue("boundary_ids", face, 3)
                if face_coords[1] == 1. and face_coords[4] == 1. and face_coords[7] == 1.:
                    plex.setLabelValue("boundary_ids", face, 4)
                if face_coords[2] == 0. and face_coords[5] == 0. and face_coords[8] == 0.:
                    plex.setLabelValue("boundary_ids", face, 5)
                if face_coords[2] == 1. and face_coords[5] == 1. and face_coords[8] == 1.:
                    plex.setLabelValue("boundary_ids", face, 6)

        super(UnitCubeMesh, self).__init__(self.name, plex=plex, reorder=reorder)


class UnitCircleMesh(Mesh):

    """Class that represents a structured triangle mesh of a 2D circle of an
    unit circle.

    :arg resolution: The number of cells lying along the radius and the arc of
      the quadrant.
    :arg reorder: Should the mesh be reordered?
    """

    def __init__(self, resolution, reorder=None):
        source = """
            lc = %g;
            Point(1) = {0, -0.5, 0, lc};
            Point(2) = {0, 0.5, 0, lc};
            Line(1) = {1, 2};
            surface[] = Extrude{{0, 0, 1},{0, 0, 0}, 0.9999 * Pi}{
                    Line{1};Layers{%d};
            };
            Physical Surface(2) = { surface[1] };
            """ % (0.5 / resolution, resolution * 4)
        self.name = "unitcircle_%d" % resolution

        output = _get_msh_file(source, self.name, 2)
        super(UnitCircleMesh, self).__init__(output, reorder=reorder)


class IntervalMesh(Mesh):
    """
    Generate a uniform mesh of the interval [0,L] for user specified L.

    :arg ncells: The number of the cells over the interval.
    :arg length: The length of the interval.

    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=L`) point has marker 2.
    """
    def __init__(self, ncells, length):
        self.name = "interval"
        dx = length / ncells
        # This ensures the rightmost point is actually present.
        coords = np.arange(0, length + 0.01 * dx, dx).reshape(-1, 1)
        cells = np.dstack((np.arange(0, len(coords) - 1, dtype=np.int32),
                           np.arange(1, len(coords), dtype=np.int32))).reshape(-1, 2)
        plex = dmplex._from_cell_list(1, cells, coords)
        # Apply boundary IDs
        plex.createLabel("boundary_ids")
        coordinates = plex.getCoordinates()
        coord_sec = plex.getCoordinateSection()
        vStart, vEnd = plex.getDepthStratum(0)  # vertices
        for v in range(vStart, vEnd):
            vcoord = plex.vecGetClosure(coord_sec, coordinates, v)
            if vcoord[0] == coords[0]:
                plex.setLabelValue("boundary_ids", v, 1)
            if vcoord[0] == coords[-1]:
                plex.setLabelValue("boundary_ids", v, 2)

        super(IntervalMesh, self).__init__(self.name, plex=plex, reorder=False)


class UnitIntervalMesh(IntervalMesh):
    """
    Generate a uniform mesh of the interval [0,1].

    :arg ncells: The number of the cells over the interval.
    The left hand (:math:`x=0`) boundary point has boundary marker 1,
    while the right hand (:math:`x=1`) point has marker 2.
    """
    def __init__(self, ncells):
        self.name = "unitinterval"
        IntervalMesh.__init__(self, ncells, length=1.0)


class PeriodicIntervalMesh(Mesh):
    """Generate a periodic uniform mesh of the interval [0, L], for
    user specified L.

    :arg ncells: The number of cells over the interval.
    :arg length: The length the interval."""
    def __init__(self, ncells, length):
        self.name = "periodicinterval"

        """Build the periodic Plex by hand"""

        if MPI.comm.size > 1:
            raise NotImplementedError("Periodic intervals not yet implemented in parallel")
        nvert = ncells
        nedge = ncells
        plex = PETSc.DMPlex().create()
        plex.setDimension(1)
        plex.setChart(0, nvert+nedge)
        for e in range(nedge):
            plex.setConeSize(e, 2)
        plex.setUp()
        for e in range(nedge-1):
            plex.setCone(e, [nedge+e, nedge+e+1])
            plex.setConeOrientation(e, [0, 0])
        # Connect v_(n-1) with v_0
        plex.setCone(nedge-1, [nedge+nvert-1, nedge])
        plex.setConeOrientation(nedge-1, [0, 0])
        plex.symmetrize()
        plex.stratify()

        # Build coordinate section
        dx = length / ncells
        coords = [x for x in np.arange(0, length + 0.01 * dx, dx)]

        coordsec = plex.getCoordinateSection()
        coordsec.setChart(nedge, nedge+nvert)
        for v in range(nedge, nedge+nvert):
            coordsec.setDof(v, 1)
        coordsec.setUp()
        size = coordsec.getStorageSize()
        coordvec = PETSc.Vec().createWithArray(coords, size=size)
        plex.setCoordinatesLocal(coordvec)

        dx = length / ncells
        # HACK ALERT!
        # Almost certainly not right when symbolic geometry stuff lands.
        # Hopefully DMPlex will eventually give us a DG coordinate
        # field.  Until then, we build one by hand.
        coords = np.dstack((np.arange(dx, length + dx*0.01, dx),
                            np.arange(0, length - dx*0.01, dx))).flatten()
        # Last cell is back to front.
        coords[-2:] = coords[-2:][::-1]
        Mesh.__init__(self, self.name, plex=plex,
                      periodic_coords=coords, reorder=False)


class PeriodicUnitIntervalMesh(PeriodicIntervalMesh):
    """Generate a periodic uniform mesh of the interval [0, 1].
    :arg ncells: The number of cells over the interval."""
    def __init__(self, ncells):
        self.name = "periodicunitinterval"
        PeriodicIntervalMesh.__init__(self, ncells, length=1.0)


class UnitTetrahedronMesh(Mesh):

    """Class that represents a tetrahedron mesh that is composed of one
    element.
    """

    def __init__(self):
        self.name = "unittetra"
        coords = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        cells = [[1, 0, 3, 2]]
        plex = dmplex._from_cell_list(3, cells, coords)
        super(UnitTetrahedronMesh, self).__init__(self.name, plex=plex)


class UnitTriangleMesh(Mesh):

    """Class that represents a triangle mesh composed of one element."""

    def __init__(self):
        self.name = "unittri"
        coords = [[0., 0.], [1., 0.], [0., 1.]]
        cells = [[1, 2, 0]]
        plex = dmplex._from_cell_list(2, cells, coords)
        super(UnitTriangleMesh, self).__init__(self.name, plex=plex)


class IcosahedralSphereMesh(Mesh):

    from math import sqrt

    """An icosahedral mesh of the surface of the sphere"""
    phi = (1 + sqrt(5)) / 2
    del sqrt
    # vertices of an icosahedron with an edge length of 2
    _base_vertices = np.array([[-1, phi, 0],
                               [1, phi, 0],
                               [-1, -phi, 0],
                               [1, -phi, 0],
                               [0, -1, phi],
                               [0, 1, phi],
                               [0, -1, -phi],
                               [0, 1, -phi],
                               [phi, 0, -1],
                               [phi, 0, 1],
                               [-phi, 0, -1],
                               [-phi, 0, 1]])
    del phi
    # faces of the base icosahedron
    _base_faces = np.array([[0, 11, 5],
                            [0, 5, 1],
                            [0, 1, 7],
                            [0, 7, 10],
                            [0, 10, 11],
                            [1, 5, 9],
                            [5, 11, 4],
                            [11, 10, 2],
                            [10, 7, 6],
                            [7, 1, 8],
                            [3, 9, 4],
                            [3, 4, 2],
                            [3, 2, 6],
                            [3, 6, 8],
                            [3, 8, 9],
                            [4, 9, 5],
                            [2, 4, 11],
                            [6, 2, 10],
                            [8, 6, 7],
                            [9, 8, 1]], dtype=np.int32)

    def __init__(self, radius=1, refinement_level=0, reorder=None):
        """
        :arg radius: the radius of the sphere to approximate.
             For a radius R the edge length of the underlying
             icosahedron will be.

             .. math::

                a = \\frac{R}{\\sin(2 \\pi / 5)}

        :arg refinement_level: how many levels of refinement, zero
                               corresponds to an icosahedron.
        :arg reorder: Should the mesh be reordered?
        """

        self.name = "icosahedralspheremesh_%d_%g" % (refinement_level, radius)

        self._R = radius
        self._refinement = refinement_level

        self._vertices = np.empty_like(IcosahedralSphereMesh._base_vertices)
        self._faces = np.copy(IcosahedralSphereMesh._base_faces)
        # Rescale so that vertices live on sphere of specified radius
        for i, vtx in enumerate(IcosahedralSphereMesh._base_vertices):
            self._vertices[i] = self._force_to_sphere(vtx)

        for i in range(refinement_level):
            self._refine()

        plex = dmplex._from_cell_list(2, self._faces, self._vertices)
        super(IcosahedralSphereMesh, self).__init__(self.name, plex=plex, dim=3, reorder=reorder)

    def _force_to_sphere(self, vtx):
        """
        Scale `vtx` such that it sits on surface of the sphere this mesh
        represents.

        """
        scale = self._R / np.linalg.norm(vtx)
        return vtx * scale

    def _refine(self):
        """Refine mesh by one level.

        This increases the number of faces in the mesh by a factor of four."""
        cache = {}
        new_faces = np.empty((4 * len(self._faces), 3), dtype=np.int32)
        # Dividing each face adds 1.5 extra vertices (each vertex on
        # the midpoint is shared two ways).
        new_vertices = np.empty((len(self._vertices) + 3 * len(self._faces) / 2, 3))
        f_idx = 0
        v_idx = len(self._vertices)
        new_vertices[:v_idx] = self._vertices

        def midpoint(v1, v2):
            return self._force_to_sphere((self._vertices[v1] + self._vertices[v2])/2)

        # Walk old faces, splitting into 4
        for (v1, v2, v3) in self._faces:
            a = midpoint(v1, v2)
            b = midpoint(v2, v3)
            c = midpoint(v3, v1)
            ka = tuple(sorted((v1, v2)))
            kb = tuple(sorted((v2, v3)))
            kc = tuple(sorted((v3, v1)))
            if ka not in cache:
                cache[ka] = v_idx
                new_vertices[v_idx] = a
                v_idx += 1
            va = cache[ka]
            if kb not in cache:
                cache[kb] = v_idx
                new_vertices[v_idx] = b
                v_idx += 1
            vb = cache[kb]
            if kc not in cache:
                cache[kc] = v_idx
                new_vertices[v_idx] = c
                v_idx += 1
            vc = cache[kc]
            #
            #         v1
            #        /  \
            #       /    \
            #      v2----v3
            #
            #         v1
            #        /  \
            #       a--- c
            #      / \  / \
            #     /   \/   \
            #   v2----b----v3
            #
            new_faces[f_idx][:] = (v1, va, vc)
            new_faces[f_idx+1][:] = (v2, vb, va)
            new_faces[f_idx+2][:] = (v3, vc, vb)
            new_faces[f_idx+3][:] = (va, vb, vc)
            f_idx += 4
        self._vertices = new_vertices
        self._faces = new_faces


class UnitIcosahedralSphereMesh(IcosahedralSphereMesh):
    """An icosahedral approximation to the unit sphere."""
    def __init__(self, refinement_level=0, reorder=None):
        """
        :arg refinement_level: how many levels to refine the mesh.
        :arg reorder: Should the mesh be reordered?
        """
        super(UnitIcosahedralSphereMesh, self).__init__(1, refinement_level, reorder=reorder)

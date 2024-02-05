import collections
import itertools
import numpy
import os
import ufl
import finat.ufl
from ufl.domain import extract_unique_domain
from itertools import chain
from pyop2.mpi import COMM_WORLD, internal_comm
from pyop2.utils import as_tuple
from pyadjoint import no_annotations
from firedrake.petsc import PETSc
from firedrake.utils import IntType

from .paraview_reordering import (
    vtk_lagrange_tet_reorder,
    vtk_lagrange_hex_reorder,
    vtk_lagrange_interval_reorder,
    vtk_lagrange_triangle_reorder,
    vtk_lagrange_quad_reorder,
    vtk_lagrange_wedge_reorder,
)

__all__ = ("File", )


VTK_INTERVAL = 3
VTK_TRIANGLE = 5
VTK_QUADRILATERAL = 9
VTK_TETRAHEDRON = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
#  Lagrange VTK cells:
VTK_LAGRANGE_CURVE = 68
VTK_LAGRANGE_TRIANGLE = 69
VTK_LAGRANGE_QUADRILATERAL = 70
VTK_LAGRANGE_TETRAHEDRON = 71
VTK_LAGRANGE_HEXAHEDRON = 72
VTK_LAGRANGE_WEDGE = 73


ufl_quad = ufl.TensorProductCell(ufl.Cell("interval"),
                                 ufl.Cell("interval"))
ufl_wedge = ufl.TensorProductCell(ufl.Cell("triangle"),
                                  ufl.Cell("interval"))
ufl_hex = ufl.TensorProductCell(ufl.Cell("quadrilateral"),
                                ufl.Cell("interval"))
cells = {
    (ufl.Cell("interval"), False): VTK_INTERVAL,
    (ufl.Cell("interval"), True): VTK_LAGRANGE_CURVE,
    (ufl.Cell("triangle"), False): VTK_TRIANGLE,
    (ufl.Cell("triangle"), True): VTK_LAGRANGE_TRIANGLE,
    (ufl.Cell("quadrilateral"), False): VTK_QUADRILATERAL,
    (ufl.Cell("quadrilateral"), True): VTK_LAGRANGE_QUADRILATERAL,
    (ufl_quad, True): VTK_LAGRANGE_QUADRILATERAL,
    (ufl_quad, False): VTK_QUADRILATERAL,
    (ufl.Cell("tetrahedron"), False): VTK_TETRAHEDRON,
    (ufl.Cell("tetrahedron"), True): VTK_LAGRANGE_TETRAHEDRON,
    (ufl_wedge, False): VTK_WEDGE,
    (ufl_wedge, True): VTK_LAGRANGE_WEDGE,
    (ufl_hex, False): VTK_HEXAHEDRON,
    (ufl_hex, True): VTK_LAGRANGE_HEXAHEDRON,
    (ufl.Cell("hexahedron"), False): VTK_HEXAHEDRON,
    (ufl.Cell("hexahedron"), True): VTK_LAGRANGE_HEXAHEDRON,
}


OFunction = collections.namedtuple("OFunction", ["array", "name", "function"])


def is_cg(V):
    """Is the provided space continuous?

    :arg V: A FunctionSpace.
    """
    nvertex = V.mesh().ufl_cell().num_vertices()
    entity_dofs = V.finat_element.entity_dofs()
    # If there are as many dofs on vertices as there are vertices,
    # assume a continuous space.
    try:
        return sum(map(len, entity_dofs[0].values())) == nvertex
    except KeyError:
        return sum(map(len, entity_dofs[(0, 0)].values())) == nvertex


def is_dg(V):
    """Is the provided space fully discontinuous?

    :arg V: A FunctionSpace.
    """
    return V.finat_element.entity_dofs() == V.finat_element.entity_closure_dofs()


def is_linear(V):
    """Is the provided space linear?

    :arg V: A FunctionSpace.
    """
    nvertex = V.mesh().ufl_cell().num_vertices()
    return V.finat_element.space_dimension() == nvertex


def get_sup_element(*elements, continuous=False, max_degree=None):
    """Given ufl elements and a continuity flag, return
    a new ufl element that contains all elements.
    :arg elements: ufl elements.
    :continous: A flag indicating if all elements are continous.
    :returns: A ufl element containing all elements.
    """
    try:
        cell, = set(e.cell for e in elements)
    except ValueError:
        raise ValueError("All cells must be identical")
    degree = max(chain(*(as_tuple(e.degree()) for e in elements)))
    if continuous:
        family = "CG"
    else:
        if cell.cellname() in {"interval", "triangle", "tetrahedron"}:
            family = "DG"
        else:
            family = "DQ"
    return finat.ufl.FiniteElement(
        family, cell=cell, degree=degree if max_degree is None else max_degree,
        variant="equispaced")


@PETSc.Log.EventDecorator()
def get_topology(coordinates):
    r"""Get the topology for VTU output.

    :arg coordinates: The coordinates defining the mesh.
    :returns: A tuple of ``(connectivity, offsets, types)``
        :class:`OFunction`\s.
    """
    V = coordinates.function_space()

    nonLinear = not is_linear(V)
    mesh = V.mesh().topology
    cell = mesh.ufl_cell()
    values = V.cell_node_map().values
    value_shape = values.shape
    basis_dim = value_shape[1]
    offsetMap = V.cell_node_map().offset
    perm = None
    # Non-simplex cells and non-linear cells need reordering
    # Connectivity of bottom cell in extruded mesh
    if cells[cell, nonLinear] == VTK_QUADRILATERAL:
        # Quad is
        #
        # 1--3    3--2
        # |  | -> |  |
        # 0--2    0--1
        values = values[:, [0, 2, 3, 1]]
    elif cells[cell, nonLinear] == VTK_WEDGE:
        # Wedge is
        #
        #    5          5
        #   /|\        /|\
        #  / | \      / | \
        # 1----3     3----4
        # |  4 | ->  |  2 |
        # | /\ |     | /\ |
        # |/  \|     |/  \|
        # 0----2     0----1
        values = values[:, [0, 2, 4, 1, 3, 5]]
    elif cells[cell, nonLinear] == VTK_HEXAHEDRON:
        # Hexahedron is
        #
        #   5----7      7----6
        #  /|   /|     /|   /|
        # 4----6 | -> 4----5 |
        # | 1--|-3    | 3--|-2
        # |/   |/     |/   |/
        # 0----2      0----1
        values = values[:, [0, 2, 3, 1, 4, 6, 7, 5]]
    elif cells[cell, nonLinear] == VTK_LAGRANGE_TETRAHEDRON:
        perm = vtk_lagrange_tet_reorder(V.ufl_element())
        values = values[:, perm]
    elif cells[cell, nonLinear] == VTK_LAGRANGE_HEXAHEDRON:
        perm = vtk_lagrange_hex_reorder(V.ufl_element())
        values = values[:, perm]
    elif cells[cell, nonLinear] == VTK_LAGRANGE_CURVE:
        perm = vtk_lagrange_interval_reorder(V.ufl_element())
        values = values[:, perm]
    elif cells[cell, nonLinear] == VTK_LAGRANGE_TRIANGLE:
        perm = vtk_lagrange_triangle_reorder(V.ufl_element())
        values = values[:, perm]
    elif cells[cell, nonLinear] == VTK_LAGRANGE_QUADRILATERAL:
        perm = vtk_lagrange_quad_reorder(V.ufl_element())
        values = values[:, perm]
    elif cells[cell, nonLinear] == VTK_LAGRANGE_WEDGE:
        perm = vtk_lagrange_wedge_reorder(V.ufl_element())
        values = values[:, perm]
    elif cells.get((cell, nonLinear)) is None:
        # Never reached, but let's be safe.
        raise ValueError("Unhandled cell type %r" % cell)

    # Repeat up the column
    num_cells = mesh.cell_set.size
    if not mesh.cell_set._extruded:
        cell_layers = 1
        offsets = 0
    else:
        if perm is not None:
            offsetMap = offsetMap[perm]
        if mesh.variable_layers:
            layers = mesh.cell_set.layers_array[:num_cells, ...]
            cell_layers = layers[:, 1] - layers[:, 0] - 1

            def vrange(cell_layers):
                return numpy.repeat(cell_layers - cell_layers.cumsum(),
                                    cell_layers) + numpy.arange(cell_layers.sum())
            offsets = numpy.outer(vrange(cell_layers), offsetMap).astype(IntType)
            num_cells = cell_layers.sum()
        else:
            cell_layers = mesh.cell_set.layers - 1
            offsets = numpy.outer(numpy.arange(cell_layers, dtype=IntType), offsetMap)
            offsets = numpy.tile(offsets, (num_cells, 1))
            num_cells *= cell_layers
    connectivity = numpy.repeat(values, cell_layers, axis=0)
    # Add offsets going up the column
    con = connectivity + offsets
    connectivity = con.flatten()
    if not nonLinear:
        offsets_into_con = numpy.arange(start=cell.num_vertices(),
                                        stop=cell.num_vertices() * (num_cells + 1),
                                        step=cell.num_vertices(),
                                        dtype=IntType)
    else:
        offsets_into_con = numpy.arange(start=basis_dim,
                                        stop=basis_dim * (num_cells + 1),
                                        step=basis_dim,
                                        dtype=IntType)
    cell_types = numpy.full(num_cells, cells[cell, nonLinear], dtype="uint8")
    return (OFunction(connectivity, "connectivity", None),
            OFunction(offsets_into_con, "offsets", None),
            OFunction(cell_types, "types", None))


def get_byte_order(dtype):
    import sys
    native = {"little": "LittleEndian", "big": "BigEndian"}[sys.byteorder]
    return {"=": native,
            "|": "LittleEndian",
            "<": "LittleEndian",
            ">": "BigEndian"}[dtype.byteorder]


def prepare_ofunction(ofunction, real):
    array, name, _ = ofunction
    if array.dtype.kind == "c":
        if real:
            arrays = (array.real, )
            names = (name, )
        else:
            arrays = (array.real, array.imag)
            names = (name + " (real part)", name + " (imaginary part)")
    else:
        arrays = (array, )
        names = (name,)
    return arrays, names


def write_array(f, ofunction, real=False):
    arrays, _ = prepare_ofunction(ofunction, real=real)
    for array in arrays:
        numpy.uint32(array.nbytes).tofile(f)
        if get_byte_order(array.dtype) == "BigEndian":
            array = array.byteswap()
        array.tofile(f)


def write_array_descriptor(f, ofunction, offset=None, parallel=False, real=False):
    arrays, names = prepare_ofunction(ofunction, real)
    nbytes = 0
    for array, name in zip(arrays, names):
        shape = array.shape[1:]
        ncmp = {0: "",
                1: "3",
                2: "9"}[len(shape)]
        typ = {numpy.dtype("float32"): "Float32",
               numpy.dtype("float64"): "Float64",
               numpy.dtype("int32"): "Int32",
               numpy.dtype("int64"): "Int64",
               numpy.dtype("uint8"): "UInt8"}[array.dtype]
        if parallel:
            f.write(('<PDataArray Name="%s" type="%s" '
                     'NumberOfComponents="%s" />' % (name, typ, ncmp)).encode('ascii'))
        else:
            if offset is None:
                raise ValueError("Must provide offset")
            offset += nbytes
            nbytes += (4 + array.nbytes)  # 4 is for the array size (uint32)
            f.write(('<DataArray Name="%s" type="%s" '
                     'NumberOfComponents="%s" '
                     'format="appended" '
                     'offset="%d" />\n' % (name, typ, ncmp, offset)).encode('ascii'))
    return nbytes


def active_field_attributes(ofunctions):
    # select first function of each rank present as "active field"
    # and return the corresponding attributes for the (P)PointData element
    s = ''
    ranks = set()
    for ofunction in ofunctions:
        array, name, _ = ofunction
        rank = len(array.shape[1:])
        if rank in ranks:
            continue
        ranks.add(rank)
        if rank == 0:
            s += ' Scalars="%s"' % name
        elif rank == 1:
            s += ' Vectors="%s"' % name
        elif rank == 2:
            s += ' Tensors="%s"' % name
    return s.encode('ascii')


def get_vtu_name(basename, rank, size):
    if size == 1:
        return "%s.vtu" % basename
    else:
        return "%s_%s.vtu" % (basename, rank)


def get_pvtu_name(basename):
    return "%s.pvtu" % basename


def get_array(function):
    shape = function.ufl_shape
    # Despite not writing connectivity data in the halo, we need to
    # write data arrays in the halo because the cell node map for
    # owned cells can index into ghost data.
    array = function.dat.data_ro_with_halos
    if len(shape) == 0:
        pass
    elif len(shape) == 1:
        # Vectors must be padded to three components
        reshape = (-1, ) + shape
        if shape != (3, ):
            array = numpy.pad(array.reshape(reshape), ((0, 0), (0, 3 - shape[0])),
                              mode="constant")
    elif len(shape) == 2:
        # Tensors must be padded to 3x3.
        reshape = (-1, ) + shape
        if shape != (3, 3):
            array = numpy.pad(array.reshape(reshape), ((0, 0), (0, 3 - shape[0]), (0, 3 - shape[1])),
                              mode="constant")
    else:
        raise ValueError("Can't write data with shape %s" % (shape, ))
    return array


class File(object):
    _header = (b'<?xml version="1.0" ?>\n'
               b'<VTKFile type="Collection" version="0.1" '
               b'byte_order="LittleEndian">\n'
               b'<Collection>\n')
    _footer = (b'</Collection>\n'
               b'</VTKFile>\n')

    def __init__(self, filename, project_output=False, comm=None, mode="w",
                 target_degree=None, target_continuity=None, adaptive=False):
        """Create an object for outputting data for visualisation.

        This produces output in VTU format, suitable for visualisation
        with Paraview or other VTK-capable visualisation packages.


        :arg filename: The name of the output file (must end in
            ``.pvd``).
        :kwarg project_output: Should the output be projected to
            a computed output space?  Default is to use interpolation.
        :kwarg comm: The MPI communicator to use.
        :kwarg mode: "w" to overwrite any existing file, "a" to append to an existing file.
        :kwarg target_degree: override the degree of the output space.
        :kwarg target_continuity: override the continuity of the output space;
            A UFL :class:`ufl.sobolevspace.SobolevSpace` object: `H1` for a
            continuous output and `L2` for a discontinuous output.
        :kwarg adaptive: allow different meshes at different exports if `True`.

        .. note::

           Visualisation is only possible for Lagrange fields (either
           continuous or discontinuous).  All other fields are first
           either projected or interpolated to Lagrange elements
           before storing for visualisation purposes.
        """
        filename = os.path.abspath(filename)
        basename, ext = os.path.splitext(filename)
        if ext not in (".pvd", ):
            raise ValueError("Only output to PVD is supported")

        if mode not in ["w", "a"]:
            raise ValueError("Mode must be 'a' or 'w'")
        if mode == "a" and not os.path.isfile(filename):
            mode = "w"

        self.comm = comm or COMM_WORLD
        self._comm = internal_comm(self.comm, self)

        if self._comm.rank == 0 and mode == "w":
            if not os.path.exists(basename):
                os.makedirs(basename)
        elif self._comm.rank == 0 and mode == "a":
            if not os.path.exists(os.path.abspath(filename)):
                raise ValueError("Need a file to restart from.")
        self._comm.barrier()

        self.filename = filename
        self.basename = basename
        # The vtu files will be in the subdirectory named same with the pvd file.
        # Assuming the absolute path of the pvd file is "/path/to/foo.pvd", the vtu
        # files will be in the subdirectory "foo" like this:
        #
        #     /path/to/foo/foo_0.vtu
        #     /path/to/foo/foo_1.vtu
        #
        # The basename for this example is `/path/to/foo`, and the vtu_basename
        # is `/path/to/foo/foo`.
        self.vtu_basename = os.path.join(basename, os.path.basename(basename))
        self.project = project_output
        self.target_degree = target_degree
        self.target_continuity = target_continuity
        if target_degree is not None and target_degree < 0:
            raise ValueError("Invalid target_degree")
        if target_continuity is not None and target_continuity not in {ufl.H1, ufl.L2}:
            raise ValueError("target_continuity must be either 'H1' or 'L2'.")
        countstart = 0

        if self._comm.rank == 0 and mode == "w":
            with open(self.filename, "wb") as f:
                f.write(self._header)
                f.write(self._footer)
        elif self._comm.rank == 0 and mode == "a":
            import xml.etree.ElementTree as ElTree
            tree = ElTree.parse(os.path.abspath(filename))
            # Count how many the file already has
            for parent in tree.iter():
                for child in list(parent):
                    if child.tag != "DataSet":
                        continue
                    countstart += 1

        if mode == "a":
            # Need to communicate the count across all cores involved; default op is SUM
            countstart = self._comm.allreduce(countstart)

        self.counter = itertools.count(countstart)
        self.timestep = itertools.count(countstart)

        self._fnames = None
        self._topology = None
        self._adaptive = adaptive

    @no_annotations
    def _prepare_output(self, function, max_elem):
        from firedrake import FunctionSpace, VectorFunctionSpace, \
            TensorFunctionSpace, Function
        from tsfc.finatinterface import create_element as create_finat_element

        name = function.name()
        # Need to project/interpolate?
        # If space is not the max element, we must do so.
        finat_elem = function.function_space().finat_element
        if finat_elem == create_finat_element(max_elem):
            return OFunction(array=get_array(function),
                             name=name, function=function)
        #  OK, let's go and do it.
        # Build appropriate space for output function.
        shape = function.ufl_shape
        if len(shape) == 0:
            V = FunctionSpace(extract_unique_domain(function), max_elem)
        elif len(shape) == 1:
            if numpy.prod(shape) > 3:
                raise ValueError("Can't write vectors with more than 3 components")
            V = VectorFunctionSpace(extract_unique_domain(function), max_elem,
                                    dim=shape[0])
        elif len(shape) == 2:
            if numpy.prod(shape) > 9:
                raise ValueError("Can't write tensors with more than 9 components")
            V = TensorFunctionSpace(extract_unique_domain(function), max_elem,
                                    shape=shape)
        else:
            raise ValueError("Unsupported shape %s" % (shape, ))
        output = Function(V)
        if self.project:
            output.project(function)
        else:
            output.interpolate(function)

        return OFunction(array=get_array(output), name=name, function=output)

    def _write_vtu(self, *functions):
        from firedrake.function import Function

        # Check if the user has requested to write out a plain mesh
        if len(functions) == 1 and isinstance(functions[0], ufl.Mesh):
            from firedrake.functionspace import FunctionSpace
            mesh = functions[0]
            V = FunctionSpace(mesh, "CG", 1)
            functions = [Function(V)]

        for f in functions:
            if not isinstance(f, Function):
                raise ValueError("Can only output Functions or a single mesh, not %r" % type(f))
        meshes = tuple(extract_unique_domain(f) for f in functions)
        if not all(m == meshes[0] for m in meshes):
            raise ValueError("All functions must be on same mesh")

        mesh = meshes[0]
        cell = mesh.topology.ufl_cell()
        if (cell, True) not in cells and (cell, False) not in cells:
            raise ValueError("Unhandled cell type %r" % cell)

        if self._fnames is not None:
            if tuple(f.name() for f in functions) != self._fnames:
                raise ValueError("Writing different set of functions")
        else:
            self._fnames = tuple(f.name() for f in functions)
        continuous = all(is_cg(f.function_space()) for f in functions) and \
            is_cg(mesh.coordinates.function_space())
        if self.target_continuity is not None:
            continuous = self.target_continuity == ufl.H1
        # Since Points define nodes for both the mesh and function, we must
        # interpolate/project ALL involved elements onto a single larger
        # finite element.
        mesh_elem = mesh.coordinates.ufl_element()
        max_elem = get_sup_element(mesh_elem, *(f.ufl_element()
                                                for f in functions),
                                   continuous=continuous,
                                   max_degree=self.target_degree)
        coordinates = self._prepare_output(mesh.coordinates, max_elem)

        functions = tuple(self._prepare_output(f, max_elem)
                          for f in functions)

        if self._topology is None or self._adaptive:
            self._topology = get_topology(coordinates.function)

        basename = f"{self.vtu_basename}_{next(self.counter)}"

        vtu = self._write_single_vtu(basename, coordinates, *functions)

        if self.comm.size > 1:
            vtu = self._write_single_pvtu(basename, coordinates, *functions)

        return vtu

    def _write_single_vtu(self, basename,
                          coordinates,
                          *functions):
        connectivity, offsets, types = self._topology
        num_points = coordinates.array.shape[0]
        num_cells = types.array.shape[0]
        fname = get_vtu_name(basename, self.comm.rank, self.comm.size)
        with open(fname, "wb") as f:
            # Running offset for appended data
            offset = 0
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(b'<VTKFile type="UnstructuredGrid" version="2.2" '
                    b'byte_order="LittleEndian" '
                    b'header_type="UInt32">\n')
            f.write(b'<UnstructuredGrid>\n')

            f.write(('<Piece NumberOfPoints="%d" '
                     'NumberOfCells="%d">\n' % (num_points, num_cells)).encode('ascii'))
            f.write(b'<Points>\n')
            # Vertex coordinates
            offset += write_array_descriptor(f, coordinates, offset=offset, real=True)
            f.write(b'</Points>\n')

            f.write(b'<Cells>\n')
            offset += write_array_descriptor(f, connectivity, offset=offset)
            offset += write_array_descriptor(f, offsets, offset=offset)
            offset += write_array_descriptor(f, types, offset=offset)
            f.write(b'</Cells>\n')

            f.write(b'<PointData%s>\n' % active_field_attributes(functions))
            for function in functions:
                offset += write_array_descriptor(f, function, offset=offset)
            f.write(b'</PointData>\n')

            f.write(b'</Piece>\n')
            f.write(b'</UnstructuredGrid>\n')

            f.write(b'<AppendedData encoding="raw">\n')
            # Appended data must start with "_", separating whitespace
            # from data
            f.write(b'_')
            write_array(f, coordinates, real=True)
            write_array(f, connectivity)
            write_array(f, offsets)
            write_array(f, types)
            for function in functions:
                write_array(f, function)
            f.write(b'\n</AppendedData>\n')

            f.write(b'</VTKFile>\n')
        return fname

    def _write_single_pvtu(self, basename,
                           coordinates,
                           *functions):
        connectivity, offsets, types = self._topology
        fname = get_pvtu_name(basename)
        with open(fname, "wb") as f:
            f.write(b'<?xml version="1.0" ?>\n')
            f.write(b'<VTKFile type="PUnstructuredGrid" version="2.2" '
                    b'byte_order="LittleEndian">\n')
            f.write(b'<PUnstructuredGrid>\n')

            f.write(b'<PPoints>\n')
            # Vertex coordinates
            write_array_descriptor(f, coordinates, parallel=True, real=True)
            f.write(b'</PPoints>\n')

            f.write(b'<PCells>\n')
            write_array_descriptor(f, connectivity, parallel=True)
            write_array_descriptor(f, offsets, parallel=True)
            write_array_descriptor(f, types, parallel=True)
            f.write(b'</PCells>\n')

            f.write(b'<PPointData%s>\n' % active_field_attributes(functions))
            for function in functions:
                write_array_descriptor(f, function, parallel=True)
            f.write(b'</PPointData>\n')

            size = self.comm.size
            for rank in range(size):
                # need a relative path so files can be moved around:
                vtu_name = os.path.relpath(get_vtu_name(basename, rank, size),
                                           os.path.dirname(self.vtu_basename))
                f.write(('<Piece Source="%s" />\n' % vtu_name).encode('ascii'))

            f.write(b'</PUnstructuredGrid>\n')
            f.write(b'</VTKFile>\n')
        return fname

    @PETSc.Log.EventDecorator()
    def write(self, *functions, **kwargs):
        """Write functions to this :class:`File`.

        :arg functions: list of functions to write.
        :kwarg time: optional timestep value.

        You may save more than one function to the same file.
        However, all calls to :meth:`write` must use the same set of
        functions.
        """
        time = kwargs.get("time", None)
        vtu = self._write_vtu(*functions)
        if time is None:
            time = next(self.timestep)

        # Write into collection as relative path, so we can move
        # things around.
        vtu = os.path.relpath(vtu, os.path.dirname(self.basename))
        if self.comm.rank == 0:
            with open(self.filename, "r+b") as f:
                # Seek backwards from end to beginning of footer
                f.seek(-len(self._footer), 2)
                # Write new dataset name
                f.write(('<DataSet timestep="%s" '
                         'file="%s" />\n' % (time, vtu)).encode('ascii'))
                # And add footer again, so that the file is valid
                f.write(self._footer)

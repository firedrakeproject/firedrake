
import collections
import itertools
import numpy
import os
import ufl
import weakref
from pyop2.mpi import COMM_WORLD, dup_comm
from pyop2.datatypes import IntType

__all__ = ("File", )


VTK_INTERVAL = 3
VTK_TRIANGLE = 5
VTK_QUADRILATERAL = 9
VTK_TETRAHEDRON = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13

cells = {
    ufl.Cell("interval"): VTK_INTERVAL,
    ufl.Cell("triangle"): VTK_TRIANGLE,
    ufl.Cell("quadrilateral"): VTK_QUADRILATERAL,
    ufl.TensorProductCell(ufl.Cell("interval"),
                          ufl.Cell("interval")): VTK_QUADRILATERAL,
    ufl.Cell("tetrahedron"): VTK_TETRAHEDRON,
    ufl.TensorProductCell(ufl.Cell("triangle"),
                          ufl.Cell("interval")): VTK_WEDGE,
    ufl.TensorProductCell(ufl.Cell("quadrilateral"),
                          ufl.Cell("interval")): VTK_HEXAHEDRON
}


OFunction = collections.namedtuple("OFunction", ["array", "name", "function"])


def is_cg(V):
    """Is the provided space continuous?

    :arg V: A FunctionSpace.
    """
    nvertex = V.ufl_domain().ufl_cell().num_vertices()
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
    nvertex = V.ufl_domain().ufl_cell().num_vertices()
    return V.finat_element.space_dimension() == nvertex


def get_topology(coordinates):
    r"""Get the topology for VTU output.

    :arg coordinates: The coordinates defining the mesh.
    :returns: A tuple of ``(connectivity, offsets, types)``
        :class:`OFunction`\s.
    """
    V = coordinates.function_space()
    mesh = V.ufl_domain().topology
    cell = mesh.ufl_cell()
    values = V.cell_node_map().values
    # Non-simplex cells need reordering
    # Connectivity of bottom cell in extruded mesh
    if cells[cell] == VTK_QUADRILATERAL:
        # Quad is
        #
        # 1--3    3--2
        # |  | -> |  |
        # 0--2    0--1
        values = values[:, [0, 2, 3, 1]]
    elif cells[cell] == VTK_WEDGE:
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
    elif cells[cell] == VTK_HEXAHEDRON:
        # Hexahedron is
        #
        #   5----7      7----6
        #  /|   /|     /|   /|
        # 4----6 | -> 4----5 |
        # | 1--|-3    | 3--|-2
        # |/   |/     |/   |/
        # 0----2      0----1
        values = values[:, [0, 2, 3, 1, 4, 6, 7, 5]]
    elif cells.get(cell) is None:
        # Never reached, but let's be safe.
        raise ValueError("Unhandled cell type %r" % cell)

    if is_cg(V):
        scale = 1
    else:
        scale = cell.num_vertices()

    # Repeat up the column
    num_cells = mesh.cell_set.size
    if not mesh.cell_set._extruded:
        cell_layers = 1
        offsets = 0
    else:
        if mesh.variable_layers:
            layers = mesh.cell_set.layers_array[:num_cells, ...]
            cell_layers = layers[:, 1] - layers[:, 0] - 1

            def vrange(cell_layers):
                return numpy.repeat(cell_layers - cell_layers.cumsum(),
                                    cell_layers) + numpy.arange(cell_layers.sum())

            offsets = vrange(cell_layers) * scale
            offsets = offsets.reshape(-1, 1)
            num_cells = cell_layers.sum()
        else:
            cell_layers = mesh.cell_set.layers - 1
            offsets = numpy.arange(cell_layers, dtype=IntType) * scale
            offsets = numpy.tile(offsets.reshape(-1, 1), (num_cells, 1))
            num_cells *= cell_layers

    connectivity = numpy.repeat(values, cell_layers, axis=0)

    # Add offsets going up the column
    connectivity += offsets

    connectivity = connectivity.flatten()

    offsets = numpy.arange(start=cell.num_vertices(),
                           stop=cell.num_vertices() * (num_cells + 1),
                           step=cell.num_vertices(),
                           dtype=IntType)
    cell_types = numpy.full(num_cells, cells[cell], dtype="uint8")
    return (OFunction(connectivity, "connectivity", None),
            OFunction(offsets, "offsets", None),
            OFunction(cell_types, "types", None))


def get_byte_order(dtype):
    import sys
    native = {"little": "LittleEndian", "big": "BigEndian"}[sys.byteorder]
    return {"=": native,
            "|": "LittleEndian",
            "<": "LittleEndian",
            ">": "BigEndian"}[dtype.byteorder]


def write_array(f, ofunction):
    array = ofunction.array
    numpy.uint32(array.nbytes).tofile(f)
    if get_byte_order(array.dtype) == "BigEndian":
        array = array.byteswap()
    array.tofile(f)


def write_array_descriptor(f, ofunction, offset=None, parallel=False):
    array, name, _ = ofunction
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
        f.write(('<DataArray Name="%s" type="%s" '
                 'NumberOfComponents="%s" '
                 'format="appended" '
                 'offset="%d" />\n' % (name, typ, ncmp, offset)).encode('ascii'))
    return 4 + array.nbytes     # 4 is for the array size (uint32)


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

    def __init__(self, filename, project_output=False, comm=None, mode="w"):
        """Create an object for outputting data for visualisation.

        This produces output in VTU format, suitable for visualisation
        with Paraview or other VTK-capable visualisation packages.


        :arg filename: The name of the output file (must end in
            ``.pvd``).
        :kwarg project_output: Should the output be projected to
            linears?  Default is to use interpolation.
        :kwarg comm: The MPI communicator to use.
        :kwarg mode: "w" to overwrite any existing file, "a" to append to an existing file.

        .. note::

           Visualisation is only possible for linear fields (either
           continuous or discontinuous).  All other fields are first
           either projected or interpolated to linear before storing
           for visualisation purposes.
        """
        filename = os.path.abspath(filename)
        basename, ext = os.path.splitext(filename)
        if ext not in (".pvd", ):
            raise ValueError("Only output to PVD is supported")

        if mode not in ["w", "a"]:
            raise ValueError("Mode must be 'a' or 'w'")
        if mode == "a" and not os.path.isfile(filename):
            mode = "w"

        comm = dup_comm(comm or COMM_WORLD)

        if comm.rank == 0 and mode == "w":
            outdir = os.path.dirname(os.path.abspath(filename))
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        elif comm.rank == 0 and mode == "a":
            if not os.path.exists(os.path.abspath(filename)):
                raise ValueError("Need a file to restart from.")
        comm.barrier()

        self.comm = comm
        self.filename = filename
        self.basename = basename
        self.project = project_output
        countstart = 0

        if self.comm.rank == 0 and mode == "w":
            with open(self.filename, "wb") as f:
                f.write(self._header)
                f.write(self._footer)
        elif self.comm.rank == 0 and mode == "a":
            import xml.etree.ElementTree as ET
            tree = ET.parse(os.path.abspath(filename))
            # Count how many the file already has
            for parent in tree.iter():
                for child in list(parent):
                    if child.tag != "DataSet":
                        continue
                    countstart += 1

        if mode == "a":
            # Need to communicate the count across all cores involved; default op is SUM
            countstart = self.comm.allreduce(countstart)

        self.counter = itertools.count(countstart)
        self.timestep = itertools.count(countstart)

        self._fnames = None
        self._topology = None
        self._output_functions = weakref.WeakKeyDictionary()
        self._mappers = weakref.WeakKeyDictionary()

    def _prepare_output(self, function, cg):
        from firedrake import FunctionSpace, VectorFunctionSpace, \
            TensorFunctionSpace, Function, Projector, Interpolator

        name = function.name()

        # Need to project/interpolate?
        # If space is linear and continuity of output space matches
        # continuity of current space, then we can just use the
        # input function.
        if is_linear(function.function_space()) and \
           is_dg(function.function_space()) == (not cg) and \
           is_cg(function.function_space()) == cg:
            return OFunction(array=get_array(function),
                             name=name, function=function)

        # OK, let's go and do it.
        if cg:
            family = "Lagrange"
        else:
            family = "Discontinuous Lagrange"

        output = self._output_functions.get(function)
        if output is None:
            # Build appropriate space for output function.
            shape = function.ufl_shape
            if len(shape) == 0:
                V = FunctionSpace(function.ufl_domain(), family, 1)
            elif len(shape) == 1:
                if numpy.prod(shape) > 3:
                    raise ValueError("Can't write vectors with more than 3 components")
                V = VectorFunctionSpace(function.ufl_domain(), family, 1,
                                        dim=shape[0])
            elif len(shape) == 2:
                if numpy.prod(shape) > 9:
                    raise ValueError("Can't write tensors with more than 9 components")
                V = TensorFunctionSpace(function.ufl_domain(), family, 1,
                                        shape=shape)
            else:
                raise ValueError("Unsupported shape %s" % (shape, ))
            output = Function(V)
            self._output_functions[function] = output

        if self.project:
            projector = self._mappers.get(function)
            if projector is None:
                projector = Projector(function, output)
                self._mappers[function] = projector
            projector.project()
        else:
            interpolator = self._mappers.get(function)
            if interpolator is None:
                interpolator = Interpolator(function, output)
                self._mappers[function] = interpolator
            interpolator.interpolate()

        return OFunction(array=get_array(output), name=name, function=output)

    def _write_vtu(self, *functions):
        from firedrake.function import Function
        for f in functions:
            if not isinstance(f, Function):
                raise ValueError("Can only output Functions, not %r" % type(f))
        meshes = tuple(f.ufl_domain() for f in functions)
        if not all(m == meshes[0] for m in meshes):
            raise ValueError("All functions must be on same mesh")

        mesh = meshes[0]
        cell = mesh.topology.ufl_cell()
        if cell not in cells:
            raise ValueError("Unhandled cell type %r" % cell)

        if self._fnames is not None:
            if tuple(f.name() for f in functions) != self._fnames:
                raise ValueError("Writing different set of functions")
        else:
            self._fnames = tuple(f.name() for f in functions)

        continuous = all(is_cg(f.function_space()) for f in functions) and \
            is_cg(mesh.coordinates.function_space())

        coordinates = self._prepare_output(mesh.coordinates, continuous)

        functions = tuple(self._prepare_output(f, continuous)
                          for f in functions)

        if self._topology is None:
            self._topology = get_topology(coordinates.function)

        basename = "%s_%s" % (self.basename, next(self.counter))

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
            f.write(b'<VTKFile type="UnstructuredGrid" version="0.1" '
                    b'byte_order="LittleEndian" '
                    b'header_type="UInt32">\n')
            f.write(b'<UnstructuredGrid>\n')

            f.write(('<Piece NumberOfPoints="%d" '
                     'NumberOfCells="%d">\n' % (num_points, num_cells)).encode('ascii'))
            f.write(b'<Points>\n')
            # Vertex coordinates
            offset += write_array_descriptor(f, coordinates, offset=offset)
            f.write(b'</Points>\n')

            f.write(b'<Cells>\n')
            offset += write_array_descriptor(f, connectivity, offset=offset)
            offset += write_array_descriptor(f, offsets, offset=offset)
            offset += write_array_descriptor(f, types, offset=offset)
            f.write(b'</Cells>\n')

            f.write(b'<PointData>\n')
            for function in functions:
                offset += write_array_descriptor(f, function, offset=offset)
            f.write(b'</PointData>\n')

            f.write(b'</Piece>\n')
            f.write(b'</UnstructuredGrid>\n')

            f.write(b'<AppendedData encoding="raw">\n')
            # Appended data must start with "_", separating whitespace
            # from data
            f.write(b'_')
            write_array(f, coordinates)
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
            f.write(b'<VTKFile type="PUnstructuredGrid" version="0.1" '
                    b'byte_order="LittleEndian">\n')
            f.write(b'<PUnstructuredGrid>\n')

            f.write(b'<PPoints>\n')
            # Vertex coordinates
            write_array_descriptor(f, coordinates, parallel=True)
            f.write(b'</PPoints>\n')

            f.write(b'<PCells>\n')
            write_array_descriptor(f, connectivity, parallel=True)
            write_array_descriptor(f, offsets, parallel=True)
            write_array_descriptor(f, types, parallel=True)
            f.write(b'</PCells>\n')

            f.write(b'<PPointData>\n')
            for function in functions:
                write_array_descriptor(f, function, parallel=True)
            f.write(b'</PPointData>\n')

            size = self.comm.size
            for rank in range(size):
                # need a relative path so files can be moved around:
                vtu_name = os.path.relpath(get_vtu_name(basename, rank, size),
                                           os.path.dirname(self.basename))
                f.write(('<Piece Source="%s" />\n' % vtu_name).encode('ascii'))

            f.write(b'</PUnstructuredGrid>\n')
            f.write(b'</VTKFile>\n')
        return fname

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

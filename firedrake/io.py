from __future__ import absolute_import
from evtk import hl
from evtk.vtk import _get_byte_order
from evtk.hl import _requiresLargeVTKFileSize
from ufl import Cell, OuterProductCell
import numpy as np
import os

from pyop2.logger import warning, RED
from pyop2.mpi import MPI

import firedrake.functionspace as fs
import firedrake.projection as projection


__all__ = ['File']


# Dictionary used to translate the cellname of firedrake
# to the celltype of evtk module.
_cells = {}
_cells[Cell("interval")] = hl.VtkLine
_cells[Cell("interval", 2)] = hl.VtkLine
_cells[Cell("interval", 3)] = hl.VtkLine
_cells[Cell("triangle")] = hl.VtkTriangle
_cells[Cell("triangle", 3)] = hl.VtkTriangle
_cells[Cell("tetrahedron")] = hl.VtkTetra
_cells[OuterProductCell(Cell("triangle"), Cell("interval"))] = hl.VtkWedge
_cells[OuterProductCell(Cell("triangle", 3), Cell("interval"))] = hl.VtkWedge
_cells[Cell("quadrilateral")] = hl.VtkQuad
_cells[Cell("quadrilateral", 3)] = hl.VtkQuad
_cells[OuterProductCell(Cell("interval"), Cell("interval"))] = hl.VtkQuad
_cells[OuterProductCell(Cell("interval", 2), Cell("interval"))] = hl.VtkQuad
_cells[OuterProductCell(Cell("interval", 2), Cell("interval"), gdim=3)] = hl.VtkQuad
_cells[OuterProductCell(Cell("interval", 3), Cell("interval"))] = hl.VtkQuad
_cells[OuterProductCell(Cell("quadrilateral"), Cell("interval"))] = hl.VtkHexahedron
_cells[OuterProductCell(Cell("quadrilateral", 3), Cell("interval"))] = hl.VtkHexahedron

_points_per_cell = {}
_points_per_cell[Cell("interval")] = 2
_points_per_cell[Cell("interval", 2)] = 2
_points_per_cell[Cell("interval", 3)] = 2
_points_per_cell[Cell("triangle")] = 3
_points_per_cell[Cell("triangle", 3)] = 3
_points_per_cell[Cell("quadrilateral")] = 4
_points_per_cell[Cell("quadrilateral", 3)] = 4
_points_per_cell[Cell("tetrahedron")] = 4
_points_per_cell[OuterProductCell(Cell("triangle"), Cell("interval"))] = 6
_points_per_cell[OuterProductCell(Cell("triangle", 3), Cell("interval"))] = 6
_points_per_cell[OuterProductCell(Cell("interval"), Cell("interval"))] = 4
_points_per_cell[OuterProductCell(Cell("interval", 2), Cell("interval"))] = 4
_points_per_cell[OuterProductCell(Cell("interval", 2), Cell("interval"), gdim=3)] = 4
_points_per_cell[OuterProductCell(Cell("interval", 3), Cell("interval"))] = 4
_points_per_cell[OuterProductCell(Cell("quadrilateral"), Cell("interval"))] = 8
_points_per_cell[OuterProductCell(Cell("quadrilateral", 3), Cell("interval"))] = 8


class File(object):

    """A pvd file object to which :class:`~.Function`\s can be output.
    Parallel output is handled automatically.

    File output is achieved using the left shift operator:

    .. code-block:: python

      a = Function(...)
      f = File("foo.pvd")
      f << a

    .. note::

       A single :class:`File` object only supports output in a single
       function space.  The supported function spaces for output are
       CG1 or DG1; any functions which do not live in these spaces
       will automatically be projected to one or the other as
       appropriate.  The selection of which space is used for output
       in this :class:`File` depends on both the continuity of the
       coordinate field and the continuity of the output function.
       The logic for selecting the output space is as follows:

       * If the both the coordinate field and the output function are
         in :math:`H^1`, the output will be in CG1.
       * Otherwise, both the coordinate field and the output function
         will be in DG1.

    """

    def __init__(self, filename):
        # Ensure output directory exists
        outdir = os.path.dirname(os.path.abspath(filename))
        if MPI.comm.rank == 0:
            if not os.path.exists(outdir):
                os.makedirs(outdir)
        MPI.comm.barrier()
        # Parallel
        if MPI.comm.size > 1:
            new_file = os.path.splitext(os.path.abspath(filename))[0]
            # If the rank of process is 0, then create PVD file that can create
            # PVTU file and VTU file.
            if MPI.comm.rank == 0:
                self._file = _PVDFile(new_file)
            # Else, create VTU file.
            elif os.path.splitext(filename)[1] == ".pvd":
                self._file = _VTUFile(new_file)
            else:
                raise ValueError("On parallel writing, the filename written must be vtu file.")
        else:
            new_file = os.path.splitext(os.path.abspath(filename))[0]
            if os.path.splitext(filename)[1] == ".vtu":
                self._file = _VTUFile(new_file)
            elif os.path.splitext(filename)[1] == ".pvd":
                self._file = _PVDFile(new_file)
            else:
                raise ValueError("File name is wrong. It must be either vtu file or pvd file.")

    def __lshift__(self, data):
        self._file << data


class _VTUFile(object):

    """Class that represents a VTU file."""

    def __init__(self, filename, warnings=None):
        # _filename : full path to the file without extension.
        self._filename = filename
        if warnings:
            self._warnings = warnings
        else:
            self._warnings = [None, None]
        if MPI.parallel:
            self._time_step = -1
            # If _generate_time, _time_step would be incremented by
            # one everytime.
            self._generate_time = False

    def __lshift__(self, data):
        """It allows file << function syntax for writing data out to disk.

        In the case of parallel, it would also accept (function, timestep)
        tuple as an argument. If only function is given, then the timestep
        will be automatically generated."""
        # If parallel, it needs to keep track of its timestep.
        if MPI.parallel:
            # if statements to keep the consistency of how to update the
            # timestep.
            if isinstance(data, tuple):
                if self._time_step == -1 or not self._generate_time:
                    function = data[0]
                    self._time_step = data[1]
                else:
                    raise TypeError("Expected function, got tuple.")
            else:
                if self._time_step != -1 and not self._generate_time:
                    raise TypeError("Expected tuple, got function.")
                function = data
                self._time_step += 1
                self._generate_time = True
        else:
            function = data

        def is_family1(e, family):
            import ufl.finiteelement.hdivcurl as hc
            if isinstance(e, (hc.HDiv, hc.HCurl)):
                return False
            if e.family() == 'OuterProductElement':
                if e.degree() == (1, 1):
                    if e._A.family() == family \
                       and e._B.family() == family:
                        return True
            elif e.family() == family and e.degree() == 1:
                return True
            return False

        def is_cgN(e):
            import ufl.finiteelement.hdivcurl as hc
            if isinstance(e, (hc.HDiv, hc.HCurl)):
                return False
            if e.family() == 'OuterProductElement':
                if e._A.family() in ('Lagrange', 'Q') \
                   and e._B.family() == 'Lagrange':
                    return True
            elif e.family() in ('Lagrange', 'Q'):
                return True
            return False

        mesh = function.function_space().mesh()
        e = function.function_space().ufl_element()

        if len(e.value_shape()) > 1:
            raise RuntimeError("Can't output tensor valued functions")

        ce = mesh.coordinates.function_space().ufl_element()

        coords_p1 = is_family1(ce, 'Lagrange') or is_family1(ce, 'Q')
        coords_p1dg = is_family1(ce, 'Discontinuous Lagrange') or is_family1(ce, 'DQ')
        coords_cgN = is_cgN(ce)
        function_p1 = is_family1(e, 'Lagrange') or is_family1(e, 'Q')
        function_p1dg = is_family1(e, 'Discontinuous Lagrange') or is_family1(e, 'DQ')
        function_cgN = is_cgN(e)

        project_coords = False
        project_function = False
        discontinuous = False
        # We either output in P1 or P1dg.
        if coords_cgN and function_cgN:
            family = 'CG'
            project_coords = not coords_p1
            project_function = not function_p1
        else:
            family = 'DG'
            project_coords = not coords_p1dg
            project_function = not function_p1dg
            discontinuous = True

        if project_function:
            if len(e.value_shape()) == 0:
                Vo = fs.FunctionSpace(mesh, family, 1)
            elif len(e.value_shape()) == 1:
                Vo = fs.VectorFunctionSpace(mesh, family, 1, dim=e.value_shape()[0])
            else:
                # Never reached
                Vo = None
            if not self._warnings[0]:
                warning(RED % "*** Projecting output function to %s1", family)
                self._warnings[0] = True
            output = projection.project(function, Vo, name=function.name())
        else:
            output = function
            Vo = output.function_space()
        if project_coords:
            Vc = fs.VectorFunctionSpace(mesh, family, 1, dim=mesh.coordinates.function_space().dim)
            if not self._warnings[1]:
                warning(RED % "*** Projecting coordinates to %s1", family)
                self._warnings[1] = True
            coordinates = projection.project(mesh.coordinates, Vc, name=mesh.coordinates.name())
        else:
            coordinates = mesh.coordinates
            Vc = coordinates.function_space()

        num_points = Vo.node_count

        layers = mesh.layers - 1 if mesh.layers else 1
        num_cells = mesh.num_cells() * layers

        if not isinstance(e.cell(), OuterProductCell) and e.cell().cellname() != "quadrilateral":
            connectivity = Vc.cell_node_map().values_with_halo.flatten()
        else:
            # Connectivity of bottom cell in extruded mesh
            base = Vc.cell_node_map().values_with_halo
            if _cells[mesh.ufl_cell()] == hl.VtkQuad:
                # Quad is
                #
                # 1--3
                # |  |
                # 0--2
                #
                # needs to be
                #
                # 3--2
                # |  |
                # 0--1
                base = base[:, [0, 2, 3, 1]]
                points_per_cell = 4
            elif _cells[mesh.ufl_cell()] == hl.VtkWedge:
                # Wedge is
                #
                #    5
                #   /|\
                #  / | \
                # 1----3
                # |  4 |
                # | /\ |
                # |/  \|
                # 0----2
                #
                # needs to be
                #
                #    5
                #   /|\
                #  / | \
                # 3----4
                # |  2 |
                # | /\ |
                # |/  \|
                # 0----1
                #
                base = base[:, [0, 2, 4, 1, 3, 5]]
                points_per_cell = 6
            elif _cells[mesh.ufl_cell()] == hl.VtkHexahedron:
                # Hexahedron is
                #
                #   5----7
                #  /|   /|
                # 4----6 |
                # | 1--|-3
                # |/   |/
                # 0----2
                #
                # needs to be
                #
                #   7----6
                #  /|   /|
                # 4----5 |
                # | 3--|-2
                # |/   |/
                # 0----1
                #
                base = base[:, [0, 2, 3, 1, 4, 6, 7, 5]]
                points_per_cell = 8
            # Repeat up the column
            connectivity_temp = np.repeat(base, layers, axis=0)

            if discontinuous:
                scale = points_per_cell
            else:
                scale = 1
            offsets = np.arange(layers) * scale

            # Add offsets going up the column
            connectivity_temp += np.tile(offsets.reshape(-1, 1), (mesh.num_cells(), 1))

            connectivity = connectivity_temp.flatten()

        if isinstance(output.function_space(), fs.VectorFunctionSpace):
            tmp = output.dat.data_ro_with_halos
            vdata = [None]*3
            if output.dat.dim[0] == 1:
                vdata[0] = tmp.flatten()
            else:
                for i in range(output.dat.dim[0]):
                    vdata[i] = tmp[:, i].flatten()
            for i in range(output.dat.dim[0], 3):
                vdata[i] = np.zeros_like(vdata[0])
            data = tuple(vdata)
            # only for checking large file size
            flat_data = {function.name(): tmp.flatten()}
        else:
            data = output.dat.data_ro_with_halos.flatten()
            flat_data = {function.name(): data}

        coordinates = self._fd_to_evtk_coord(coordinates.dat.data_ro_with_halos)

        cell_types = np.empty(num_cells, dtype="uint8")

        # Assume that all cells are of same shape.
        cell_types[:] = _cells[mesh.ufl_cell()].tid
        p_c = _points_per_cell[mesh.ufl_cell()]

        # This tells which are the last nodes of each cell.
        offsets = np.arange(start=p_c, stop=p_c * (num_cells + 1), step=p_c,
                            dtype='int32')
        large_file_flag = _requiresLargeVTKFileSize("VtkUnstructuredGrid",
                                                    numPoints=num_points,
                                                    numCells=num_cells,
                                                    pointData=flat_data,
                                                    cellData=None)
        new_name = self._filename

        # When vtu file makes part of a parallel process, aggregated by a
        # pvtu file, the output is : filename_timestep_rank.vtu
        if MPI.parallel:
            new_name += "_" + str(self._time_step) + "_" + str(MPI.comm.rank)

        self._writer = hl.VtkFile(
            new_name, hl.VtkUnstructuredGrid, large_file_flag)

        self._writer.openGrid()

        self._writer.openPiece(ncells=num_cells, npoints=num_points)

        # openElement allows the stuff in side of the tag <arg></arg>
        # to be editted.
        self._writer.openElement("Points")
        # addData adds the DataArray in the tag <arg1>
        self._writer.addData("Points", coordinates)

        self._writer.closeElement("Points")
        self._writer.openElement("Cells")
        self._writer.addData("connectivity", connectivity)
        self._writer.addData("offsets", offsets)
        self._writer.addData("types", cell_types)
        self._writer.closeElement("Cells")

        self._writer.openData("Point", scalars=function.name())
        self._writer.addData(function.name(), data)
        self._writer.closeData("Point")
        self._writer.closePiece()
        self._writer.closeGrid()

        # Create the AppendedData
        self._writer.appendData(coordinates)
        self._writer.appendData(connectivity)
        self._writer.appendData(offsets)
        self._writer.appendData(cell_types)
        self._writer.appendData(data)
        self._writer.save()

    def _fd_to_evtk_coord(self, fdcoord):
        """In firedrake function, the coordinates are represented by the
        array."""
        if len(fdcoord.shape) == 1:
            # 1D case.
            return (fdcoord,
                    np.zeros(fdcoord.shape[0]),
                    np.zeros(fdcoord.shape[0]))
        if len(fdcoord[0]) == 3:
            return (fdcoord[:, 0].ravel(),
                    fdcoord[:, 1].ravel(),
                    fdcoord[:, 2].ravel())
        else:
            return (fdcoord[:, 0].ravel(),
                    fdcoord[:, 1].ravel(),
                    np.zeros(fdcoord.shape[0]))


class _PVTUFile(object):

    """Class that represents PVTU file."""

    def __init__(self, filename):
        # filename is full path to the file without the extension.
        # eg: /home/dir/dir1/filename
        self._filename = filename
        self._writer = PVTUWriter(self._filename)

    def __del__(self):
        self._writer.save()

    def _update(self, function):
        """Add all the vtu to be added to pvtu file."""
        for i in xrange(0, MPI.comm.size):
            new_vtk_name = os.path.splitext(
                self._filename)[0] + "_" + str(i) + ".vtu"
            self._writer.addFile(new_vtk_name, function)


class PVTUWriter(object):

    """Class that is responsible for writing the PVTU file."""

    def __init__(self, filename):
        self.xml = hl.XmlWriter(filename + ".pvtu")
        self.root = os.path.dirname(filename)
        self.xml.openElement("VTKFile")
        self.xml.addAttributes(type="PUnstructuredGrid", version="0.1",
                               byte_order=_get_byte_order())
        self.xml.openElement("PUnstructuredGrid")
        self._initialised = False

    def save(self):
        """Close up the File by completing the tag."""
        self.xml.closeElement("PUnstructuredGrid")
        self.xml.closeElement("VTKFile")

    def addFile(self, filepath, function):
        """Add VTU files to the PVTU file given in the filepath. For now, the
        attributes in vtu is assumed e.g. connectivity, offsets."""

        # I think I can improve this part by creating PVTU file
        # from VTU file, passing the dictionary of
        # {attribute_name : (data type, number of components)}
        # but for now it is quite pointless since writing vtu
        # is not dynamic either.

        assert filepath[-4:] == ".vtu"
        if not self._initialised:

            self.xml.openElement("PPointData")
            if len(function.shape()) == 1:
                self.addData("Float64", function.name(), num_of_components=3)
            elif len(function.shape()) == 0:
                self.addData("Float64", function.name(), num_of_components=1)
            else:
                raise RuntimeError("Don't know how to write data with shape %s\n",
                                   function.shape())
            self.xml.closeElement("PPointData")
            self.xml.openElement("PCellData")
            self.addData("Int32", "connectivity")
            self.addData("Int32", "offsets")
            self.addData("UInt8", "types")
            self.xml.closeElement("PCellData")
            self.xml.openElement("PPoints")
            self.addData("Float64", "Points", 3)
            self.xml.closeElement("PPoints")
            self._initialised = True
        vtu_name = os.path.relpath(filepath, start=self.root)
        self.xml.stream.write('<Piece Source="%s"/>\n' % vtu_name)

    def addData(self, dtype, name, num_of_components=1):
        """Adds data array description of PDataArray. The header is as follows:
        <PDataArray type="dtype" Name="name"
        NumberOfComponents=num_of_components/>"""
        self.xml.openElement("PDataArray")
        self.xml.addAttributes(type=dtype, Name=name,
                               NumberOfComponents=num_of_components)
        self.xml.closeElement("PDataArray")


class _PVDFile(object):

    """Class that represents PVD file."""

    def __init__(self, filename):
        # Full path to the file without extension.
        self._filename = filename
        self._writer = hl.VtkGroup(self._filename)
        self._warnings = [False, False]
        # Keep the index of child file
        # (parallel -> pvtu, else vtu)
        self._child_index = 0
        self._time_step = -1
        # _generate_time -> This file does not accept (function, time) tuple
        #                   for __lshift__, and it generates the integer
        #                   time step by itself instead.
        self._generate_time = False

    def __lshift__(self, data):
        if isinstance(data, tuple):
            if self._time_step == -1 or not self._generate_time:
                self._time_step = data[1]
                self._update_PVD(data[0])
            else:
                raise TypeError(
                    "You cannot start setting the time by giving a tuple.")
        else:
            if self._time_step == -1:
                self._generate_time = True
            if self._generate_time:
                self._time_step += 1
                self._update_PVD(data)
            else:
                raise TypeError("You need to provide time stamp")

    def __del__(self):
        self._writer.save()

    def _update_PVD(self, function):
        """Update a pvd file.

        * In parallel: create a vtu file and update it with the function given.
          Then it will create a pvtu file that includes all the vtu file
          produced in the parallel writing.
        * In serial: a VTU file is created and is added to PVD file."""
        if not MPI.parallel:
            new_vtk_name = self._filename + "_" + str(self._child_index)
            new_vtk = _VTUFile(new_vtk_name, warnings=self._warnings)

            new_vtk << function
            self._writer.addFile(new_vtk_name + ".vtu", self._time_step)
            self._child_index += 1

        else:
            new_pvtu_name = self._filename + "_" + str(self._time_step)
            new_vtk = _VTUFile(self._filename, warnings=self._warnings)
            new_pvtu = _PVTUFile(new_pvtu_name)
            # The new_vtk object has its timestep initialised to -1 each time,
            # so we need to provide the timestep ourselves here otherwise
            # the VTU of timestep 0 (belonging to the process with rank 0)
            # will be over-written each time _update_PVD is called.
            new_vtk << (function, self._time_step)
            new_pvtu._update(function)
            self._writer.addFile(new_pvtu_name + ".pvtu", self._time_step)

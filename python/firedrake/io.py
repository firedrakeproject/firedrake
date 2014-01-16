from evtk.hl import *
from evtk.vtk import _get_byte_order
from evtk.hl import _requiresLargeVTKFileSize
from ufl import Cell, OuterProductCell
import numpy as np
import os
from pyop2.mpi import *
import core_types


# Dictionary used to translate the cellname of firedrake
# to the celltype of evtk module.
_cells = {}
_cells[Cell("interval")] = VtkLine
_cells[Cell("interval", 2)] = VtkLine
_cells[Cell("interval", 3)] = VtkLine
_cells[Cell("triangle")] = VtkTriangle
_cells[Cell("triangle", 3)] = VtkTriangle
_cells[Cell("tetrahedron")] = VtkTetra
_cells[OuterProductCell(Cell("triangle"), Cell("interval"))] = VtkWedge

_points_per_cell = {}
_points_per_cell[Cell("interval")] = 2
_points_per_cell[Cell("interval", 2)] = 2
_points_per_cell[Cell("interval", 3)] = 2
_points_per_cell[Cell("triangle")] = 3
_points_per_cell[Cell("triangle", 3)] = 3
_points_per_cell[Cell("tetrahedron")] = 4
_points_per_cell[OuterProductCell(Cell("triangle"), Cell("interval"))] = 6


class File(object):

    """Any file can be declared with ``f = File("filename"")``,
    then it will be directed to the correct class according to
    its extensions and also to the parallelism.

    If there is a parallel, and the rank of its process is 0,
    then :class:`_PVDFile` is created, and this takes care of writing
    :class:`_PVTUFile` , and :class:`VTUFile` that the process is
    meant to write. If rank is not 0, then :class:`_VTUFile` must be
    created. The expected argument for this class is string file name
    of a pvd file in the case of parallel writing.

    When there is no parallelism, the class will be created solely
    according to the extension of the filename passed as an argument."""

    def __init__(self, filename):
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
                raise ValueError("On parallel writing, the filename written\
                        must be vtu file.")
        else:
            new_file = os.path.splitext(os.path.abspath(filename))[0]
            if os.path.splitext(filename)[1] == ".vtu":
                self._file = _VTUFile(new_file)
            elif os.path.splitext(filename)[1] == ".pvd":
                self._file = _PVDFile(new_file)
            else:
                raise ValueError("File name is wrong. It must be either vtu\
                        file or pvd file.")

    def __lshift__(self, data):
        self._file << data


class _VTUFile(object):

    """Class that represents a VTU file."""

    def __init__(self, filename):
        #_filename : full path to the file without extension.
        self._filename = filename
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
        else:
            function = data

        # We have not supported the case where the degree of finite element
        # is not 1.

        e = function.function_space().ufl_element()
        # finite element supported : CG1 and DG0
        if not ((e.family() == "Lagrange" and e.degree() == 1) or
                (e.family() == "Discontinuous Lagrange" and e.degree() == 0) or
                (e.family() == "OuterProductElement")):
            raise ValueError("ufl element given is not supported.")
        mesh = function.function_space().mesh()
        if not (e.family() == "OuterProductElement"):
            num_points = mesh._entities[0]
            num_cells = mesh._entities[-1]
        else:
            num_points = mesh._entities[0] * mesh.layers
            num_cells = mesh._entities[-1] * (mesh.layers - 1)

        # In firedrake, the indexing of the points starts from 1, instead of 0.
        # However paraview expects them to have indexing starting from 0.

        if not (e.family() == "OuterProductElement"):
            connectivity = mesh._cells.flatten() - 1
        else:
            # Horribly slow, hard-coded first attempt
            connectivity_temp = np.empty(
                [mesh.num_cells(), mesh.layers - 1, _points_per_cell[mesh._ufl_cell]], dtype="int32")
            for (ii, verts) in enumerate(mesh._cells):
                # number of vert cells = layers - 1
                for nl in range(mesh.layers - 1):
                    connectivity_temp[ii, nl] = np.concatenate(
                        (mesh.layers * (mesh._cells[ii] - 1) + nl, mesh.layers * (mesh._cells[ii] - 1) + nl + 1))
            connectivity = connectivity_temp.flatten()  # no need to subtract 1

        if isinstance(function.function_space(), core_types.VectorFunctionSpace):
            tmp = function.dat.data_ro_with_halos
            vdata = [None]*3
            for i in range(function.dat.dim[0]):
                vdata[i] = tmp[:, i].flatten()
            for i in range(function.dat.dim[0], 3):
                vdata[i] = np.zeros_like(vdata[0])
            data = tuple(vdata)
            # only for checking large file size
            flat_data = {function.name(): tmp.flatten()}
        else:
            data = function.dat.data_ro_with_halos.flatten()
            flat_data = {function.name(): data}

        coordinates = self._fd_to_evtk_coord(mesh._coordinates)

        cell_types = np.empty(num_cells, dtype="uint8")

        # Assume that all cells are of same shape.
        cell_types[:] = _cells[mesh._ufl_cell].tid
        p_c = _points_per_cell[mesh._ufl_cell]

        # This tells which are the last nodes of each cell.
        offsets = np.arange(start=p_c, stop=p_c * (num_cells + 1), step=p_c,
                            dtype='int32')
        if (e.family() == "Lagrange" and e.degree() == 1):     # if CG1
            large_file_flag = _requiresLargeVTKFileSize("VtkUnstructuredGrid",
                                                        numPoints=num_points,
                                                        numCells=num_cells,
                                                        pointData=flat_data,
                                                        cellData=None)
        # if DG0
        elif (e.family() == "Discontinuous Lagrange" and e.degree() == 0):
            large_file_flag = _requiresLargeVTKFileSize("VtkUnstructuredGrid",
                                                        numPoints=num_points,
                                                        numCells=num_cells,
                                                        pointData=None,
                                                        cellData=flat_data)
        elif (e.family() == "OuterProductElement" and
              e._A.family() == "Lagrange" and e._A.degree() == 1 and
              e._B.family() == "Lagrange" and e._B.degree() == 1):
            large_file_flag = _requiresLargeVTKFileSize("VtkUnstructuredGrid",
                                                        numPoints=num_points,
                                                        numCells=num_cells,
                                                        pointData=flat_data,
                                                        cellData=None)

        elif (e.family() == "OuterProductElement" and
              e._A.family() == "Discontinuous Lagrange" and
              e._A.degree() == 0 and
              e._B.family() == "Discontinuous Lagrange" and
              e._B.degree() == 0):
            large_file_flag = _requiresLargeVTKFileSize("VtkUnstructuredGrid",
                                                        numPoints=num_points,
                                                        numCells=num_cells,
                                                        pointData=None,
                                                        cellData=flat_data)
        else:
            raise ValueError("Only P1, P0, P1xP1, P0xP0 are supported.")

        new_name = self._filename

        # When vtu file makes part of a parallel process, aggregated by a
        # pvtu file, the output is : filename_timestep_rank.vtu
        if MPI.parallel:
            new_name += "_" + str(self._time_step) + "_" + str(MPI.comm.rank)

        self._writer = VtkFile(
            new_name, VtkUnstructuredGrid, large_file_flag)

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

        # CG1
        if e.degree() == 1:
            self._writer.openData("Point", scalars=function.name())
            self._writer.addData(function.name(), data)
            self._writer.closeData("Point")
        # DG0
        else:
            self._writer.openData("Cell", scalars=function.name())
            self._writer.addData(function.name(), data)
            self._writer.closeData("Cell")
        self._writer.closePiece()
        self._writer.closeGrid()

        # Create the AppendedData
        self._writer.appendData(coordinates)
        self._writer.appendData(connectivity)
        self._writer.appendData(offsets)
        self._writer.appendData(cell_types)
        # CG1 or DG0
        self._writer.appendData(data)
        self._writer.save()

    def _fd_to_evtk_coord(self, fdcoord):
        """In firedrake function, the coordinates are represented by the
        array."""
        if  len(fdcoord.shape) == 1:
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

    def _update(self, name):
        """Add all the vtu to be added to pvtu file."""
        for i in xrange(0, MPI.comm.size):
            new_vtk_name = os.path.splitext(
                self._filename)[0] + "_" + str(i) + ".vtu"
            self._writer.addFile(new_vtk_name, name)


class PVTUWriter(object):

    """Class that is responsible for writing the PVTU file."""

    def __init__(self, filename):
        self.xml = XmlWriter(filename + ".pvtu")
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

    def addFile(self, filepath, name):
        """Add VTU files to the PVTU file given in the filepath. For now, the
        attributes in vtu is assumed e.g. connectivity, offsets."""

        # I think I can improve this part by creating PVTU file
        # from VTU file, passing the dictionary of
        #{attribute_name : (data type, number of components)}
        # but for now it is quite pointless since writing vtu
        # is not dynamic either.

        assert filepath[-4:] == ".vtu"
        if not self._initialised:

            self.xml.openElement("PPointData")
            self.addData("Float64", name)
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
        self._writer = VtkGroup(self._filename)
        # Keep the index of child file
        #(parallel -> pvtu, else vtu)
        self._child_index = 0
        self._time_step = -1
        #_generate_time -> This file does not accept (function, time) tuple
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
          produced in the paralell writing.
        * In serial: a VTU file is created and is added to PVD file."""
        if not MPI.parallel:
            new_vtk_name = self._filename + "_" + str(self._child_index)
            new_vtk = _VTUFile(new_vtk_name)

            new_vtk << function
            self._writer.addFile(new_vtk_name + ".vtu", self._time_step)
            self._child_index += 1

        else:
            new_pvtu_name = self._filename + "_" + str(self._time_step)
            new_vtk = _VTUFile(self._filename)
            new_pvtu = _PVTUFile(new_pvtu_name)
            new_vtk << function
            new_pvtu._update(function.name())
            self._writer.addFile(new_pvtu_name + ".pvtu", self._time_step)

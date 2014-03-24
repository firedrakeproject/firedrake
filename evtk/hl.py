# ************************************************************************
# * Copyright 2010 - 2012 Paulo A. Herrera. All rights reserved.                    *
# *                                                                                 *
# * Redistribution and use in source and binary forms, with or without              *
# * modification, are permitted provided that the following conditions are met:     *
# *                                                                                 *
# *  1. Redistributions of source code must retain the above copyright notice,      *
# *  this list of conditions and the following disclaimer.                          *
# *                                                                                 *
# *  2. Redistributions in binary form must reproduce the above copyright notice,   *
# *  this list of conditions and the following disclaimer in the documentation      *
# *  and/or other materials provided with the distribution.                         *
# *                                                                                 *
# * THIS SOFTWARE IS PROVIDED BY PAULO A. HERRERA ``AS IS'' AND ANY EXPRESS OR      *
# * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    *
# * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO      *
# * EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,        *
# * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,  *
# * BUT NOT LIMITED TO, PROCUREMEN OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    *
# * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY           *
# * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  *
# * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS              *
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                    *
# ************************************************************************

# **************************************
# *  High level Python library to      *
# *  export data to binary VTK file.   *
# **************************************

from vtk import *  # VtkFile, VtkUnstructuredGrid, etc.
import numpy as np

# =================================
#       Helper functions
# =================================


def _addDataToFile(vtkFile, cellData, pointData):
    # Point data
    if pointData <> None:
        keys = pointData.keys()
        vtkFile.openData("Point", scalars=keys[0])
        for key in keys:
            data = pointData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Point")

    # Cell data
    if cellData <> None:
        keys = cellData.keys()
        vtkFile.openData("Cell", scalars=keys[0])
        for key in keys:
            data = cellData[key]
            vtkFile.addData(key, data)
        vtkFile.closeData("Cell")


def _appendDataToFile(vtkFile, cellData, pointData):
    # Append data to binary section
    if pointData <> None:
        keys = pointData.keys()
        for key in keys:
            data = pointData[key]
            vtkFile.appendData(data)

    if cellData <> None:
        keys = cellData.keys()
        for key in keys:
            data = cellData[key]
            vtkFile.appendData(data)


def _requiresLargeVTKFileSize(vtkFileType, numPoints, numCells, pointData, cellData):
    """
        If the block size will be close to or larger than what a uint32 can hold we
        need to use VTK's large file ability. note that this is only
        available in VTK 6.0 and greater.
        Contributed by Andy Bauer.

        PARAMETERS:
            vtkFileType: name of the type of grid. If it is
                         "VtkUnstructuredGrid" or VtkStructuredGrid"
                         then it will add extra space needed
                         for storing points. If it is "VtkUnstructuredGrid" it wil
                         also add extra space for cell connectivities.
            numPoints: the number of points in the data set. It is only used
                       if vtkFileType is "VtkUnstructuredGrid" or VtkStructuredGrid".
            numCells: the number of cells in the data set. It is only used
                      if vtkFileType is "VtkUnstructuredGrid".
            cellData: dictionary containing arrays with cell centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have the same dimensions in all directions and must contain
                      only scalar data.
            pointData: dictionary containing arrays with node centered data.
                       Keys should be the names of the data arrays.
                       Arrays must have same dimension in each direction and
                       they should be equal to the dimensions of the cell data plus one and
                       must contain only scalar data.

        RETURNS:
            True if the large file format should be used.

    """
    sum = 0
    if pointData <> None:
        keys = pointData.keys()
        for key in keys:
            data = pointData[key]
            sum = sum + data.size * data.dtype.itemsize

    if cellData <> None:
        keys = cellData.keys()
        for key in keys:
            data = cellData[key]
            sum = sum + data.size * data.dtype.itemsize

    if vtkFileType == "VtkUnstructuredGrid":
        sum = sum + numPoints * 3 * 8 + numCells * 8 * 8
    elif vtkFileType == "VtkStructuredGrid":
        sum = sum + numPoints * 3 * 8

    # leave a little wiggle room instead of using 4294967295
    return sum > 4000000000

# =================================
#       High level functions
# =================================


def imageToVTK(path, origin=(0.0, 0.0, 0.0), spacing = (1.0, 1.0, 1.0), cellData = None, pointData = None):
    """ Exports data values as a rectangular image.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            origin: grid origin (default = (0,0,0))
            spacing: grid spacing (default = (1,1,1))
            cellData: dictionary containing arrays with cell centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have the same dimensions in all directions and must contain
                      only scalar data.
            nodeData: dictionary containing arrays with node centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have same dimension in each direction and
                      they should be equal to the dimensions of the cell data plus one and
                      must contain only scalar data.

         RETURNS:
            Full path to saved file.

        NOTE: At least, cellData or pointData must be present to infer the dimensions of the image.
    """
    assert (cellData <> None or pointData <> None)

    # Extract dimensions
    start = (0, 0, 0)
    end = None
    if cellData <> None:
        keys = cellData.keys()
        data = cellData[keys[0]]
        end = data.shape
    elif pointData <> None:
        keys = pointData.keys()
        data = pointData[keys[0]]
        end = data.shape
        end = (end[0] - 1, end[1] - 1, end[2] - 1)

    largeFileFlag = _requiresLargeVTKFileSize(
        "VtkImageData", numPoints=0, numCells=0, pointData=pointData, cellData=cellData)

    # Write data to file
    w = VtkFile(path, VtkImageData, largeFileFlag)
    w.openGrid(start=start, end=end, origin=origin, spacing=spacing)
    w.openPiece(start=start, end=end)
    _addDataToFile(w, cellData, pointData)
    w.closePiece()
    w.closeGrid()
    _appendDataToFile(w, cellData, pointData)
    w.save()
    return w.getFileName()

# ========================================================================


def gridToVTK(path, x, y, z, cellData=None, pointData=None):
    """
        Writes data values as a rectilinear or rectangular grid.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: coordinates of the nodes of the grid. They can be 1D or 3D depending if
                     the grid should be saved as a rectilinear or logically structured grid, respectively.
                     Arrays should contain coordinates of the nodes of the grid.
                     If arrays are 1D, then the grid should be Cartesian, i.e. faces in all cells are orthogonal.
                     If arrays are 3D, then the grid should be logically structured with hexahedral cells.
                     In both cases the arrays dimensions should be equal to the number of nodes of the grid.
            cellData: dictionary containing arrays with cell centered data.
                      Keys should be the names of the data arrays.
                      Arrays must have the same dimensions in all directions and must contain
                      only scalar data.
            pointData: dictionary containing arrays with node centered data.
                       Keys should be the names of the data arrays.
                       Arrays must have same dimension in each direction and
                       they should be equal to the dimensions of the cell data plus one and
                       must contain only scalar data.

        RETURNS:
            Full path to saved file.

    """
    # Extract dimensions
    start = (0, 0, 0)
    nx = ny = nz = 0

    if (x.ndim == 1 and y.ndim == 1 and z.ndim == 1):
        nx, ny, nz = x.size - 1, y.size - 1, z.size - 1
        isRect = True
        ftype = VtkRectilinearGrid
        largeFileFlag = _requiresLargeVTKFileSize(
            "VtkRectilinearGrid", 0, 0, pointData, cellData)
    elif (x.ndim == 3 and y.ndim == 3 and z.ndim == 3):
        s = x.shape
        nx, ny, nz = s[0] - 1, s[1] - 1, s[2] - 1
        isRect = False
        ftype = VtkStructuredGrid
        largeFileFlag = _requiresLargeVTKFileSize(
            "VtkStructuredGrid", numPoints=x.size, numCells=0, pointData=pointData, cellData=cellData)
    else:
        assert(False)
    end = (nx, ny, nz)

    w = VtkFile(path, ftype, largeFileFlag)
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)

    if isRect:
        w.openElement("Coordinates")
        w.addData("x_coordinates", x)
        w.addData("y_coordinates", y)
        w.addData("z_coordinates", z)
        w.closeElement("Coordinates")
    else:
        w.openElement("Points")
        w.addData("points", (x, y, z))
        w.closeElement("Points")

    _addDataToFile(w, cellData, pointData)
    w.closePiece()
    w.closeGrid()
    # Write coordinates
    if isRect:
        w.appendData(x).appendData(y).appendData(z)
    else:
        w.appendData((x, y, z))
    # Write data
    _appendDataToFile(w, cellData, pointData)
    w.save()
    return w.getFileName()


# ========================================================================
def pointsToVTK(path, x, y, z, data):
    """
        Export points and associated data as an unstructured grid.

        PARAMETERS:
            path: name of the file without extension where data should be saved.
            x, y, z: 1D arrays with coordinates of the points.
            data: dictionary with variables associated to each point.
                  Keys should be the names of the variable stored in each array.
                  All arrays must have the same number of elements.

        RETURNS:
            Full path to saved file.

    """
    assert (x.size == y.size == z.size)
    npoints = x.size

    # create some temporary arrays to write grid topology
    # index of last node in each cell
    offsets = np.arange(start=1, stop=npoints + 1, dtype='int32')
    # each point is only connected to itself
    connectivity = np.arange(npoints, dtype='int32')
    cell_types = np.empty(npoints, dtype='uint8')

    cell_types[:] = VtkVertex.tid

    largeFileFlag = _requiresLargeVTKFileSize(
        "VtkUnstructuredGrid", numPoints=npoints, numCells=npoints, pointData=data, cellData=None)

    w = VtkFile(path, VtkUnstructuredGrid, largeFileFlag)
    w.openGrid()
    w.openPiece(ncells=npoints, npoints=npoints)

    w.openElement("Points")
    w.addData("points", (x, y, z))
    w.closeElement("Points")
    w.openElement("Cells")
    w.addData("connectivity", connectivity)
    w.addData("offsets", offsets)
    w.addData("types", cell_types)
    w.closeElement("Cells")

    _addDataToFile(w, cellData=None, pointData=data)

    w.closePiece()
    w.closeGrid()
    w.appendData((x, y, z))
    w.appendData(connectivity).appendData(offsets).appendData(cell_types)

    _appendDataToFile(w, cellData=None, pointData=data)

    w.save()
    return w.getFileName()

#!/usr/bin/env python

import math
import sys
import numpy
import vtk

# All returned arrays are cast into either numpy or numarray arrays
arr = numpy.array


class vtu:

    """Unstructured grid object to deal with VTK unstructured grids."""

    def __init__(self, filename=None):
        """Creates a vtu object by reading the specified file."""
        if filename is None:
            self.ugrid = vtk.vtkUnstructuredGrid()
        else:
            self.gridreader = None
            if filename[-4:] == ".vtu":
                self.gridreader = vtk.vtkXMLUnstructuredGridReader()
            elif filename[-5:] == ".pvtu":
                self.gridreader = vtk.vtkXMLPUnstructuredGridReader()
            else:
                raise Exception(
                    "ERROR: don't recognise file extension" + filename)
            self.gridreader.SetFileName(filename)
            self.gridreader.Update()
            self.ugrid = self.gridreader.GetOutput()
            if self.ugrid.GetNumberOfPoints() + self.ugrid.GetNumberOfCells() == 0:
                raise Exception("No points or cells found after loading vtu " + filename)
        self.filename = filename

    def GetScalarField(self, name):
        """Returns an array with the values of the specified scalar field."""
        try:
            pointdata = self.ugrid.GetPointData()
            vtkdata = pointdata.GetScalars(name)
            vtkdata.GetNumberOfTuples()
        except:
            try:
                celldata = self.ugrid.GetCellData()
                vtkdata = celldata.GetScalars(name)
                vtkdata.GetNumberOfTuples()
            except:
                raise Exception("Couldn't find point or cell scalar field data \
                                with name " + name + " in file " + self.filename + ".")
        return arr([vtkdata.GetTuple1(i)
                    for i in range(vtkdata.GetNumberOfTuples())])

    def GetScalarRange(self, name):
        """Returns the range (min, max) of the specified scalar field."""
        try:
            pointdata = self.ugrid.GetPointData()
            vtkdata = pointdata.GetScalars(name)
            vtkdata.GetRange()
        except:
            try:
                celldata = self.ugrid.GetCellData()
                vtkdata = celldata.GetScalars(name)
                vtkdata.GetRange()
            except:
                raise Exception("Couldn't find point or cell scalar field data \
                                with name " + name + " in file " + self.filename + ".")
        return vtkdata.GetRange()

    def GetVectorField(self, name):
        """Returns an array with the values of the specified vector field."""
        try:
            pointdata = self.ugrid.GetPointData()
            vtkdata = pointdata.GetScalars(name)
            vtkdata.GetNumberOfTuples()
        except:
            try:
                celldata = self.ugrid.GetCellData()
                vtkdata = celldata.GetScalars(name)
                vtkdata.GetNumberOfTuples()
            except:
                raise Exception("Couldn't find point or cell vector field data \
                                with name " + name + " in file " + self.filename + ".")
        return arr([vtkdata.GetTuple3(i)
                    for i in range(vtkdata.GetNumberOfTuples())])

    def GetVectorNorm(self, name):
        """Return the field with the norm of the specified vector field."""
        v = self.GetVectorField(name)
        n = []

        try:
            from scipy.linalg import norm
        except ImportError:
            def norm(v):
                r = 0.0
                for x in v:
                    r = r + x ** 2
                r = math.sqrt(r)
                return r

        for node in range(self.ugrid.GetNumberOfPoints()):
            n.append(norm(v[node]))

        return arr(n)

    def GetField(self, name):
        """Returns an array with the values of the specified field."""
        try:
            pointdata = self.ugrid.GetPointData()
            vtkdata = pointdata.GetArray(name)
            vtkdata.GetNumberOfTuples()
        except:
            try:
                celldata = self.ugrid.GetCellData()
                vtkdata = celldata.GetArray(name)
                vtkdata.GetNumberOfTuples()
            except:
                raise Exception("Couldn't find point or cell field data with \
                                name " + name + " in file " + self.filename + ".")
        nc = vtkdata.GetNumberOfComponents()
        nt = vtkdata.GetNumberOfTuples()
        array = arr([vtkdata.GetValue(i) for i in range(nc * nt)])
        if nc == 9:
            return array.reshape(nt, 3, 3)
        elif nc == 4:
            return array.reshape(nt, 2, 2)
        else:
            return array.reshape(nt, nc)

    def GetFieldRank(self, name):
        """
        Returns the rank of the supplied field.
        """
        try:
            pointdata = self.ugrid.GetPointData()
            vtkdata = pointdata.GetArray(name)
            vtkdata.GetNumberOfTuples()
        except:
            try:
                celldata = self.ugrid.GetCellData()
                vtkdata = celldata.GetArray(name)
                vtkdata.GetNumberOfTuples()
            except:
                raise Exception("Couldn't find point or cell field data with \
                                name " + name + " in file " + self.filename + ".")
        comps = vtkdata.GetNumberOfComponents()
        if comps == 1:
            return 0
        elif comps in [2, 3]:
            return 1
        elif comps in [4, 9]:
            return 2
        else:
            raise Exception("Field rank > 2 encountered")

    def Write(self, filename=[]):
        """Writes the grid to a vtu file.

        If no filename is specified it will use the name of the file originally
        read in, thus overwriting it!
        """
        if filename == []:
            filename = self.filename
        if filename is None:
            raise Exception("No file supplied")
        if filename.endswith('pvtu'):
            gridwriter = vtk.vtkXMLPUnstructuredGridWriter()
        else:
            gridwriter = vtk.vtkXMLUnstructuredGridWriter()

        gridwriter.SetFileName(filename)
        gridwriter.SetInput(self.ugrid)
        gridwriter.Write()

    def AddScalarField(self, name, array):
        """Adds a scalar field with the specified name using the values from
        the array."""
        data = vtk.vtkDoubleArray()
        data.SetNumberOfValues(len(array))
        data.SetName(name)
        for i in range(len(array)):
            data.SetValue(i, array[i])

        if len(array) == self.ugrid.GetNumberOfPoints():
            pointdata = self.ugrid.GetPointData()
            pointdata.AddArray(data)
            pointdata.SetActiveScalars(name)
        elif len(array) == self.ugrid.GetNumberOfCells():
            celldata = self.ugrid.GetCellData()
            celldata.AddArray(data)
            celldata.SetActiveScalars(name)
        else:
            raise Exception(
                "Length neither number of nodes nor number of cells")

    def AddVectorField(self, name, array):
        """Adds a vector field with the specified name using the values from
        the array."""
        n = array.size
        data = vtk.vtkDoubleArray()
        data.SetNumberOfComponents(array.shape[1])
        data.SetNumberOfValues(n)
        data.SetName(name)
        for i in range(n):
            data.SetValue(i, array.reshape(n)[i])

        if array.shape[0] == self.ugrid.GetNumberOfPoints():
            pointdata = self.ugrid.GetPointData()
            pointdata.AddArray(data)
            pointdata.SetActiveVectors(name)
        elif array.shape[0] == self.ugrid.GetNumberOfCells():
            celldata = self.ugrid.GetCellData()
            celldata.AddArray(data)
        else:
            raise Exception(
                "Length neither number of nodes nor number of cells")

    def AddField(self, name, array):
        """Adds a field with arbitrary number of components under the specified
        name using."""
        n = array.size
        sh = arr(array.shape)
        data = vtk.vtkDoubleArray()
        # number of tuples is sh[0]
        # number of components is the product of the rest of sh
        data.SetNumberOfComponents(sh[1:].prod())
        data.SetNumberOfValues(n)
        data.SetName(name)
        flatarray = array.reshape(n)
        for i in range(n):
            data.SetValue(i, flatarray[i])

        if sh[0] == self.ugrid.GetNumberOfPoints():
            pointdata = self.ugrid.GetPointData()
            pointdata.AddArray(data)
        elif sh[0] == self.ugrid.GetNumberOfCells():
            celldata = self.ugrid.GetCellData()
            celldata.AddArray(data)
        else:
            raise Exception(
                "Length neither number of nodes nor number of cells")

    def ApplyProjection(self, projection_x, projection_y, projection_z):
        """Applys a projection to the grid coordinates. This overwrites the
        existing values."""
        npoints = self.ugrid.GetNumberOfPoints()
        for i in range(npoints):
            (x, y, z) = self.ugrid.GetPoint(i)
            new_x = eval(projection_x)
            new_y = eval(projection_y)
            new_z = eval(projection_z)
            self.ugrid.GetPoints().SetPoint(i, new_x, new_y, new_z)

    def ApplyCoordinateTransformation(self, f):
        """Applys a coordinate transformation to the grid coordinates. This
        overwrites the existing values."""
        npoints = self.ugrid.GetNumberOfPoints()

        for i in range(npoints):
            (x, y, z) = self.ugrid.GetPoint(i)
            newX = f(arr([x, y, z]), t=0)
            self.ugrid.GetPoints().SetPoint(i, newX[0], newX[1], newX[2])

    def ApplyEarthProjection(self):
        """ Assume the input geometry is the Earth in Cartesian geometry and
        project to longatude, latitude, depth."""
        npoints = self.ugrid.GetNumberOfPoints()

        earth_radius = 6378000.0
        rad_to_deg = 180.0 / math.pi

        for i in range(npoints):
            (x, y, z) = self.ugrid.GetPoint(i)

            r = math.sqrt(x * x + y * y + z * z)
            depth = r - earth_radius
            longitude = rad_to_deg * math.atan2(y, x)
            latitude = 90.0 - rad_to_deg * math.acos(z / r)

            self.ugrid.GetPoints().SetPoint(i, longitude, latitude, depth)

    def ProbeData(self, coordinates, name):
        """Interpolate field values at these coordinates."""

        # Initialise locator
        locator = vtk.vtkPointLocator()
        locator.SetDataSet(self.ugrid)
        locator.SetTolerance(10.0)
        locator.Update()

        # Initialise probe
        points = vtk.vtkPoints()
        points.SetDataTypeToDouble()
        ilen, jlen = coordinates.shape
        for i in range(ilen):
            points.InsertNextPoint(
                coordinates[i][0], coordinates[i][1], coordinates[i][2])
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        probe = vtk.vtkProbeFilter()
        probe.SetInput(polydata)
        probe.SetSource(self.ugrid)
        probe.Update()

        # Generate a list invalidNodes, containing a map from invalid nodes in
        # the result to their closest nodes in the input
        valid_ids = probe.GetValidPoints()
        valid_loc = 0
        invalidNodes = []
        for i in range(ilen):
            if valid_ids.GetTuple1(valid_loc) == i:
                valid_loc += 1
            else:
                nearest = locator.FindClosestPoint(
                    [coordinates[i][0], coordinates[i][1], coordinates[i][2]])
                invalidNodes.append((i, nearest))

        # Get final updated values
        pointdata = probe.GetOutput().GetPointData()
        vtkdata = pointdata.GetArray(name)
        nc = vtkdata.GetNumberOfComponents()
        nt = vtkdata.GetNumberOfTuples()
        array = arr([vtkdata.GetValue(i) for i in range(nt * nc)])

        # Fix the point data at invalid nodes
        if len(invalidNodes) > 0:
            try:
                oldField = self.ugrid.GetPointData().GetArray(name)
            except:
                try:
                    oldField = self.ugrid.GetCellData().GetArray(name)
                except:
                    raise Exception("Couldn't find point or cell field data \
                                    with name " + name + " in file " + self.filename + ".")
            for invalidNode, nearest in invalidNodes:
                for comp in range(nc):
                    array[invalidNode * nc + comp] = oldField.GetValue(
                        nearest * nc + comp)

        valShape = self.GetField(name)[0].shape
        array.shape = tuple([nt] + list(valShape))

        return array

    def RemoveField(self, name):
        """Removes said field from the unstructured grid."""
        pointdata = self.ugrid.GetPointData()
        pointdata.RemoveArray(name)

    def GetLocations(self):
        """Returns an array with the locations of the nodes."""
        vtkPoints = self.ugrid.GetPoints()
        if vtkPoints is None:
            vtkData = vtk.vtkDoubleArray()
        else:
            vtkData = vtkPoints.GetData()
        return arr([vtkData.GetTuple3(i)
                    for i in range(vtkData.GetNumberOfTuples())])

    def GetCellPoints(self, id):
        """Returns an array with the node numbers of each cell (ndglno)."""
        idlist = vtk.vtkIdList()
        self.ugrid.GetCellPoints(id, idlist)
        return arr([idlist.GetId(i) for i in range(idlist.GetNumberOfIds())])

    def GetFieldNames(self):
        """Returns the names of the available fields."""
        vtkdata = self.ugrid.GetPointData()
        return [vtkdata.GetArrayName(i)
                for i in range(vtkdata.GetNumberOfArrays())]

    def GetPointCells(self, id):
        """Return an array with the elements which contain a node."""
        idlist = vtk.vtkIdList()
        self.ugrid.GetPointCells(id, idlist)
        return arr([idlist.GetId(i) for i in range(idlist.GetNumberOfIds())])

    def GetPointPoints(self, id):
        """Return the nodes connecting to a given node."""
        cells = self.GetPointCells(id)
        lst = []
        for cell in cells:
            lst = lst + list(self.GetCellPoints(cell))

        s = set(lst)  # remove duplicates
        return arr(list(s))  # make into a list again

    def GetDistance(self, x, y):
        """Return the distance in physical space between x and y."""
        posx = self.ugrid.GetPoint(x)
        posy = self.ugrid.GetPoint(y)
        return math.sqrt(sum([(posx[i] - posy[i]) ** 2
                              for i in range(len(posx))]))

    def Crop(self, min_x, max_x, min_y, max_y, min_z, max_z):
        """Trim off the edges defined by a bounding box."""
        trimmer = vtk.vtkExtractUnstructuredGrid()
        trimmer.SetInput(self.ugrid)
        trimmer.SetExtent(min_x, max_x, min_y, max_y, min_z, max_z)
        trimmer.Update()
        trimmed_ug = trimmer.GetOutput()

        self.ugrid = trimmed_ug

    def IntegrateField(self, field):
        """Integrate the supplied scalar field, assuming a linear
        representation on a tetrahedral mesh. Needs numpy-izing for speed."""

        assert field[0].shape in [(), (1,)]

        integral = 0.0

        n_cells = self.ugrid.GetNumberOfCells()
        vtkGhostLevels = self.ugrid.GetCellData().GetArray("vtkGhostLevels")
        for cell_no in range(n_cells):
            integrate_cell = True

            if vtkGhostLevels:
                integrate_cell = (vtkGhostLevels.GetTuple1(cell_no) == 0)

            if integrate_cell:
                Cell = self.ugrid.GetCell(cell_no)

                Cell_points = Cell.GetPoints()
                nCell_points = Cell.GetNumberOfPoints()
                if nCell_points == 4:
                    Volume = abs(Cell.ComputeVolume(Cell_points.GetPoint(0),
                                                    Cell_points.GetPoint(1),
                                                    Cell_points.GetPoint(2),
                                                    Cell_points.GetPoint(3)))
                elif nCell_points == 3:
                    Volume = abs(Cell.TriangleArea(Cell_points.GetPoint(0),
                                                   Cell_points.GetPoint(1),
                                                   Cell_points.GetPoint(2)))
                else:
                    raise Exception(
                        "Unexpected number of points: " + str(nCell_points))

                Cell_ids = Cell.GetPointIds()

                for point in range(Cell_ids.GetNumberOfIds()):
                    PointId = Cell_ids.GetId(point)
                    integral = integral + \
                        (Volume * field[PointId] / float(nCell_points))

        return integral

    def GetCellVolume(self, id):
        cell = self.ugrid.GetCell(id)
        pts = cell.GetPoints()
        if isinstance(cell, vtk.vtkTriangle):
            return cell.TriangleArea(pts.GetPoint(0), pts.GetPoint(1),
                                     pts.GetPoint(2))
        elif cell.GetNumberOfPoints() == 4:
            return abs(cell.ComputeVolume(pts.GetPoint(0), pts.GetPoint(1),
                       pts.GetPoint(2), pts.GetPoint(3)))
        elif cell.GetNumberOfPoints() == 3:
            return abs(cell.ComputeVolume(pts.GetPoint(0), pts.GetPoint(1),
                       pts.GetPoint(2)))
        else:
            raise Exception("Unexpected number of points")

    def GetFieldIntegral(self, name):
        """
        Integrate the named field.
        """

        return self.IntegrateField(self.GetField(name))

    def GetFieldRms(self, name):
        """
        Return the rms of the supplied scalar or vector field.
        """

        field = self.GetField(name)
        rank = self.GetFieldRank(name)
        if rank == 0:
            normField = arr([field[i] ** 2.0 for i in range(len(field))])
        elif rank == 1:
            normField = self.GetVectorNorm(name)
        else:
            raise Exception("Cannot calculate norm field for field rank > 1")
        volField = arr([1.0 for i in range(len(field))])
        rms = self.IntegrateField(normField)
        rms /= self.IntegrateField(volField)
        rms = numpy.sqrt(rms)

        return float(rms)

    def StructuredPointProbe(self, nx, ny, nz, bounding_box=None):
        """Probe the unstructured grid dataset using a structured points
        dataset."""

        probe = vtk.vtkProbeFilter()
        probe.SetSource(self.ugrid)

        sgrid = vtk.vtkStructuredPoints()

        bbox = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if bounding_box is None:
            bbox = self.ugrid.GetBounds()
        else:
            bbox = bounding_box

        sgrid.SetOrigin([bbox[0], bbox[2], bbox[4]])

        sgrid.SetDimensions(nx, ny, nz)

        spacing = [0.0, 0.0, 0.0]
        if nx > 1:
            spacing[0] = (bbox[1] - bbox[0]) / (nx - 1.0)
        if ny > 1:
            spacing[1] = (bbox[3] - bbox[2]) / (ny - 1.0)
        if nz > 1:
            spacing[2] = (bbox[5] - bbox[4]) / (nz - 1.0)

        sgrid.SetSpacing(spacing)

        probe.SetInput(sgrid)
        probe.Update()

        return probe.GetOutput()

    # Field manipulation methods ###

    def ManipulateField(self, fieldName, manipFunc, newFieldName=None):
        """Generic field manipulation method. Applies the supplied manipulation
        function manipFunc to the field fieldName. manipFunc must be a function
        of the form:

          def manipFunc(field, index):
            # ...
            return fieldValAtIndex
        """

        field = self.GetField(fieldName)
        if newFieldName is None or fieldName == newFieldName:
            self.RemoveField(fieldName)
            newFieldName = fieldName

        field = arr([manipFunc(field, i) for i in range(len(field))])
        self.AddField(newFieldName, field)

        return

    def AddFieldToField(self, fieldName, array, newFieldName=None):
        def ManipFunc(field, index):
            return field[index] + array[index]

        self.ManipulateField(fieldName, ManipFunc, newFieldName)

        return

    def SubFieldFromField(self, fieldName, array, newFieldName=None):
        def ManipFunc(field, index):
            return field[index] - array[index]

        self.ManipulateField(fieldName, ManipFunc, newFieldName)

        return

    def DotFieldWithField(self, fieldName, array, newFieldName=None):
        """
        Dot product
        """

        def ManipFunc(field, index):
            sum = 0.0
            for i, val in enumerate(field[index]):
                sum += val * array[index][i]

            return sum

        self.ManipulateField(fieldName, ManipFunc, newFieldName)

        return

    def CrossFieldWithField(self, fieldName, array, newFieldName=None,
                            postMultiply=True):
        """
        Cross product
        """

        def ManipFunc(field, index):
            if postMultiply:
                return numpy.cross(field[index], array[index])
            else:
                return numpy.cross(array[index], field[index])

        self.ManipulateField(fieldName, ManipFunc, newFieldName)

        return

    def MatMulFieldWithField(self, fieldName, array, newFieldName=None,
                             postMultiply=True):
        """
        Matrix multiplication
        """

        def ManipFunc(field, index):
            if postMultiply:
                return numpy.matrix(field[i]) * numpy.matix(array[i])
            else:
                return numpy.matix(array[i]) * numpy.matrix(field[i])

        self.ManipulateField(fieldName, ManipFunc, newFieldName)

        return

    # Default multiplication is dot product
    MulFieldByField = DotFieldWithField

    def GetDerivative(self, name):
        """
        Returns the derivative of field 'name', a
        vector field if 'name' is scalar, and a tensor field
        if 'name' is a vector. The field 'name' has to be point-wise data.
        The returned array gives a cell-wise derivative.
        """
        cd = vtk.vtkCellDerivatives()
        cd.SetInput(self.ugrid)
        pointdata = self.ugrid.GetPointData()
        nc = pointdata.GetArray(name).GetNumberOfComponents()
        if nc == 1:
            cd.SetVectorModeToComputeGradient()
            cd.SetTensorModeToPassTensors()
            pointdata.SetActiveScalars(name)
            cd.Update()
            vtkdata = cd.GetUnstructuredGridOutput(
            ).GetCellData().GetArray('ScalarGradient')
            return arr([vtkdata.GetTuple3(i)
                        for i in range(vtkdata.GetNumberOfTuples())])
        else:
            cd.SetTensorModeToComputeGradient()
            cd.SetVectorModeToPassVectors()
            pointdata.SetActiveVectors(name)
            cd.Update()
            vtkdata = cd.GetUnstructuredGridOutput(
            ).GetCellData().GetArray('VectorGradient')
            return arr([vtkdata.GetTuple9(i)
                        for i in range(vtkdata.GetNumberOfTuples())])

    def GetVorticity(self, name):
        """
        Returns the vorticity of vectorfield 'name'.
        The field 'name' has to be point-wise data.
        The returned array gives a cell-wise derivative.
        """
        cd = vtk.vtkCellDerivatives()
        cd.SetInput(self.ugrid)
        pointdata = self.ugrid.GetPointData()
        cd.SetVectorModeToComputeVorticity()
        cd.SetTensorModeToPassTensors()
        pointdata.SetActiveVectors(name)
        cd.Update()
        vtkdata = cd.GetUnstructuredGridOutput(
        ).GetCellData().GetArray('VectorGradient')
        return arr([vtkdata.GetTuple3(i)
                    for i in range(vtkdata.GetNumberOfTuples())])

    def CellDataToPointData(self):
        """
        Transforms all cell-wise fields in the vtu to point-wise fields.
        All existing fields will remain.
        """
        cdtpd = vtk.vtkCellDataToPointData()
        cdtpd.SetInput(self.ugrid)
        cdtpd.PassCellDataOn()
        cdtpd.Update()
        self.ugrid = cdtpd.GetUnstructuredGridOutput()


def VtuMatchLocations(vtu1, vtu2, tolerance=1.0e-6):
    """Check that the locations in the supplied vtus match exactly, returning
    True if they match and False otherwise.
    The locations must be in the same order.
    """

    locations1 = vtu1.GetLocations().tolist()
    locations2 = vtu2.GetLocations()
    if not len(locations1) == len(locations2):
        return False
    for i in range(len(locations1)):
        if not len(locations1[i]) == len(locations2[i]):
            return False
        for j in range(len(locations1[i])):
            if abs(locations1[i][j] - locations2[i][j]) > tolerance:
                return False

    return True


def VtuMatchLocationsArbitrary(vtu1, vtu2, tolerance=1.0e-6):
    """
    Check that the locations in the supplied vtus match, returning True if they
    match and False otherwise.
    The locations may be in a different order.
    """

    locations1 = vtu1.GetLocations()
    locations2 = vtu2.GetLocations()
    if not locations1.shape == locations2.shape:
        return False

    for j in range(locations1.shape[1]):
        # compute the smallest possible precision given the range of this
        # coordinate
        epsilon = numpy.finfo(numpy.float).eps * numpy.abs(
            locations1[:, j]).max()
        if tolerance < epsilon:
            # the specified tolerance is smaller than possible machine
            # precision (or something else went wrong)
            raise Exception("Specified tolerance is smaller than machine \
                            precision of given locations")
        # ensure epsilon doesn't get too small (might be for zero for instance)
        epsilon = max(epsilon, tolerance / 100.0)

        # round to that many decimal places (-2 to be sure) so that
        # we don't get rounding issues with lexsort
        locations1[:, j] = numpy.around(
            locations1[:, j], int(-numpy.log10(epsilon)) - 2)
        locations2[:, j] = numpy.around(
            locations2[:, j], int(-numpy.log10(epsilon)) - 2)

    # lexical sort on x,y and z coordinates resp. of locations1 and locations2
    sort_index1 = numpy.lexsort(locations1.T)
    sort_index2 = numpy.lexsort(locations2.T)

    # should now be in same order, so we can check for its biggest difference
    return numpy.allclose(locations1[sort_index1], locations2[sort_index2],
                          atol=tolerance)


def VtuDiff(vtu1, vtu2, filename=None):
    """Generate a vtu with fields generated by taking the difference between
    the field values in the two supplied vtus. Fields that are not common
    between the two vtus are neglected. If probe is True, the fields of vtu2
    are projected onto the cell points of vtu1. Otherwise, the cell points of
    vtu1 and vtu2 must match."""

    # Generate empty output vtu
    resultVtu = vtu()
    resultVtu.filename = filename

    # If the input vtu point locations match, do not use probe
    useProbe = not VtuMatchLocations(vtu1, vtu2)

    # Copy the grid from the first input vtu into the output vtu
    resultVtu.ugrid.DeepCopy(vtu1.ugrid)

    # Find common field names between the input vtus and generate corresponding
    # difference fields
    fieldNames1 = vtu1.GetFieldNames()
    fieldNames2 = vtu2.GetFieldNames()
    for fieldName in fieldNames1:
        if fieldName in fieldNames2:
            if useProbe:
                field2 = vtu2.ProbeData(vtu1.GetLocations(), fieldName)
            else:
                field2 = vtu2.GetField(fieldName)
            resultVtu.SubFieldFromField(fieldName, field2)
        else:
            resultVtu.RemoveField(fieldName)

    return resultVtu


def usage():
    print 'Usage:'
    print 'COMMAND LINE: vtktools.py [-h] [-p] [-e var1,var2, ...] INPUT_FILENAME'
    print ''
    print 'INPUT_FILENAME:'
    print '           The input file name.'
    print ''
    print 'OPTIONS:'
    print '   -h      Prints this usage message.'
    print '   -p      Converts the coordinates from xyz to latitude and longitude.'
    print '   -e      Extracts the data point from the variables provided.'

if __name__ == "__main__":
    import vtktools
    import getopt

    optlist, args = getopt.getopt(sys.argv[1:], 'hpe:')

    v = vtktools.vtu(args[0])

    # Parse arguments
    LongLat = False
    for o, a in optlist:
        if o == '-h':
            usage()
        elif o == '-p':
            LongLat = True
        elif o == '-e':
            scalars = a.strip().split(",")

    # Project domain if necessary
    if(LongLat):
        v.ApplyEarthProjection()

    # Extract variables
    if(scalars):
        npoints = v.ugrid.GetNumberOfPoints()
        nvar = len(scalars)
        for i in range(npoints):
            (x, y, z) = v.ugrid.GetPoint(i)
            line = "%lf " % x + "%lf " % y
            for scalar in scalars:
                line = line + \
                    " %lf" % v.ugrid.GetPointData().GetArray(
                        scalar).GetTuple1(i)
            print line

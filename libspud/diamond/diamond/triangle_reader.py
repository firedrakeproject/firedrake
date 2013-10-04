# Author: Daryl Harrison

# Enthought library imports.
from enthought.traits.api import Instance, String, List, Int
from enthought.traits.ui.api import View, Item
from enthought.tvtk.api import tvtk

# Local imports.
from enthought.mayavi.core.file_data_source import FileDataSource
from enthought.mayavi.core.pipeline_info import PipelineInfo
from enthought.mayavi.core.traits import DEnum
from os import path
from numpy import array, append, fromstring
from re import compile, MULTILINE

######################################################################
# `TriangleReader` class
######################################################################
class TriangleReader(FileDataSource):
    """
    Reader for the Triangle file formats:
        2D <http://www.cs.cmu.edu/~quake/triangle.html>
            Supports opening .egde files to construct a surface mesh comprised of lines
            and .ele files to construct a solid mesh comprised of triangles.
        3D <http://tetgen.berlios.de/fformats.html>
            Supports opening .face files to construct a surface mesh comprised of triangles
            and .ele files to construct a solid mesh comprised of tetrahedra.
    Outputs an unstructured grid dataset.
    """

    # The version of this class.  Used for persistence.
    __version__ = 0

    # Information about what this object can produce.
    output_info = PipelineInfo(datasets=['unstructured_grid'],
                               attribute_types=['any'],
                               attributes=['scalars'])

    # The active point scalar name.
    point_scalars_name = DEnum(values_name='_point_scalars_list',
                               desc='scalar point data attribute to use')

    # The active cell scalar name.
    cell_scalars_name = DEnum(values_name='_cell_scalars_list',
                               desc='scalar cell data attribute to use')

    ########################################
    # Private traits.

    # These private traits store the list of available data
    # attributes.  The non-private traits use these lists internally.
    _cell_scalars_list = List(String)
    _point_scalars_list = List(String)

    # The VTK dataset to manage.
    _grid = Instance(tvtk.UnstructuredGrid, args=(), allow_none=False)

    # The basename of the file which has been loaded.
    _basename = String

    # Indicates whether nodes are numbered from 0 or 1 (the file
    # format allows both).
    _numbered_from = Int

    # This filter allows us to change the attributes of the data
    # object and will ensure that the pipeline is properly taken care
    # of.
    _assign_attribute = Instance(tvtk.AssignAttribute, args=(), allow_none=False)

    ########################################
    # The view.

    view = View(Item(name='point_scalars_name'),
                Item(name='cell_scalars_name'))

    ########################################
    # `FileDataSource` interface.

    def initialize(self, base_file_name):
        split = path.splitext(base_file_name)
        self._basename = split[0]
        extension = split[1]

        self._assign_attribute.input = self._grid

        self._read_node_file()
        if (extension == '.face' or extension == '.edge'):
            self._read_face_edge_file(extension)
        else:
            self._read_ele_file()

        self.outputs = [self._assign_attribute.output]
        self.name = 'Triangle file (%s%s)' %(path.basename(self._basename), extension)

    ########################################
    # File reading methods.

    def _read_node_file(self):
        """
        Loads data from {basename}.node, and inserts points and point
        scalars into the unstructured grid.
        """
        file_name = '%s.node' %self._basename

        # Load all data.
        all_data = self._get_data(file_name)
        # Grab values from the first line of data file.
        points, dimensions, attributes, boundary_marker = map(int, all_data[0:4])
        # Reshape remainder of array.
        data_array = all_data[4:].reshape(points, 1+dimensions+attributes+boundary_marker)

        self._numbered_from = int(data_array[0][0])

        points_array = array(data_array[:, 1:(1+dimensions)], 'double')
        if (dimensions == 2):
            # Add a 0 to each point if it is 2D.
            points_array = array(map (lambda a: append(a, 0), points_array))
        self._grid.points = points_array

        for i in range(attributes):
            attribute_array = data_array[:, (i+dimensions+1):(i+dimensions+2)]
            self._add_attribute_array(attribute_array, i, 'point')

        if (boundary_marker):
            boundary_marker_array = data_array[:, (dimensions+attributes+1):(dimensions+attributes+2)]
            self._add_boundary_marker_array(boundary_marker_array, 'point')


    def _read_face_edge_file(self, extension):
        """
        Loads data from 2D {basename}.edge or 3D {basename}.face, and inserts
        triangle/line cells and cell scalars into the unstructured grid.
        """
        file_name = '%s%s' %(self._basename, extension)

        if (extension == '.edge'):
            # 2D. Expect two endpoints which form a line.
            npoints = 2
            cell_type = tvtk.Line().cell_type
        else:
            # 3D. Expect three points which form a triangle.
            npoints = 3
            cell_type = tvtk.Triangle().cell_type

        # Load all data.
        all_data = self._get_data(file_name)
        # Grab values from the first line of data file.
        faces_edges, boundary_marker = map(int, all_data[0:2])
        # Reshape remainder of array.
        data_array = all_data[2:].reshape(faces_edges, npoints+1+boundary_marker)

        nodes_array = data_array[:, 1:npoints+1] - self._numbered_from
        self._grid.set_cells(cell_type, nodes_array)

        if (boundary_marker):
            boundary_marker_array = data_array[:, npoints+1:npoints+2]
            self._add_boundary_marker_array(boundary_marker_array, 'cell')


    def _read_ele_file(self):
        """
        Loads data from {basename}.ele, and inserts triangle/tetrahedron cells
        and cell scalars into the unstructured grid.
        """
        file_name = '%s.ele' %self._basename

        # Load all data.
        all_data = self._get_data(file_name)
        # Grab values from the first line of data file.
        tet_tri, nodes_per_tet_tri, attributes =  map(int, all_data[0:3])
        # Reshape remainder of array.
        data_array = all_data[3:].reshape(tet_tri, 1+nodes_per_tet_tri+attributes)

        nodes_array = data_array[:, 1:(nodes_per_tet_tri+1)] - self._numbered_from

        if (nodes_per_tet_tri == 3):
            cell_type = tvtk.Triangle().cell_type
        else:
            cell_type = tvtk.Tetra().cell_type

        self._grid.set_cells(cell_type, nodes_array)

        for i in range(attributes):
            attribute_array = data_array[:, (i+nodes_per_tet_tri+1):(i+nodes_per_tet_tri+2)]
            self._add_attribute_array(attribute_array, i, 'cell')


    def _get_data(self, file_name):
        """
        Returns a 1D array containing all the data from the given file.
        """
        file = open(file_name)
        file_string = file.read()

        # Strip comments.
        pattern = compile('#.*?$', MULTILINE)
        file_string = pattern.sub('', file_string)

        # Load all data into array.
        return fromstring(file_string, dtype=float, sep=" ")

    ########################################
    # Unstructured grid construction
    # methods.

    def _add_attribute_array(self, attribute_array, i, type):
        """
        Adds the given attribute array to either point_data or
        cell_data of the unstructured grid.
        """
        attribute_array_name = 'Attribute %i' %i
        if (type == 'cell'): # .ele file attributes are of type Int
            tvtk_attribute_array = tvtk.IntArray(name=attribute_array_name)
            attribute_array = map(int, attribute_array)
        else:                # .node file attributes are of type Float
            tvtk_attribute_array = tvtk.FloatArray(name=attribute_array_name)
        tvtk_attribute_array.from_array(attribute_array)
        getattr(self._grid, '%s_data' %type).add_array(tvtk_attribute_array)
        getattr(self, '_%s_scalars_list' %type).append(attribute_array_name)

        if (i == 0):
            self._set_data_name(type, 'Attribute 0')


    def _add_boundary_marker_array(self, boundary_marker_array, type):
        """
        Adds the given boundary marker array to either point_data or
        cell_data of the unstructured grid.
        """
        boundary_marker_array_name = 'Boundary Marker'
        tvtk_boundary_marker_array = tvtk.IntArray(name=boundary_marker_array_name)
        tvtk_boundary_marker_array.from_array(boundary_marker_array)
        getattr(self._grid, '%s_data' %type).add_array(tvtk_boundary_marker_array)
        getattr(self, '_%s_scalars_list' %type).append(boundary_marker_array_name)
        self._set_data_name(type, 'Boundary Marker')

    ########################################
    # Methods taken and modified from
    # SetActiveAttribute filter.

    def _point_scalars_name_changed(self, value):
        self._set_data_name('point', value)


    def _cell_scalars_name_changed(self, value):
        self._set_data_name('cell', value)


    def _set_data_name(self, attr_type, value):
        """
        Sets the selected point or cell scalar to be active, and
        deactivates the scalar of the other type.
        """
        if value is None:
            return

        if (attr_type == 'point'):
            data = self._grid.point_data
            other_data = self._grid.cell_data
        else:
            data = self._grid.cell_data
            other_data = self._grid.point_data

        method = getattr(data, 'set_active_scalars')
        method(value)

        # Deactivate other attribute.
        method = getattr(other_data, 'set_active_scalars')
        method(None)

        self._assign_attribute.assign(value, 'SCALARS', attr_type.upper()+'_DATA')
        self._assign_attribute.update()

        # Fire an event, so the changes propagate.
        self.data_changed = True

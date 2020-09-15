import enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches
import matplotlib.tri
from matplotlib.path import Path
from matplotlib.collections import LineCollection, PolyCollection
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from ufl import Cell
from tsfc.finatinterface import create_base_element
from firedrake import (interpolate, sqrt, inner, Function, SpatialCoordinate,
                       FunctionSpace, VectorFunctionSpace, PointNotInDomainError,
                       Constant, assemble, dx)
from firedrake.mesh import MeshGeometry

__all__ = ["plot", "triplot", "tricontourf", "tricontour", "trisurf", "tripcolor",
           "quiver", "streamplot"]


def _autoscale_view(axes, coords):
    axes.autoscale_view()

    # Dirty hack; autoscale_view doesn't appear to work for 3D plots.
    if isinstance(axes, mpl_toolkits.mplot3d.Axes3D):
        setters = ["set_xlim", "set_ylim", "set_zlim"]
        for setter, idx in zip(setters, range(coords.shape[1])):
            try:
                setter = getattr(axes, setter)
            except AttributeError:
                continue
            amin = coords[:, idx].min()
            amax = coords[:, idx].max()
            extra = (amax - amin) / 20
            if extra == 0.0:
                # 1D interval
                extra = 0.5
            amin -= extra
            amax += extra
            setter(amin, amax)


def _get_collection_types(gdim, tdim):
    if gdim == 2:
        if tdim == 1:
            # Probably a CircleCollection?
            raise NotImplementedError("Didn't get to this yet...")
        elif tdim == 2:
            return LineCollection, PolyCollection
    elif gdim == 3:
        if tdim == 1:
            raise NotImplementedError("Didn't get to this one yet...")
        elif tdim == 2:
            return Line3DCollection, Poly3DCollection
        elif tdim == 3:
            return Poly3DCollection, Poly3DCollection

    raise ValueError("Geometric dimension must be either 2 or 3!")


def triplot(mesh, axes=None, interior_kw={}, boundary_kw={}):
    r"""Plot a mesh with a different color for each boundary segment

    The interior and boundary keyword arguments can be any keyword argument for
    :class:`LineCollection <matplotlib.collections.LinecCollection>` and
    related types.

    :arg mesh: mesh to be plotted
    :arg axes: matplotlib :class:`Axes <matplotlib.axes.Axes>` object on which to plot mesh
    :arg interior_kw: keyword arguments to apply when plotting the mesh interior
    :arg boundary_kw: keyword arguments to apply when plotting the mesh boundary
    :return: list of matplotlib :class:`Collection <matplotlib.collections.Collection>` objects
    """
    gdim = mesh.geometric_dimension()
    tdim = mesh.topological_dimension()
    BoundaryCollection, InteriorCollection = _get_collection_types(gdim, tdim)
    quad = mesh.ufl_cell().cellname() == "quadrilateral"

    if axes is None:
        figure = plt.figure()
        if gdim == 3:
            axes = figure.add_subplot(111, projection='3d')
        else:
            axes = figure.add_subplot(111)

    coordinates = mesh.coordinates
    element = coordinates.function_space().ufl_element()
    if element.degree() != 1:
        # Interpolate to piecewise linear.
        V = VectorFunctionSpace(mesh, element.family(), 1)
        coordinates = interpolate(coordinates, V)

    coords = coordinates.dat.data_ro
    result = []
    interior_kw = dict(interior_kw)
    # If the domain isn't a 3D volume, draw the interior.
    if tdim <= 2:
        cell_node_map = coordinates.cell_node_map().values
        idx = (tuple(range(tdim + 1)) if not quad else (0, 1, 3, 2)) + (0,)
        vertices = coords[cell_node_map[:, idx]]

        interior_kw["edgecolors"] = interior_kw.get("edgecolors", "k")
        interior_kw["linewidths"] = interior_kw.get("linewidths", 1.0)
        if gdim == 2:
            interior_kw["facecolors"] = interior_kw.get("facecolors", "none")

        interior_collection = InteriorCollection(vertices, **interior_kw)
        axes.add_collection(interior_collection)
        result.append(interior_collection)

    # Add colored lines/polygons for the boundary facets
    facets = mesh.exterior_facets
    local_facet_ids = facets.local_facet_dat.data_ro
    exterior_facet_node_map = coordinates.exterior_facet_node_map().values
    topology = coordinates.function_space().finat_element.cell.get_topology()

    mask = np.zeros(exterior_facet_node_map.shape, dtype=bool)
    for facet_index, local_facet_index in enumerate(local_facet_ids):
        mask[facet_index, topology[tdim - 1][local_facet_index]] = True
    faces = exterior_facet_node_map[mask].reshape(-1, tdim)

    markers = facets.unique_markers
    color_key = "colors" if tdim <= 2 else "facecolors"
    boundary_colors = boundary_kw.pop(color_key, None)
    if boundary_colors is None:
        cmap = matplotlib.cm.get_cmap("Dark2")
        num_markers = len(markers)
        colors = cmap([k / num_markers for k in range(num_markers)])
    else:
        colors = matplotlib.colors.to_rgba_array(boundary_colors)

    boundary_kw = dict(boundary_kw)
    if tdim == 3:
        boundary_kw["edgecolors"] = boundary_kw.get("edgecolors", "k")
        boundary_kw["linewidths"] = boundary_kw.get("linewidths", 1.0)
    for marker, color in zip(markers, colors):
        face_indices = facets.subset(int(marker)).indices
        marker_faces = faces[face_indices, :]
        vertices = coords[marker_faces]
        _boundary_kw = dict(**{color_key: color, "label": marker}, **boundary_kw)
        marker_collection = BoundaryCollection(vertices, **_boundary_kw)
        axes.add_collection(marker_collection)
        result.append(marker_collection)

    # Dirty hack to enable legends for 3D volume plots. See the function
    # `Poly3DCollection.set_3d_properties`.
    for collection in result:
        if isinstance(collection, Poly3DCollection):
            collection._facecolors2d = PolyCollection.get_facecolor(collection)
            collection._edgecolors2d = PolyCollection.get_edgecolor(collection)

    _autoscale_view(axes, coords)
    return result


def _plot_2d_field(method_name, function, *args, **kwargs):
    axes = kwargs.pop("axes", None)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    if len(function.ufl_shape) == 1:
        mesh = function.ufl_domain()
        element = function.ufl_element().sub_elements()[0]
        Q = FunctionSpace(mesh, element)
        function = interpolate(sqrt(inner(function, function)), Q)

    num_sample_points = kwargs.pop("num_sample_points", 10)
    coords, vals, triangles = _two_dimension_triangle_func_val(function,
                                                               num_sample_points)

    x, y = coords[:, 0], coords[:, 1]
    triangulation = matplotlib.tri.Triangulation(x, y, triangles=triangles)

    method = getattr(axes, method_name)
    return method(triangulation, vals, *args, **kwargs)


def tricontourf(function, *args, **kwargs):
    r"""Create a filled contour plot of a 2D Firedrake :class:`~.Function`

    If the input function is a vector field, the magnitude will be plotted.

    :arg function: the Firedrake :class:`~.Function` to plot
    :arg args: same as for matplotlib :func:`tricontourf <matplotlib.pyplot.tricontourf>`
    :arg kwargs: same as for matplotlib
    :return: matplotlib :class:`ContourSet <matplotlib.contour.ContourSet>` object
    """
    return _plot_2d_field("tricontourf", function, *args, **kwargs)


def tricontour(function, *args, **kwargs):
    r"""Create a contour plot of a 2D Firedrake :class:`~.Function`

    If the input function is a vector field, the magnitude will be plotted.

    :arg function: the Firedrake :class:`~.Function` to plot
    :arg args: same as for matplotlib :func:`tricontour <matplotlib.pyplot.tricontour>`
    :arg kwargs: same as for matplotlib
    :return: matplotlib :class:`ContourSet <matplotlib.contour.ContourSet>` object
    """
    return _plot_2d_field("tricontour", function, *args, **kwargs)


def tripcolor(function, *args, **kwargs):
    r"""Create a pseudo-color plot of a 2D Firedrake :class:`~.Function`

    If the input function is a vector field, the magnitude will be plotted.

    :arg function: the function to plot
    :arg args: same as for matplotlib :func:`tripcolor <matplotlib.pyplot.tripcolor>`
    :arg kwargs: same as for matplotlib
    :return: matplotlib :class:`PolyCollection <matplotlib.collections.PolyCollection>` object
    """
    return _plot_2d_field("tripcolor", function, *args, **kwargs)


def _trisurf_3d(axes, function, *args, vmin=None, vmax=None, norm=None, **kwargs):
    num_sample_points = kwargs.pop("num_sample_points", 10)
    coords, vals, triangles = _two_dimension_triangle_func_val(function,
                                                               num_sample_points)
    vertices = coords[triangles]
    collection = Poly3DCollection(vertices, *args, **kwargs)

    avg_vals = vals[triangles].mean(axis=1)
    collection.set_array(avg_vals)
    if (vmin is not None) or (vmax is not None):
        collection.set_clim(vmin, vmax)
    if norm is not None:
        collection.set_norm(norm)

    axes.add_collection(collection)
    _autoscale_view(axes, coords)

    return collection


def trisurf(function, *args, **kwargs):
    r"""Create a 3D surface plot of a 2D Firedrake :class:`~.Function`

    If the input function is a vector field, the magnitude will be plotted.

    :arg function: the Firedrake :class:`~.Function` to plot
    :arg args: same as for matplotlib :meth:`plot_trisurf <mpl_toolkits.mplot3d.axes3d.Axes3D.plot_trisurf>`
    :arg kwargs: same as for matplotlib
    :return: matplotlib :class:`Poly3DCollection <mpl_toolkits.mplot3d.art3d.Poly3DCollection>` object
    """
    axes = kwargs.pop("axes", None)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')

    _kwargs = {"antialiased": False, "edgecolor": "none",
               "cmap": plt.rcParams["image.cmap"]}
    _kwargs.update(kwargs)

    mesh = function.ufl_domain()
    if mesh.geometric_dimension() == 3:
        return _trisurf_3d(axes, function, *args, **_kwargs)

    if len(function.ufl_shape) == 1:
        element = function.ufl_element().sub_elements()[0]
        Q = FunctionSpace(mesh, element)
        function = interpolate(sqrt(inner(function, function)), Q)

    num_sample_points = kwargs.pop("num_sample_points", 10)
    coords, vals, triangles = _two_dimension_triangle_func_val(function,
                                                               num_sample_points)
    x, y = coords[:, 0], coords[:, 1]
    triangulation = matplotlib.tri.Triangulation(x, y, triangles=triangles)
    _kwargs.update({"shade": False})
    return axes.plot_trisurf(triangulation, vals, *args, **_kwargs)


def quiver(function, **kwargs):
    r"""Make a quiver plot of a 2D vector Firedrake :class:`~.Function`

    :arg function: the vector field to plot
    :arg kwargs: same as for matplotlib :func:`quiver <matplotlib.pyplot.quiver>`
    :return: matplotlib :class:`Quiver <matplotlib.quiver.Quiver>` object
    """
    if function.ufl_shape != (2,):
        raise ValueError("Quiver plots only defined for 2D vector fields!")

    axes = kwargs.pop("axes", None)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    coords = function.ufl_domain().coordinates.dat.data_ro
    V = function.ufl_domain().coordinates.function_space()
    vals = interpolate(function, V).dat.data_ro
    C = np.linalg.norm(vals, axis=1)
    return axes.quiver(*(coords.T), *(vals.T), C, **kwargs)


def _step_to_boundary(mesh, x, u, dt, loc_tolerance):
    bracket = (0., dt)
    while bracket[1] - bracket[0] > loc_tolerance * dt:
        ds = (bracket[1] + bracket[0]) / 2
        if mesh.locate_cell(x + ds * u, tolerance=loc_tolerance) is None:
            bracket = (bracket[0], ds)
        else:
            bracket = (ds, bracket[1])

    return bracket[0]


def streamline(function, point, direction=+1, tolerance=3e-3, loc_tolerance=1e-10):
    r"""Generate a streamline of a vector field starting from a point

    :arg function: the Firedrake :class:`~.Function` to plot
    :arg point: the starting point of the streamline
    :arg direction: either +1 or -1 to integrate forward or backward
    :arg tolerance: dimensionless tolerance for the RK12 adaptive integration
    :arg loc_tolerance: tolerance for point location
    :returns: a generator of the position, velocity, and timestep ``(x, v, dt)``
    """
    mesh = function.ufl_domain()
    cell_sizes = mesh.cell_sizes

    x = np.array(point)
    v1 = direction * function.at(x, tolerance=loc_tolerance)
    r = cell_sizes.at(x, tolerance=loc_tolerance)
    dt = 0.5 * r / np.sqrt(np.sum(v1**2))

    while True:
        try:
            v2 = direction * function.at(x + dt * v1, tolerance=loc_tolerance)
        except PointNotInDomainError:
            ds = _step_to_boundary(mesh, x, v1, dt, loc_tolerance)
            y = x + ds * v1
            v1 = direction * function.at(y, tolerance=loc_tolerance)
            yield y, v1, ds
            break

        dx1 = dt * v1
        dx2 = dt * (v1 + v2) / 2
        error = np.sqrt(np.sum((dx2 - dx1)**2)) / r

        if error <= tolerance:
            y = x + dx2
            try:
                vy = direction * function.at(y, tolerance=loc_tolerance)
                r = cell_sizes.at(y, tolerance=loc_tolerance)
            except PointNotInDomainError:
                v = (v1 + v2) / 2
                ds = _step_to_boundary(mesh, x, v, dt, loc_tolerance)
                y = x + ds * v
                v1 = direction * function.at(y, tolerance=loc_tolerance)
                yield y, v1, ds
                break

            x[:] = y
            v1[:] = vy
            yield y, v1, dt

        # TODO: increase the step length if the error < fraction * tol
        max_step_length = 0.5 * r / np.sqrt(np.sum(v1**2))
        if error == 0.:
            dt = max(1.5 * dt, max_step_length)
        else:
            proposed_dt = 0.85 * np.sqrt(tolerance / error) * dt
            dt = min(max_step_length, proposed_dt)


class Reason(enum.IntEnum):
    LENGTH = enum.auto()
    TIME = enum.auto()
    BOUNDARY = enum.auto()


class Streamplotter(object):
    def __init__(self, function, resolution, min_length, max_time, tolerance,
                 loc_tolerance):
        r"""Generates a dense set of streamlines of a vector field

        This class is a utility for the :func:`~firedrake.plot.streamplot`
        function.
        """
        self.function = function
        self.resolution = resolution
        self.min_length = min_length
        self.max_time = max_time
        self.tolerance = tolerance
        self.loc_tolerance = loc_tolerance

        # Create a grid to track the distance to the nearest streamline
        mesh = self.function.ufl_domain()
        coords = mesh.coordinates.dat.data_ro
        self._xmin = coords.min(axis=0)
        xmax = coords.max(axis=0)
        self._r = self.resolution / np.sqrt(mesh.geometric_dimension())
        shape = tuple(((xmax - self._xmin) / self._r).astype(int) + 2)
        self._grid = np.full(shape, 4 * self.resolution)

        self.streamlines = []

    def _grid_index(self, x):
        r"""Return the indices in the grid where the given point lies"""
        return tuple(((x - self._xmin) / self._r).astype(int))

    def _grid_point(self, index):
        r"""Return the position of the given grid index"""
        return self._xmin + self._r * np.array(index)

    def _approx_distance_to_streamlines(self, x):
        r"""Return the approximate distance to the set of streamlines that have
        been added, capped out to twice the resolution"""
        index = self._grid_index(x)
        g = self._grid[index[0]:index[0] + 2, index[1]:index[1] + 2]
        lx, ly = (x - self._grid_point(index)) / self._r
        return ((1 - ly) * ((1 - lx) * g[0, 0] + lx * g[1, 0])
                + ly * ((1 - lx) * g[0, 1] + lx * g[1, 1]))

    def _compute_chunk(self, start_point, direction):
        r"""Compute a short segment of a streamline starting at a given point"""
        s = [start_point]
        L = 0.
        T = 0.
        reason = Reason.BOUNDARY
        for x, v, dt in streamline(self.function, start_point, direction,
                                   self.tolerance, self.loc_tolerance):
            delta = x - s[-1]
            s.append(x)
            T += dt
            L += np.sqrt(np.sum(delta**2))

            if L >= self.min_length:
                reason = Reason.LENGTH
                break

            if T >= self.max_time:
                reason = Reason.TIME
                break

        return np.array(s), reason

    def _enter_distance_to_chunk(self, chunk):
        shape = self._grid.shape
        # TODO: Make this distance to segments, not just distance to points --
        # could be overestimating the distance in the case of very long segments
        for x in chunk:
            ix, iy = self._grid_index(x)
            for i in range(max(ix - 2, 0), min(ix + 4, shape[0])):
                for j in range(max(iy - 2, 0), min(iy + 4, shape[1])):
                    y = self._grid_point((i, j))
                    dist = min(np.sqrt(np.sum((x - y)**2)), 2 * self.resolution)
                    self._grid[i, j] = min(dist, self._grid[i, j])

    def _index_of_first_bad_point(self, chunk):
        r"""Return the index of the first point in the chunk that is close to
        another streamline"""
        for k, x in enumerate(chunk):
            if self._approx_distance_to_streamlines(x) < self.resolution:
                return k

        return None

    def _add_streamline_direction(self, chunk, index, reason, direction):
        chunks = []
        while (index is None) and (reason == Reason.LENGTH):
            next_point = chunk[-1, :]
            next_chunk, next_reason = self._compute_chunk(next_point, direction)

            # Cut off the first point of the next chunk -- it's identical to
            # the last point of the previous one
            next_chunk = next_chunk[1:, :]
            next_index = self._index_of_first_bad_point(next_chunk)

            # Add the previous chunk
            self._enter_distance_to_chunk(chunk[:index, :])
            chunks.append(chunk[:index, :])
            chunk, reason, index = next_chunk, next_reason, next_index

        if index != 0:
            self._enter_distance_to_chunk(chunk[:index])
            chunks.append(chunk[:index])

        return np.concatenate(chunks, axis=0)

    def add_streamline(self, point):
        # If the point isn't inside the domain, bail out
        outside = self.function.ufl_domain().locate_cell(point) is None
        too_close = self._approx_distance_to_streamlines(point) < self.resolution
        if outside or too_close:
            return

        # Compute the first segments of the forward and backward chunks from
        # the current point
        fchunk, freason = self._compute_chunk(point, direction=+1)
        findex = self._index_of_first_bad_point(fchunk)

        bchunk, breason = self._compute_chunk(point, direction=-1)
        bindex = self._index_of_first_bad_point(bchunk)

        # If the initial segments aren't long enough, bail out
        flength = np.sum(np.sqrt(np.sum(np.diff(fchunk[:findex], axis=0)**2, axis=1)))
        blength = np.sum(np.sqrt(np.sum(np.diff(bchunk[:bindex], axis=0)**2, axis=1)))
        if flength + blength < self.min_length:
            return

        forward = self._add_streamline_direction(fchunk, findex, freason, +1)
        backward = self._add_streamline_direction(bchunk, bindex, breason, -1)

        streamline = np.vstack((backward[::-1], forward[1:]))
        self.streamlines.append(streamline)


def streamplot(function, resolution=None, min_length=None, max_time=None,
               start_width=0.5, end_width=1.5, tolerance=3e-3, loc_tolerance=1e-10,
               seed=None, **kwargs):
    r"""Create a streamline plot of a vector field

    Similar to matplotlib :func:`streamplot <matplotlib.pyplot.streamplot>`

    :arg function: the Firedrake :class:`~.Function` to plot
    :arg resolution: minimum spacing between streamlines (defaults to domain size / 20)
    :arg min_length: minimum length of a streamline (defaults to 4x resolution)
    :arg max_time: maximum time to integrate a streamline
    :arg start_width: line width at beginning of streamline
    :arg end_width: line width at end of streamline, to convey direction
    :arg tolerance: dimensionless tolerance for adaptive ODE integration
    :arg loc_tolerance: point location tolerance for :meth:`~firedrake.functions.Function.at`
    :kwarg kwargs: same as for matplotlib :class:`~matplotlib.collections.LineCollection`
    """
    import randomgen

    if function.ufl_shape != (2,):
        raise ValueError("Streamplot only defined for 2D vector fields!")

    axes = kwargs.pop("axes", None)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    mesh = function.ufl_domain()
    if resolution is None:
        coords = mesh.coordinates.dat.data_ro
        resolution = (coords.max(axis=0) - coords.min(axis=0)).max() / 20

    if min_length is None:
        min_length = 4 * resolution

    if max_time is None:
        area = assemble(Constant(1) * dx(mesh))
        average_speed = np.sqrt(assemble(inner(function, function) * dx) / area)
        max_time = 50 * min_length / average_speed

    streamplotter = Streamplotter(function, resolution, min_length, max_time,
                                  tolerance, loc_tolerance)

    # TODO: better way of seeding start points
    shape = streamplotter._grid.shape
    xmin = streamplotter._grid_point((0, 0))
    xmax = streamplotter._grid_point((shape[0] - 2, shape[1] - 2))
    X, Y = np.meshgrid(np.linspace(xmin[0], xmax[0], shape[0] - 2),
                       np.linspace(xmin[1], xmax[1], shape[1] - 2))
    start_points = np.vstack((X.ravel(), Y.ravel())).T

    # Randomly shuffle the start points
    generator = randomgen.MT19937(seed).generator
    for x in generator.permutation(np.array(start_points)):
        streamplotter.add_streamline(x)

    # Colors are determined by the speed, thicknesses by arc length
    speeds = []
    widths = []
    for streamline in streamplotter.streamlines:
        velocity = np.array(function.at(streamline, tolerance=loc_tolerance))
        speed = np.sqrt(np.sum(velocity**2, axis=1))
        speeds.extend(speed[:-1])

        delta = np.sqrt(np.sum(np.diff(streamline, axis=0)**2, axis=1))
        arc_length = np.cumsum(delta)
        length = arc_length[-1]
        s = arc_length / length
        linewidth = (1 - s) * start_width + s * end_width
        widths.extend(linewidth)

    points = []
    for streamline in streamplotter.streamlines:
        pts = streamline.reshape(-1, 1, 2)
        points.extend(np.hstack((pts[:-1], pts[1:])))

    speeds = np.array(speeds)
    widths = np.array(widths)

    vmin = kwargs.pop("vmin", speeds.min())
    vmax = kwargs.pop("vmax", speeds.max())
    norm = kwargs.pop("norm", matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
    cmap = plt.get_cmap(kwargs.pop("cmap", None))

    collection = LineCollection(points, cmap=cmap, norm=norm, linewidth=widths)
    collection.set_array(speeds)
    axes.add_collection(collection)

    _autoscale_view(axes, function.ufl_domain().coordinates.dat.data_ro)
    return collection


def plot(function, *args, num_sample_points=10, **kwargs):
    r"""Plot a 1D Firedrake :class:`~.Function`

    :arg function: The :class:`~.Function` to plot
    :arg args: same as for matplotlib :func:`plot <matplotlib.pyplot.plot>`
    :arg num_sample_points: number of sample points for high-degree functions
    :arg kwargs: same as for matplotlib
    :return: list of matplotlib :class:`Line2D <matplotlib.lines.Line2D>`
    """
    if isinstance(function, MeshGeometry):
        raise TypeError("Expected Function, not Mesh; see firedrake.triplot")

    if function.ufl_domain().geometric_dimension() > 1:
        raise ValueError("Expected 1D Function; for plotting higher-dimensional fields, "
                         "see tricontourf, tripcolor, quiver, trisurf")

    if function.ufl_shape != ():
        raise NotImplementedError("Plotting vector-valued 1D functions is not supported")

    axes = kwargs.pop("axes", None)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)

    if function.ufl_element().degree() < 4:
        result = _bezier_plot(function, axes, **kwargs)
    else:
        degree = function.ufl_element().degree()
        num_sample_points = max((num_sample_points // 3) * 3 + 1, 2 * degree)
        points = calculate_one_dim_points(function, num_sample_points)
        num_cells = function.function_space().mesh().num_cells()
        result = _interp_bezier(points, num_cells, axes, **kwargs)

    _autoscale_view(axes, None)
    return result


def _calculate_values(function, points, dimension, cell_mask=None):
    """Calculate function values at given reference points

    :arg function: function to be sampled
    :arg points: points to be sampled in reference space
    :arg cell_mask: Masks for cell node list
    """
    import numpy.ma as ma
    function_space = function.function_space()
    keys = {1: (0,),
            2: (0, 0)}
    # Such tabulation could be done with FInAT, but that would be more
    # complicated, and there is no clear benefit to changing.
    fiat_element = create_base_element(function_space.ufl_element()).fiat_equivalent
    elem = fiat_element.tabulate(0, points)[keys[dimension]]
    cell_node_list = function_space.cell_node_list
    if cell_mask is not None:
        cell_mask = np.tile(cell_mask.reshape(-1, 1), cell_node_list.shape[1])
        cell_node_list = ma.compress_rows(ma.masked_array(cell_node_list,
                                                          mask=cell_mask))
    data = function.dat.data_ro[cell_node_list]
    if function.ufl_shape == ():
        vec_length = 1
    else:
        vec_length = function.ufl_shape[0]
    if vec_length == 1:
        data = np.reshape(data, data.shape+(1, ))
    return np.einsum("ijk,jl->ilk", data, elem)


def _calculate_points(function, num_points, dimension, cell_mask=None):
    """Calculate points in physical space of given function with given number
    of sampling points at given dimension

    :arg function: function to be sampled
    :arg num_points: number of sampling points
    :arg dimension: dimension of the function
    :arg cell_mask: Masks for cell node list
    """
    function_space = function.function_space()
    mesh = function_space.mesh()
    if mesh.ufl_cell() == Cell('interval'):
        points = np.linspace(0.0, 1.0, num=num_points,
                             dtype=float).reshape(-1, 1)
    elif mesh.ufl_cell() == Cell('quadrilateral'):
        points_1d = np.linspace(0, 1.0, num=num_points,
                                dtype=float).reshape(-1, 1)
        points = np.array(np.meshgrid(points_1d, points_1d)).T.reshape(-1, 2)
    elif mesh.ufl_cell() == Cell('triangle'):
        points_1d = np.linspace(0, 1.0, num=num_points,
                                dtype=float).reshape(-1, 1)
        points_1d_rev = np.fliplr([points_1d]).reshape(-1)
        iu = np.triu_indices(num_points)
        points = np.array(np.meshgrid(points_1d, points_1d_rev)).T[iu]
    else:
        raise NotImplementedError("Unsupported cell type %r", mesh.ufl_cell())
    y_vals = _calculate_values(function, points, dimension, cell_mask)
    x_vals = _calculate_values(mesh.coordinates, points, dimension, cell_mask)
    return x_vals, y_vals


def calculate_one_dim_points(function, num_points, cell_mask=None):
    """Calculate a set of points for plotting for a one-dimension function as
    a numpy array

    :arg function: 1D function for plotting
    :arg num_points: Number of points per element
    :arg cell_mask: Masks for cell node list
    """
    x_vals, y_vals = _calculate_points(function, num_points, 1, cell_mask)
    x_vals = x_vals.reshape(-1)
    y_vals = y_vals.reshape(-1)
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]
    return np.array([x_vals, y_vals])


def _two_dimension_triangle_func_val(function, num_sample_points):
    """Calculate the triangulation and function values for a given 2D function

    :arg function: 2D function
    :arg num_sample_points: Number of sampling points.  This is not
       obeyed exactly, but a linear triangulation is created which
       matches it reasonably well.
    """
    from math import log
    mesh = function.function_space().mesh()
    cell = mesh.ufl_cell()
    if cell.cellname() == "triangle":
        x = np.array([0, 0, 1])
        y = np.array([0, 1, 0])
    elif cell.cellname() == "quadrilateral":
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
    else:
        raise ValueError("Unsupported cell type %s" % cell)

    base_tri = matplotlib.tri.Triangulation(x, y)
    refiner = matplotlib.tri.UniformTriRefiner(base_tri)
    sub_triangles = int(log(num_sample_points, 4))
    tri = refiner.refine_triangulation(False, sub_triangles)
    triangles = tri.get_masked_triangles()

    ref_points = np.dstack([tri.x, tri.y]).reshape(-1, 2)
    z_vals = _calculate_values(function, ref_points, 2)
    coords_vals = _calculate_values(mesh.coordinates, ref_points, 2)

    num_verts = ref_points.shape[0]
    num_cells = function.function_space().cell_node_list.shape[0]
    add_idx = np.arange(num_cells).reshape(-1, 1, 1) * num_verts
    all_triangles = (triangles + add_idx).reshape(-1, 3)

    Z = z_vals.reshape(-1)
    X = coords_vals.reshape(-1, mesh.geometric_dimension())
    return X, Z, all_triangles


def _bezier_calculate_points(function):
    """Calculate points values for a function used for bezier plotting

    :arg function: 1D Function with 1 < deg < 4
    """
    deg = function.function_space().ufl_element().degree()
    M = np.empty([deg + 1, deg + 1], dtype=float)
    finat_element = create_base_element(function.function_space().ufl_element())
    # TODO: Revise this when FInAT gets dual evaluation
    basis = finat_element.fiat_equivalent.dual_basis()
    for i in range(deg + 1):
        for j in range(deg + 1):
            M[i, j] = _bernstein(list(basis[j].get_point_dict().keys())[0][0],
                                 i, deg)
    M_inv = np.linalg.inv(M)
    cell_node_list = function.function_space().cell_node_list
    return np.dot(function.dat.data_ro[cell_node_list], M_inv)


def _bezier_plot(function, axes, **kwargs):
    """Plot a 1D function on a function space with order no more than 4 using
    Bezier curves within each cell

    :arg function: 1D :class:`~.Function` to plot
    :arg axes: :class:`Axes <matplotlib.axes.Axes>` for plotting
    :arg kwargs: additional key work arguments to plot
    :return: matplotlib :class:`PathPatch <matplotlib.patches.PathPatch>`
    """
    deg = function.function_space().ufl_element().degree()
    mesh = function.function_space().mesh()
    if deg == 0:
        V = FunctionSpace(mesh, "DG", 1)
        func = Function(V).interpolate(function)
        return _bezier_plot(func, axes, **kwargs)
    y_vals = _bezier_calculate_points(function)
    x = SpatialCoordinate(mesh)
    coords = Function(FunctionSpace(mesh, 'DG', deg))
    coords.interpolate(x[0])
    x_vals = _bezier_calculate_points(coords)
    vals = np.dstack((x_vals, y_vals))

    codes = {1: [Path.MOVETO, Path.LINETO],
             2: [Path.MOVETO, Path.CURVE3, Path.CURVE3],
             3: [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]}
    vertices = vals.reshape(-1, 2)
    path = Path(vertices, np.tile(codes[deg],
                function.function_space().cell_node_list.shape[0]))

    kwargs["facecolor"] = kwargs.pop("facecolor", "none")
    kwargs["linewidth"] = kwargs.pop("linewidth", 2.)
    patch = matplotlib.patches.PathPatch(path, **kwargs)
    axes.add_patch(patch)
    return patch


def _interp_bezier(pts, num_cells, axes, **kwargs):
    """Interpolate points of a 1D function into piece-wise Bezier curves

    :arg pts: Points of the 1D function evaluated by _calculate_one_dim_points
    :arg num_cells: Number of cells containing the points
    :arg axes: Axes to be plotted on
    :arg kwargs: Addition key word argument for plotting
    """
    pts = pts.T.reshape(num_cells, -1, 2)
    vertices = np.array([]).reshape(-1, 2)
    rows = np.arange(4)
    cols = (np.arange((pts.shape[1] - 1) // 3) * 3).reshape(-1, 1)
    idx = rows + cols

    # For transforming 1D points to Bezier curve
    M = np.array([[1., 0., 0., 0.],
                  [-5/6, 3., -3/2, 1/3],
                  [1/3, -3/2, 3., -5/6],
                  [0., 0., 0., 1.]])

    for i in range(num_cells):
        xs = np.dot(M, pts[i, idx])
        vertices = np.append(vertices, xs.transpose([1, 0, 2]).reshape(-1, 2))

    vertices = vertices.reshape(-1, 2)
    codes = np.tile([Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    vertices.shape[0] // 4)
    path = Path(vertices, codes)

    kwargs["facecolor"] = kwargs.pop("facecolor", "none")
    kwargs["linewidth"] = kwargs.pop("linewidth", 2.)
    patch = matplotlib.patches.PathPatch(path, **kwargs)
    axes.add_patch(patch)
    return patch


def _bernstein(x, k, n):
    """Compute the value of Bernstein polynomial
    (n choose k) * x ^ k * (1 - x) ^ (n - k)

    :arg x: value of x
    :arg k: value of k
    :arg n: value of n
    """
    from math import factorial
    comb = factorial(n) // factorial(k) // factorial(n - k)
    return comb * (x ** k) * ((1 - x) ** (n - k))

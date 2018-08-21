import numpy as np
from ufl import Cell
from tsfc.fiatinterface import create_element
from firedrake import Function, SpatialCoordinate, FunctionSpace
from firedrake.mesh import MeshGeometry

__all__ = ["plot"]


def _plot_mult(functions, num_points=10, axes=None, **kwargs):
    """Plot multiple functions on a figure, return a matplotlib axes

    :arg functions: Functions to be plotted
    :arg num_points: Number of points per element
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, Button
        import matplotlib.image as mpimg
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if len(functions) == 0:
        return None
    interactive = kwargs.pop("interactive", False)
    if interactive:
        return interactive_multiple_plot(functions, num_points, axes=axes, **kwargs)
    import os
    from firedrake import __file__ as firedrake__file__
    figure, ax = plt.subplots()
    func_axis = plt.axes([0.3, 0.025, 0.65, 0.03],
                         axisbg='lightgoldenrodyellow')
    func_slider = Slider(func_axis, "Func Select",
                         0.1, len(functions), valinit=0)
    func_slider.valtext.set_text('0')
    play_axis = plt.axes([0.05, 0.025, 0.1, 0.03])
    play_image = mpimg.imread(os.path.join(
        os.path.dirname(firedrake__file__), 'icons/play.png'))
    play_button = Button(play_axis, "", image=play_image)
    play_axis._button = play_button  # Hacking: keep a reference of button
    plus_axis = plt.axes([0.08, 0.025, 0.1, 0.03])
    plus_image = mpimg.imread(os.path.join(
        os.path.dirname(firedrake__file__), 'icons/plus.png'))
    plus_button = Button(plus_axis, "", image=plus_image)
    plus_axis._button = plus_button  # Hacking: keep a reference of button
    minus_axis = plt.axes([0.02, 0.025, 0.1, 0.03])
    minus_image = mpimg.imread(os.path.join(
        os.path.dirname(firedrake__file__), 'icons/minus.png'))
    minus_button = Button(minus_axis, "", image=minus_image)
    minus_axis._button = minus_button  # Hacking: keep a reference of button

    class Player:
        STOP = 0
        PLAYING = 1
        CLOSED = -1
        status = STOP
        frame_interval = 0.5
    player = Player()  # Use array to allow its value to be changed

    def handle_close(event):
        player.status = Player.CLOSED
    figure.canvas.mpl_connect('close_event', handle_close)

    def update(val):
        val = int(val - 0.1)
        func_slider.valtext.set_text('{:.0f}'.format(val))
        ax.cla()
        plot(functions[val], num_points, ax, **kwargs)
        plt.pause(0.01)
    update(0)
    func_slider.on_changed(update)

    def auto_play(event):
        if player.status == Player.PLAYING:
            player.status = Player.STOP
            return
        curr = 0
        player.status = Player.PLAYING
        while curr < len(functions) and player.status == Player.PLAYING:
            curr += 1
            func_slider.set_val(curr)
            plt.pause(player.frame_interval)
        player.status = Player.STOP
    play_button.on_clicked(auto_play)

    def increase_speed(event):
        player.frame_interval /= 2
    plus_button.on_clicked(increase_speed)

    def decrease_speed(event):
        player.frame_interval *= 2
    minus_button.on_clicked(decrease_speed)

    return ax


def plot_mesh(mesh, axes=None, surface=False, **kwargs):
    """Plot a mesh.

    :arg mesh: The mesh to plot.
    :arg axes: Optional matplotlib axes to draw on.
    :arg surface: Plot surface of mesh only?
    :arg **kwargs: Extra keyword arguments to pass to matplotlib.

    Note that high-order coordinate fields are downsampled to
    piecewise linear first.

    """
    from matplotlib import pyplot as plt
    gdim = mesh.geometric_dimension()
    tdim = mesh.topological_dimension()
    if surface:
        tdim -= 1
    if tdim not in [1, 2]:
        raise NotImplementedError("Not implemented except for %d-dimensional meshes", tdim)
    if gdim == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Line3DCollection as Lines
        projection = "3d"
    else:
        from matplotlib.collections import LineCollection as Lines
        from matplotlib.collections import CircleCollection as Circles
        projection = None
    coordinates = mesh.coordinates
    ele = coordinates.function_space().ufl_element()
    if ele.degree() != 1:
        # Interpolate to piecewise linear.
        from firedrake import VectorFunctionSpace, interpolate
        V = VectorFunctionSpace(mesh, ele.family(), 1)
        coordinates = interpolate(coordinates, V)
    idx = tuple(range(tdim + 1))
    if surface:
        values = coordinates.exterior_facet_node_map().values
        dofs = np.asarray(list(coordinates.function_space().finat_element.entity_closure_dofs()[tdim].values()))
        local_facet = mesh.exterior_facets.local_facet_dat.data_ro
        indices = dofs[local_facet]
        values = np.choose(indices, values[np.newaxis, ...].T)
    else:
        quad = mesh.ufl_cell().cellname() == "quadrilateral"
        values = coordinates.cell_node_map().values
        if tdim == 2 and quad:
            # permute for clockwise ordering
            idx = (0, 1, 3, 2)
    # Plus first vertex again to close loop
    idx = idx + (0, )
    coords = coordinates.dat.data_ro
    if tdim == gdim and tdim == 1:
        # Pad 1D array with zeros
        coords = np.dstack((coords, np.zeros_like(coords))).reshape(-1, 2)
    vertices = coords[values[:, idx]]
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111, projection=projection, **kwargs)

    lines = Lines(vertices)

    if gdim == 3:
        axes.add_collection3d(lines)
    else:
        if not surface:
            points = np.unique(vertices.reshape(-1, gdim), axis=0)
            points = Circles([10] * points.shape[0],
                             offsets=points,
                             transOffset=axes.transData,
                             edgecolors="black", facecolors="black")
            axes.add_collection(points)
        axes.add_collection(lines)
    for setter, idx in zip(["set_xlim",
                            "set_ylim",
                            "set_zlim"],
                           range(coords.shape[1])):
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
    axes.set_aspect("equal")
    return axes


def plot(function_or_mesh,
         num_sample_points=10,
         axes=None,
         plot3d=False,
         **kwargs):
    """Plot a Firedrake object.

    :arg function_or_mesh: The :class:`~.Function` or :func:`~.Mesh`
         to plot.  An iterable of :class:`~.Function`\s may also be
         provided, in which case an animated plot will be available.
    :arg num_sample_points: Number of Sample points per element, ignored if
        degree < 4 where an exact Bezier curve will be used instead of
        sampling at points.  For 2D plots, the number of sampling
        points per element will not exactly this value.  Instead, it
        is used as a guide to the number of subdivisions to use when
        triangulating the surface.
    :arg axes: Axes to be plotted on
    :kwarg plot3d: For 2D plotting, use matplotlib 3D functionality? (slow)
    :kwarg contour: For 2D plotting, True for a contour plot
    :kwarg bezier: For 1D plotting, interpolate using bezier curve instead of
        piece-wise linear
    :kwarg auto_resample: For 1D plotting for functions with degree >= 4,
        resample automatically when zoomed
    :kwarg interactive: For 1D plotting for multiple functions, use an
        interactive inferface in Jupyter Notebook
    :arg kwargs: Additional keyword arguments passed to
        ``matplotlib.plot``.
    """
    # Sanitise input
    if isinstance(function_or_mesh, MeshGeometry):
        # Mesh...
        return plot_mesh(function_or_mesh, axes=axes, **kwargs)
    if not isinstance(function_or_mesh, Function):
        # Maybe an iterable?
        functions = tuple(function_or_mesh)
        if not all(isinstance(f, Function) for f in functions):
            raise TypeError("Expected Function, Mesh, or iterable of Functions, not %r",
                            type(function_or_mesh))
        return _plot_mult(functions, num_sample_points, axes=axes, **kwargs)
    # Single Function...
    function = function_or_mesh
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    gdim = function.ufl_domain().geometric_dimension()
    tdim = function.ufl_domain().topological_dimension()
    if tdim != gdim:
        raise NotImplementedError("Not supported for topological dimension (%d) != geometric dimension (%d)",
                                  tdim, gdim)
    if gdim == 1:
        if function.ufl_shape != ():
            raise NotImplementedError("Plotting vector-valued functions is not supported")
        if function.ufl_element().degree() < 4:
            return bezier_plot(function, axes, **kwargs)
        bezier = kwargs.pop('bezier', False)
        auto_resample = kwargs.pop('auto_resample', False)
        if bezier:
            num_sample_points = (num_sample_points // 3) * 3 + 1 \
                if num_sample_points >= 4 else 4
        points = calculate_one_dim_points(function, num_sample_points)
        cell_boundary = np.fliplr(_get_cell_boundary(function).reshape(-1, 2))
        if axes is None:
            axes = plt.subplot(111)

        def update_points(axes):
            import numpy.ma as ma
            axes.set_autoscale_on(False)
            x_begin = axes.transData.inverted() \
                .transform(axes.transAxes.transform((0, 0)))[0]
            x_end = axes.transData.inverted() \
                .transform(axes.transAxes.transform((1, 0)))[0]
            x_range = np.array([x_begin, x_end])
            cell_intersect = np.empty([cell_boundary.shape[0]])
            for i in range(cell_boundary.shape[0]):
                cell_intersect[i] = _detect_intersection(x_range,
                                                         cell_boundary[i])
            width = plt.gcf().get_size_inches()[0] * plt.gcf().dpi
            cell_mask = 1 - cell_intersect
            cell_width = cell_boundary[:, 1] - cell_boundary[:, 0]
            total_cell_width = ma.masked_array(cell_width,
                                               mask=cell_mask).sum()
            num_points = int(width / cell_intersect.sum()
                             * total_cell_width / (x_end-x_begin))
            if bezier:
                num_points = (num_points // 3) * 3 + 1 \
                    if num_points >= 4 else 4
            points = calculate_one_dim_points(function, num_points, cell_mask)
            axes.cla()
            if bezier:
                interp_bezier(points,
                              int(cell_intersect.sum()),
                              axes, **kwargs)
            else:
                piecewise_linear(points, axes, **kwargs)
            axes.set_xlim(x_range)
            axes.callbacks.connect('xlim_changed', update_points)

        if auto_resample:
            axes.callbacks.connect('xlim_changed', update_points)
        if bezier:
            return interp_bezier(points,
                                 function.function_space().mesh().num_cells(),
                                 axes, **kwargs)
        return piecewise_linear(points, axes, **kwargs)
    elif gdim == 2:
        if len(function.ufl_shape) > 1:
            raise NotImplementedError("Plotting tensor valued functions not supported")
        if len(function.ufl_shape) == 1:
            # Vector-valued, produce a quiver plot interpolated at the
            # mesh coordinates
            coords = function.ufl_domain().coordinates.dat.data_ro
            X, Y = coords.T
            vals = np.asarray(function.at(coords, tolerance=1e-10))
            C = np.linalg.norm(vals, axis=1)
            U, V = vals.T
            if axes is None:
                fig = plt.figure()
                axes = fig.add_subplot(111)
            cmap = kwargs.pop("cmap", cm.coolwarm)
            pivot = kwargs.pop("pivot", "mid")
            mappable = axes.quiver(X, Y, U, V, C, cmap=cmap, pivot=pivot, **kwargs)
            plt.colorbar(mappable)
            return axes
        return two_dimension_plot(function, num_sample_points,
                                  axes, plot3d=plot3d, contour=kwargs.pop("contour", False),
                                  **kwargs)
    else:
        raise NotImplementedError("Plotting functions with geometric dimension %d unsupported",
                                  gdim)


def interactive_multiple_plot(functions, num_sample_points=10, axes=None, **kwargs):
    """Create an interactive plot for multiple 1D functions to be viewed in
    Jupyter Notebook

    :arg functions: 1D Functions to be plotted
    :arg num_sample_points: Number of sample points per element, ignore if
        degree < 4 where Bezier curve is used for an exact plot
    :arg kwargs: Additional key word arguments to be passed to
        ``matplotlib.plot``
    """
    import matplotlib.pyplot as plt
    try:
        from ipywidgets import interact, IntSlider
    except ImportError:
        raise RuntimeError("Not in notebook")

    if axes is None:
        axes = plt.subplot(111)

    def display_plot(index):
        axes.clear()
        return plot(functions[index], num_sample_points, axes=axes, **kwargs).figure

    interact(display_plot, index=IntSlider(min=0, max=len(functions)-1,
                                           step=1, value=0),
             continuous_update=False)
    return axes


def piecewise_linear(points, axes=None, **kwargs):
    """Plot a piece-wise linear plot for the given points, returns a
    matplotlib axes.

    :arg points: Points to be plotted
    :arg axes: Axes to be plotted on
    :arg kwargs: Additional key word arguments passed to plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if axes is None:
        axes = plt.subplot(111)
    axes.plot(points[0], points[1], **kwargs)
    return axes


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
    fiat_element = create_element(function_space.ufl_element(), vector_is_mixed=False)
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
    try:
        from matplotlib.tri import Triangulation, UniformTriRefiner
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if function.function_space().mesh().ufl_cell() == Cell('triangle'):
        x = np.array([0, 0, 1])
        y = np.array([0, 1, 0])
    elif function.function_space().mesh().ufl_cell() == Cell('quadrilateral'):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
    else:
        raise RuntimeError("Unsupported Functionality")
    base_tri = Triangulation(x, y)
    refiner = UniformTriRefiner(base_tri)
    sub_triangles = int(log(num_sample_points, 4))
    tri = refiner.refine_triangulation(False, sub_triangles)
    triangles = tri.get_masked_triangles()
    x_ref = tri.x
    y_ref = tri.y
    num_verts = triangles.max() + 1
    num_cells = function.function_space().cell_node_list.shape[0]
    ref_points = np.dstack([x_ref, y_ref]).reshape(-1, 2)
    z_vals = _calculate_values(function, ref_points, 2)
    coords_vals = _calculate_values(function.function_space().
                                    mesh().coordinates,
                                    ref_points, 2)
    Z = z_vals.reshape(-1)
    X = coords_vals.reshape(-1, 2).T[0]
    Y = coords_vals.reshape(-1, 2).T[1]
    add_idx = np.arange(num_cells).reshape(-1, 1, 1) * num_verts
    all_triangles = (triangles + add_idx).reshape(-1, 3)
    triangulation = Triangulation(X, Y, triangles=all_triangles)
    return triangulation, Z


def two_dimension_plot(function,
                       num_sample_points,
                       axes=None,
                       plot3d=False,
                       contour=False,
                       **kwargs):
    """Plot a 2D function as surface plotting, return the axes drawn on.

    :arg function: 2D function for plotting
    :arg num_sample_points: Number of sample points per element
    :arg axes: Axes to be plotted on
    :kwarg plot3d: Use 3D projection (slow for large meshes).
    :kwarg contour: Produce contour plot?
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib import cm
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    triangulation, Z = _two_dimension_triangle_func_val(function,
                                                        num_sample_points)

    if axes is None:
        figure = plt.figure()
        if plot3d:
            axes = figure.add_subplot(111, projection="3d")
        else:
            axes = figure.add_subplot(111)
    cmap = kwargs.pop('cmap', cm.coolwarm)
    if plot3d:
        if contour:
            mappable = axes.tricontour(triangulation, Z, edgecolor="none",
                                       cmap=cmap, antialiased=False, **kwargs)
        else:
            mappable = axes.plot_trisurf(triangulation, Z, edgecolor="none",
                                         cmap=cmap, antialiased=False,
                                         shade=False, **kwargs)
        if cmap is not None:
            plt.colorbar(mappable)
        return axes
    else:
        if contour:
            mappable = axes.tricontour(triangulation, Z, edgecolor="none",
                                       cmap=cmap, **kwargs)
        else:
            mappable = axes.tripcolor(triangulation, Z, cmap=cmap, **kwargs)
        if cmap is not None:
            plt.colorbar(mappable)
    return axes


def _bezier_calculate_points(function):
    """Calculate points values for a function used for bezier plotting

    :arg function: 1D Function with 1 < deg < 4
    """
    deg = function.function_space().ufl_element().degree()
    M = np.empty([deg + 1, deg + 1], dtype=float)
    fiat_element = create_element(function.function_space().ufl_element(), vector_is_mixed=False)
    basis = fiat_element.dual_basis()
    for i in range(deg + 1):
        for j in range(deg + 1):
            M[i, j] = _bernstein(list(basis[j].get_point_dict().keys())[0][0],
                                 i, deg)
    M_inv = np.linalg.inv(M)
    cell_node_list = function.function_space().cell_node_list
    return np.dot(function.dat.data_ro[cell_node_list], M_inv)


def bezier_plot(function, axes=None, **kwargs):
    """Plot a 1D function on a function space with order no more than 4 using
    Bezier curve within each cell, return a matplotlib axes

    :arg function: 1D function for plotting
    :arg axes: Axes for plotting, if None, a new one will be created
    :arg kwargs: additional key work arguments to plot
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")

    deg = function.function_space().ufl_element().degree()
    mesh = function.function_space().mesh()
    if deg == 0:
        V = FunctionSpace(mesh, "DG", 1)
        func = Function(V).interpolate(function)
        return bezier_plot(func, axes, **kwargs)
    y_vals = _bezier_calculate_points(function)
    x = SpatialCoordinate(mesh)
    coords = Function(FunctionSpace(mesh, 'DG', deg))
    coords.interpolate(x[0])
    x_vals = _bezier_calculate_points(coords)
    vals = np.dstack((x_vals, y_vals))

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)
    codes = {1: [Path.MOVETO, Path.LINETO],
             2: [Path.MOVETO, Path.CURVE3, Path.CURVE3],
             3: [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]}
    vertices = vals.reshape(-1, 2)
    path = Path(vertices, np.tile(codes[deg],
                function.function_space().cell_node_list.shape[0]))
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    axes.add_patch(patch)
    axes.plot(**kwargs)
    return axes


def interp_bezier(pts, num_cells, axes=None, **kwargs):
    """Interpolate points of a 1D function into piece-wise Bezier curves

    :arg pts: Points of the 1D function evaluated by _calculate_one_dim_points
    :arg num_cells: Number of cells containing the points
    :arg axes: Axes to be plotted on
    :arg kwargs: Addition key word argument for plotting
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")

    pts = pts.T.reshape(num_cells, -1, 2)
    vertices = np.array([]).reshape(-1, 2)
    rows = np.arange(4)
    cols = (np.arange((pts.shape[1] - 1) // 3) * 3).reshape(-1, 1)
    idx = rows + cols
    for i in range(num_cells):
        vertices = np.append(vertices,
                             _points_to_bezier_curve(pts[i, idx])
                             .transpose([1, 0, 2]).reshape(-1, 2))
    vertices = vertices.reshape(-1, 2)
    codes = np.tile([Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    vertices.shape[0] // 4)
    path = Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    axes.add_patch(patch)
    axes.plot(**kwargs)
    return axes


def _points_to_bezier_curve(pts):
    """Transform 4 points on a function into cubic Bezier curve control points
    In a cubic Bezier curve: P(t) = (1 - t) ^ 3 * P_0
                                  + 3 * t * (1 - t) ^ 2 * P_1
                                  + 3 * t ^ 2 * (1 - t) * P_2
                                  + t ^ 3 * P_3
    Input points are interpolated as P(0), P(1/3), P(2/3) and P(1)
    Return control points P_0, P_1, P_2, P_3

    :arg pts: Points on a 1D function
    """
    M = np.array([[1., 0., 0., 0.],
                  [-5/6, 3., -3/2., 1/3.],
                  [1/3, -3/2, 3., -5/6],
                  [0., 0., 0., 1.]])
    return np.dot(M, pts)


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


def _get_cell_boundary(function):
    """Compute the x-coordinate value of boundaries of cells of a function

    :arg function: the function for which cell boundary is to be computed
    """
    coords = function.function_space().mesh().coordinates
    return _calculate_values(coords, np.array([[0.0], [1.0]], dtype=float), 1)


def _detect_intersection(interval1, interval2):
    """Detect intersection of two intervals

    :arg interval1: Interval 1 as numpy array [x1, x2]
    :arg interval2: Interval 2 as numpy array [y1, y2]
    """
    return np.less_equal(np.amax([interval1[0], interval2[0]]),
                         np.amin([interval1[1], interval2[1]]))

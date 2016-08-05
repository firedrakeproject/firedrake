from __future__ import absolute_import
import numpy as np
from ufl import Cell
from firedrake import Function, SpatialCoordinate, FunctionSpace

__all__ = ["plot", "interactive_multiple_plot"]


def _plot_mult(functions, num_points=10, **kwargs):
    """Plot multiple functions on a figure, return a matplotlib figure

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
    is_closed = [False]  # Use array to allow its value to be changed

    def handle_close(event):
        is_closed[0] = True
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
        curr = 0
        while curr < len(functions) and not is_closed[0]:
            curr += 1
            func_slider.set_val(curr)
            plt.pause(0.5)
    play_button.on_clicked(auto_play)

    return figure


def plot(function,
         num_sample_points=100,
         axes=None,
         **kwargs):
    """Plot a function or a list of functions and return a matplotlib
    figure object.

    :arg function: The function to plot.
    :arg num_sample_points: Number of Sample points per element, ignored if
        degree < 4 where an exact Bezier curve will be used instead of
        sampling at points
    :arg axes: Axes to be plotted on
    :kwarg contour: For 2D plotting, True for a contour plot
    :kwarg bezier: For 1D plotting, interpolate using bezier curve instead of
        piece-wise linear
    :kwarg auto_resample: For 1D plotting for functions with degree >= 4,
        resample automatically when zoomed
    :arg kwargs: Additional keyword arguments passed to
        ``matplotlib.plot``.
    """
    if not isinstance(function, Function):
        if isinstance(function, list) and \
                all(isinstance(f, Function) for f in function):
            return _plot_mult(function, num_sample_points, **kwargs)
        raise TypeError("Unexpected type")
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if function.function_space().mesh().geometric_dimension() \
            == function.function_space().mesh().topological_dimension() \
            == 1:
        if function.function_space().ufl_element().degree() < 4:
            return bezier_plot(function, axes, **kwargs)
        bezier = kwargs.pop('bezier', False)
        auto_resample = kwargs.pop('auto_resample', False)
        if bezier:
            num_sample_points = int(num_sample_points / 3) * 3 + 1 \
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
                num_points = int(num_points / 3) * 3 + 1 \
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
    elif function.function_space().mesh().geometric_dimension() \
            == function.function_space().mesh().topological_dimension() \
            == 2:
        is_contour = kwargs.pop('contour', False)
        if is_contour:
            return two_dimension_contour(function, num_sample_points,
                                         axes, **kwargs)
        else:
            return two_dimension_surface(function, num_sample_points,
                                         axes, **kwargs)
    else:
        raise RuntimeError("Unsupported functionality")


def interactive_multiple_plot(functions, num_sample_points=10, **kwargs):
    """Create an interactive plot for multiple 1D functions to be viewed in
    Jupyter Notebook

    :arg functions: 1D Functions to be plotted
    :arg num_sample_points: Number of sample points per element, ignore if
        degree < 4 where Bezier curve is used for an exact plot
    :arg kwargs: Additional key word arguments to be passed to
        ``matplotlib.plot``
    """
    try:
        from ipywidgets import interact, IntSlider
    except ImportError:
        raise RuntimeError("Not in notebook")

    def display_plot(index):
        return plot(functions[index], num_sample_points, **kwargs)

    interact(display_plot, index=IntSlider(min=0, max=len(functions)-1,
                                           step=1, value=0))


def piecewise_linear(points, axes=None, **kwargs):
    """Plot a piece-wise linear plot for the given points, returns a
    matplotlib figure

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
    return plt.gcf()


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
    elem = function_space.fiat_element.tabulate(0, points)[keys[dimension]]
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
        raise RuntimeError("Unsupported functionality")
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
    :arg num_sample_points: Number of sampling points
    """
    from math import log, sqrt
    try:
        from matplotlib.tri import Triangulation, UniformTriRefiner
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if function.function_space().mesh().cell() == Cell('triangle'):
        x = np.array([0, 0, 1])
        y = np.array([0, 1, 0])
    elif function.function_space().mesh().cell() == Cell('quadrilateral'):
        x = np.array([0, 0, 1, 1])
        y = np.array([0, 1, 0, 1])
    else:
        raise RuntimeError("Unsupported Functionality")
    base_tri = Triangulation(x, y)
    refiner = UniformTriRefiner(base_tri)
    sub_triangles = int(log(sqrt(num_sample_points), 4))
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


def two_dimension_surface(function,
                          num_sample_points,
                          axes=None,
                          **kwargs):
    """Plot a 2D function as surface plotting, return a matplotlib figure

    :arg function: 2D function for plotting
    :arg num_sample_points: Number of sample points per element
    :arg axes: Axes to be plotted on
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    triangulation, Z = _two_dimension_triangle_func_val(function,
                                                        num_sample_points)

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
    cmap = kwargs.pop('cmap', cm.coolwarm)
    axes.plot_trisurf(triangulation, Z, edgecolor='none',
                      antialiased=False, shade=False, cmap=cmap,
                      **kwargs)
    return plt.gcf()


def two_dimension_contour(function,
                          num_sample_points,
                          axes=None,
                          **kwargs):
    """Plot a 2D function as contour plotting, return a matplotlib figure

    :arg function: 2D function for plotting
    :arg num_sample_points: Number of sample points per element
    :arg axes: Axes to be plotted on
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    triangulation, Z = _two_dimension_triangle_func_val(function,
                                                        num_sample_points)

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
    axes.tricontour(triangulation, Z, edgecolor='none',
                    antialiased=False, **kwargs)
    return plt.gcf()


def _bezier_calculate_points(function):
    """Calculate points values for a function used for bezier plotting

    :arg function: 1D Function with 1 < deg < 4
    """
    deg = function.function_space().ufl_element().degree()
    M = np.empty([deg + 1, deg + 1], dtype=float)
    basis = function.function_space().fiat_element.dual_basis()
    for i in range(deg + 1):
        for j in range(deg + 1):
            M[i, j] = _bernstein(basis[j].get_point_dict().keys()[0][0],
                                 i, deg)
    M_inv = np.linalg.inv(M)
    cell_node_list = function.function_space().cell_node_list
    return np.dot(function.dat.data_ro[cell_node_list], M_inv)


def bezier_plot(function, axes=None, **kwargs):
    """Plot a 1D function on a function space with order no more than 4 using
    Bezier curve within each cell, return a matplotlib figure

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
    return plt.gcf()


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
    cols = (np.arange((pts.shape[1] - 1) / 3) * 3).reshape(-1, 1)
    idx = rows + cols
    for i in range(num_cells):
        vertices = np.append(vertices,
                             _points_to_bezier_curve(pts[i, idx])
                             .transpose([1, 0, 2]).reshape(-1, 2))
    vertices = vertices.reshape(-1, 2)
    codes = np.tile([Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    vertices.shape[0] / 4)
    path = Path(vertices, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    axes.add_patch(patch)
    axes.plot(**kwargs)
    return plt.gcf()


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
                  [-5./6., 3., -3./2., 1./3.],
                  [1./3., -3./2., 3., -5./6.],
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
    comb = factorial(n) / factorial(k) / factorial(n - k)
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

from __future__ import absolute_import
import numpy as np
from ufl import cell
from firedrake import Function, SpatialCoordinate


def plot_mult(functions, num_points=100):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    figure, ax = plt.subplots()
    func_axis = plt.axes([0.25, 0.025, 0.65, 0.03], axisbg='lightgoldenrodyellow')
    func_slider = Slider(func_axis, "Func Select", 0.1, len(functions), valinit=0)
    func_slider.valtext.set_text('0')

    def _update(val):
        val = int(val - 0.1)
        func_slider.valtext.set_text('{:.0f}'.format(val))
        ax.cla()
        plot(functions[val], ax)
    _update(0)
    func_slider.on_changed(_update)
    return figure


def plot(function, axes=None, num_points=100, **kwargs):
    """Plot a function and return a matplotlib figure object.
    :arg function: The function to plot.
    :arg num_points: Number of points per element
    :arg kwargs: Additional keyword arguments passed to
    ``matplotlib.plot``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if function.function_space().mesh().ufl_cell() == cell.Cell("interval"):
        if function.function_space().ufl_element().degree() < 4:
            return bezier_plot(function, axes)
        points = one_dimension_plot(function, num_points)
    else:
        raise RuntimeError("Unsupported functionality")
    if axes is None:
        axes = plt.subplot(111)
    axes.plot(points[0], points[1], **kwargs)
    return plt.gcf()


def one_dimension_plot(function, num_points):
    """Calculate a set of points for plotting for a one-dimension function as a
    numpy array

    :arg function: 1D function for plotting
    :arg num_points: number of points per element
    """
    function_space = function.function_space()
    mesh = function_space.mesh()

    def __calculate_values(function, function_space, points):
        "Calculate function values at given points"
        elem = function_space.fiat_element.tabulate(0, points)[(0, )]
        data = function.dat.data_ro[function_space.cell_node_list]
        return np.dot(data, elem).reshape(-1)

    points = np.linspace(0, 1.0, num=num_points, dtype=float).reshape(-1, 1)
    y_vals = __calculate_values(function, function_space, points)
    x_vals = __calculate_values(mesh.coordinates,
                                mesh.coordinates.function_space(),
                                points)

    def __sort_points(x_vals, y_vals):
        "Sort the points according to x values"
        order = np.argsort(x_vals)
        return np.array([x_vals[order], y_vals[order]])
    x_vals, y_vals = __sort_points(x_vals, y_vals)

    return np.array([x_vals, y_vals])


def bezier_plot(function, axes=None):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")

    deg = function.function_space().ufl_element().degree()
    M = np.empty([deg + 1, deg + 1], dtype=float)
    basis = function.function_space().fiat_element.dual_basis()
    for i in range(deg + 1):
        for j in range(deg + 1):
            M[i, j] = _bernstein(basis[j].get_point_dict().keys()[0][0], i, deg)
    M_inv = np.linalg.inv(M)
    cell_node_list = function.function_space().cell_node_list
    y_vals = np.dot(function.dat.data_ro[cell_node_list], M_inv)
    x = SpatialCoordinate(function.function_space().mesh())
    coords = Function(function.function_space())
    coords.interpolate(x[0])
    x_vals = np.dot(coords.dat.data_ro[cell_node_list], M_inv)
    vals = np.dstack((x_vals, y_vals))

    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111)
    codes = {1: [Path.MOVETO, Path.LINETO],
             2: [Path.MOVETO, Path.CURVE3, Path.CURVE3],
             3: [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]}
    vertices = vals.reshape(-1, 2)
    path = Path(vertices, np.tile(codes[deg], cell_node_list.shape[0]))
    patch = patches.PathPatch(path, facecolor='none', lw=2)
    axes.add_patch(patch)
    axes.plot()
    return plt.gcf()


def _bernstein(x, k, n):
    from scipy.misc import comb
    return comb(n, k) * (x ** k) * ((1 - x) ** (n - k))

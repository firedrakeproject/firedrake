from __future__ import absolute_import
import numpy as np
from ufl import cell
from firedrake import Function, SpatialCoordinate

__all__ = ["plot"]


def _plot_mult(functions, num_points=100, **kwargs):
    """Plot multiple functions on a figure, return a matplotlib figure
    :arg functions: Functions to be plotted
    :arg num_points: Number of points per element
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if len(functions) == 0:
        return None
    figure, ax = plt.subplots()
    func_axis = plt.axes([0.25, 0.025, 0.65, 0.03], axisbg='lightgoldenrodyellow')
    func_slider = Slider(func_axis, "Func Select", 0.1, len(functions), valinit=0)
    func_slider.valtext.set_text('0')

    def update(val):
        val = int(val - 0.1)
        func_slider.valtext.set_text('{:.0f}'.format(val))
        ax.cla()
        plot(functions[val], ax, **kwargs)
    update(0)
    func_slider.on_changed(update)
    return figure


def plot(function, axes=None, num_points=100, **kwargs):
    """Plot a function or a list of functions and return a matplotlib
    figure object.
    :arg function: The function to plot.
    :arg axes: Axes to be plotted on
    :arg num_points: Number of points per element, ignored if degree < 4 where
        Bezier curve will be used instead of sampling at points
    :arg kwargs: Additional keyword arguments passed to
        ``matplotlib.plot``.
    """

    if not isinstance(function, Function):
        return _plot_mult(function, num_points, **kwargs)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if function.function_space().mesh().ufl_cell() == cell.Cell("interval"):
        if function.function_space().ufl_element().degree() < 4:
            return bezier_plot(function, axes, **kwargs)
        points = calculate_one_dim_points(function, num_points)
        if axes is None:
            axes = plt.subplot(111)
        axes.plot(points[0], points[1], **kwargs)
        return plt.gcf()
    else:
        raise RuntimeError("Unsupported functionality")


def _calculate_values(function, points):
    """Calculate function values at given reference points
    :arg function: function to be sampled
    :arg points: points to be sampled in reference space
    """
    function_space = function.function_space()
    elem = function_space.fiat_element.tabulate(0, points)[(0, )]
    data = function.dat.data_ro[function_space.cell_node_list]
    return np.dot(data, elem).reshape(-1)


def _calculate_points(function, num_points):
    """Calculate points in physical space of given function with given number of
    sampling points at given dimension
    :arg function: function to be sampled
    :arg num_points: number of sampling points
    """
    mesh = function.function_space().mesh()
    points = np.linspace(0, 1.0, num=num_points, dtype=float).reshape(-1, 1)
    y_vals = _calculate_values(function, points)
    x_vals = _calculate_values(mesh.coordinates, points)
    return x_vals, y_vals


def calculate_one_dim_points(function, num_points):
    """Calculate a set of points for plotting for a one-dimension function as a
    numpy array

    :arg function: 1D function for plotting
    :arg num_points: Number of points per element
    """
    x_vals, y_vals = _calculate_points(function, num_points)
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]
    return np.array([x_vals, y_vals])


def bezier_plot(function, axes=None, **kwargs):
    """Plot a 1D function on a function space with order no more than 4 using
    Bezier curve within each cell, return a matplotlib figure

    :arg function: 1D function for plotting
    :arg Axes: Axes for plotting, if None, a new one will be created
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")

    deg = function.function_space().ufl_element().degree()
    if deg == 0:
        from firedrake import FunctionSpace
        mesh = function.function_space().mesh()
        V = FunctionSpace(mesh, "DG", 1)
        func = Function(V).interpolate(function)
        return bezier_plot(func, axes, **kwargs)
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
    axes.plot(**kwargs)
    return plt.gcf()


def _bernstein(x, k, n):
    from math import factorial
    comb = factorial(n) / factorial(k) / factorial(n - k)
    return comb * (x ** k) * ((1 - x) ** (n - k))

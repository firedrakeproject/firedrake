from __future__ import absolute_import
import numpy as np
from ufl import cell


def plot(function, num_points, **kwargs):
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
        points = one_dimension_plot(function, num_points)
    else:
        raise RuntimeError("Unsupported functionality")
    return plt.plot(points[0], points[1], **kwargs)


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

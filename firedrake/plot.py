import numpy as np


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
        vals = np.array([], dtype=float)
        for cell_node in function_space.cell_node_list:
            data = function.dat.data_ro[cell_node]
            elem = function_space.fiat_element.tabulate(0, points)[(0,)]
            val = np.dot(data, elem)
            vals = np.append(vals, [val])
        return vals

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

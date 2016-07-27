from __future__ import absolute_import
import numpy as np
from ufl import Cell
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
        plot(functions[val], num_points, ax, **kwargs)
    update(0)
    func_slider.on_changed(update)
    return figure


def plot(function,
         num_sample_points=100,
         axes=None,
         **kwargs):
    """Plot a function or a list of functions and return a matplotlib
    figure object.
    :arg function: The function to plot.
    :arg num_sample_points: Number of Sample points per element, ignored if
        degree < 4 where Bezier curve will be used instead of sampling at
        points
    :arg axes: Axes to be plotted on
    :arg kwargs: Additional keyword arguments passed to
        ``matplotlib.plot``.
    """

    if not isinstance(function, Function):
        return _plot_mult(function, num_sample_points, **kwargs)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise RuntimeError("Matplotlib not importable, is it installed?")
    if function.function_space().mesh().geometric_dimension() \
            == function.function_space().mesh().topological_dimension() \
            == 1:
        if function.function_space().ufl_element().degree() < 4:
            return bezier_plot(function, axes, **kwargs)
        points = calculate_one_dim_points(function, num_sample_points)
        if axes is None:
            axes = plt.subplot(111)
        axes.plot(points[0], points[1], **kwargs)
        return plt.gcf()
    elif function.function_space().mesh().geometric_dimension() \
            == function.function_space().mesh().topological_dimension() \
            == 2:
        return two_dimension_plot(function, num_sample_points, axes,
                                  **kwargs)
    else:
        raise RuntimeError("Unsupported functionality")


def _calculate_values(function, points, dimension):
    """Calculate function values at given reference points
    :arg function: function to be sampled
    :arg points: points to be sampled in reference space
    """
    function_space = function.function_space()
    keys = {1: (0,),
            2: (0, 0)}
    elem = function_space.fiat_element.tabulate(0, points)[keys[dimension]]
    data = function.dat.data_ro[function_space.cell_node_list]
    if function.ufl_shape == ():
        vec_length = 1
    else:
        vec_length = function.ufl_shape[0]
    if vec_length == 1:
        data = np.reshape(data, data.shape+(1, ))
    return np.einsum("ijk,jl->ilk", data, elem)


def _calculate_points(function, num_points, dimension):
    """Calculate points in physical space of given function with given number of
    sampling points at given dimension
    :arg function: function to be sampled
    :arg num_points: number of sampling points
    :arg dimension: dimension of the function
    """
    function_space = function.function_space()
    mesh = function_space.mesh()
    if mesh.ufl_cell() == Cell('interval'):
        points = np.linspace(0, 1.0, num=num_points, dtype=float).reshape(-1, 1)
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
    y_vals = _calculate_values(function, function_space, points, dimension)
    x_vals = _calculate_values(mesh.coordinates,
                               mesh.coordinates.function_space(),
                               points, dimension)
    return x_vals, y_vals


def calculate_one_dim_points(function, num_points):
    """Calculate a set of points for plotting for a one-dimension function as a
    numpy array

    :arg function: 1D function for plotting
    :arg num_points: Number of points per element
    """
    x_vals, y_vals = _calculate_points(function, num_points, 1)
    x_vals = x_vals.reshape(-1)
    y_vals = y_vals.reshape(-1)
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]
    return np.array([x_vals, y_vals])


def two_dimension_plot(function,
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
        from matplotlib.tri import Triangulation, UniformTriRefiner
    except ImportError:
        raise RuntimeError("Matplotlib or Scipy not importable, is it installed?")

    x = np.array([0, 0, 1])
    y = np.array([0, 1, 0])
    base_tri = Triangulation(x, y)
    refiner = UniformTriRefiner(base_tri)
    tri = refiner.refine_triangulation(False, num_sample_points)
    triangles = tri.get_masked_triangles()
    x_ref = tri.x
    y_ref = tri.y
    num_verts = triangles.max() + 1
    num_cells = function.function_space().cell_node_list.shape[0]
    ref_points = np.dstack([x_ref, y_ref]).reshape(-1, 2)
    z_vals = _calculate_values(function, function.function_space(), ref_points, 2)
    coords_vals = _calculate_values(function.function_space().mesh().coordinates,
                                    function.function_space().mesh().coordinates.function_space(),
                                    ref_points, 2)
    Z = z_vals.reshape(-1)
    X = coords_vals.reshape(-1, 2).T[0]
    Y = coords_vals.reshape(-1, 2).T[1]
    all_triangles = triangles.copy()
    temp = triangles.copy()
    for i in range(num_cells - 1):
        temp = temp + num_verts
        all_triangles = np.append(all_triangles, temp)
    all_triangles = all_triangles.reshape(-1, 3)
    if axes is None:
        figure = plt.figure()
        axes = figure.add_subplot(111, projection='3d')
    triangulation = Triangulation(X, Y, triangles=all_triangles)
    axes.plot_trisurf(triangulation, Z, edgecolor='none', antialiased=False, **kwargs)
    return plt.gcf()


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
    from FIAT.factorial import factorial
    comb = factorial(n) / factorial(k) / factorial(n - k)
    return comb * (x ** k) * ((1 - x) ** (n - k))

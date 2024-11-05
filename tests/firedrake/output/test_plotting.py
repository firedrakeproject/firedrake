import pytest
import numpy as np
from firedrake import *
from firedrake.__future__ import *

try:
    from firedrake.pyplot import *
    import matplotlib.pyplot as plt
    import matplotlib.colors
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    # Matplotlib is not installed
    pytest.skip("Matplotlib not installed", allow_module_level=True)


def test_plotting_1d():
    mesh = UnitIntervalMesh(32)
    CG = FunctionSpace(mesh, "CG", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    x, = SpatialCoordinate(mesh)

    u = assemble(interpolate(8 * x * (1 - x), CG))
    v = project(8 * x * (1 - x), DG)

    patches = plot(u, edgecolor='black', linestyle='--')
    assert patches is not None

    fig, axes = plt.subplots()
    plot(v, edgecolor='tab:green', label='v', axes=axes)
    legend = axes.legend(loc='upper right')
    assert len(legend.get_texts()) == 1


def test_plotting_1d_high_degree():
    mesh = UnitIntervalMesh(2)
    V8 = FunctionSpace(mesh, "DG", 8)
    V12 = FunctionSpace(mesh, "DG", 12)
    x, = SpatialCoordinate(mesh)

    expr = conditional(x < .5, 2**17 * x**4 * (0.5 - x)**4, 1)
    u = project(expr, V8)
    v = project(expr, V12)
    fig, axes = plt.subplots()
    patches = plot(u, edgecolor='tab:blue', axes=axes)
    assert patches is not None
    patches = plot(v, linestyle='--', edgecolor='tab:orange', axes=axes)
    assert patches is not None


def test_plot_wrong_inputs():
    mesh = UnitSquareMesh(32, 32)
    with pytest.raises(TypeError):
        plot(mesh)

    Q = FunctionSpace(mesh, family='CG', degree=1)
    x, y = SpatialCoordinate(mesh)
    q = assemble(interpolate(x - y, Q))
    with pytest.raises(ValueError):
        plot(q)


def test_plotting_scalar_field():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] + x[1])

    # Plot without first creating axes
    contours = tricontour(f)

    # Create axes first then plot
    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)
    contours = tricontour(f, axes=axes[0])
    assert contours is not None
    assert not contours.filled
    fig.colorbar(contours, ax=axes[0])

    filled_contours = tricontourf(f, axes=axes[1])
    assert filled_contours is not None
    assert filled_contours.filled
    fig.colorbar(filled_contours, ax=axes[1])


def test_tripcolor_shading():
    mesh = UnitSquareMesh(10, 10)
    x = SpatialCoordinate(mesh)

    V0 = FunctionSpace(mesh, "DG", 0)
    f0 = Function(V0)

    V1 = FunctionSpace(mesh, "DG", 1)
    f1 = Function(V1)

    f0.project(x[0] + x[1])
    f1.project(x[0] + x[1])

    fig, axes = plt.subplots(ncols=3, sharex=True, sharey=True)

    collection = tripcolor(f0, num_sample_points=1, axes=axes[0])
    assert collection.get_array().shape == f0.dat.data_ro[:].shape

    collection = tripcolor(f1, num_sample_points=1, axes=axes[1])
    assert collection.get_array().shape == f1.dat.data_ro[:].shape

    collection = tripcolor(f1, num_sample_points=1, shading="flat", axes=axes[2])
    assert collection.get_array().shape == f0.dat.data_ro[:].shape


def test_plotting_quadratic():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] ** 2 + x[1] ** 2)

    fig, axes = plt.subplots()
    contours = tricontour(f, axes=axes)
    assert contours is not None


def test_tricontour_quad_mesh():
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    V = FunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] ** 2 + x[1] ** 2)

    fig, axes = plt.subplots()
    contours = tricontourf(f, axes=axes)
    colorbar = fig.colorbar(contours)
    assert contours is not None
    assert colorbar is not None


def test_tricontour_extruded_mesh():
    nx = 12
    Lx = Constant(3.0)
    interval = IntervalMesh(nx, float(Lx))
    rectangle = ExtrudedMesh(interval, 1)
    Vc = rectangle.coordinates.function_space()
    x = SpatialCoordinate(rectangle)
    expr = as_vector((x[0], (1 - 0.5 * x[0] / Lx) * x[1] + 0.25 * x[0] / Lx))
    f = assemble(interpolate(expr, Vc))
    mesh = Mesh(f)

    V = FunctionSpace(mesh, "CG", 1, vfamily="CG", vdegree=1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] + x[1])

    fig, axes = plt.subplots()
    contours = tricontourf(f, axes=axes)
    colorbar = fig.colorbar(contours)
    assert contours is not None
    assert colorbar is not None


def test_quiver_plot():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(as_vector((-x[1], x[0])))

    fig, axes = plt.subplots()
    arrows = quiver(f, axes=axes)
    assert arrows is not None
    fig.colorbar(arrows)


def test_streamplot():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)
    x = SpatialCoordinate(mesh)
    x0 = Constant((.5, .5))
    v = x - x0

    center = assemble(interpolate(2 * as_vector((-v[1], v[0])), V))
    saddle = assemble(interpolate(2 * as_vector((v[0], -v[1])), V))
    r = Constant(.5)
    sink = assemble(interpolate(center - r * v, V))

    fig, axes = plt.subplots(ncols=1, nrows=3, sharex=True, sharey=True)
    for ax in axes:
        ax.set_aspect("equal")

    color_norm = matplotlib.colors.PowerNorm(gamma=0.5)
    kwargses = [
        {'resolution': 1/48, 'tolerance': 2e-2, 'norm': color_norm, 'seed': 0},
        {'loc_tolerance': 1e-5, 'cmap': 'bone', 'vmax': 1., 'seed': 0},
        {'min_length': 1/4, 'max_time': 5., 'seed': 0}
    ]
    for ax, function, kwargs in zip(axes, [center, saddle, sink], kwargses):
        lines = streamplot(function, axes=ax, **kwargs)
        colorbar = fig.colorbar(lines, ax=ax)
        assert lines is not None
        assert colorbar is not None


def test_plotting_vector_field():
    mesh = UnitSquareMesh(10, 10)
    V = VectorFunctionSpace(mesh, "CG", 1)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(as_vector((-x[1], x[0])))

    fig, axes = plt.subplots()
    contours = tricontourf(f, axes=axes)
    assert contours is not None
    fig.colorbar(contours)


def test_triplot():
    mesh = UnitSquareMesh(10, 10)
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    lines = triplot(mesh, axes=axes[0])
    assert lines
    legend = axes[0].legend(loc='upper right')
    assert len(legend.get_texts()) == 4

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    lines = triplot(mesh, axes=axes[1], interior_kw={'linewidths': 0.5},
                    boundary_kw={'linewidths': 2.0, 'colors': colors})
    legend = axes[1].legend(loc='upper right')
    assert len(legend.get_texts()) == 4


def test_triplot_quad_mesh():
    mesh = UnitSquareMesh(10, 10, quadrilateral=True)
    fig, axes = plt.subplots()
    lines = triplot(mesh, axes=axes, interior_kw={'facecolors': 'tab:blue'})
    assert lines
    legend = axes.legend(loc='upper right')
    assert len(legend.get_texts()) > 0


def test_triplot_3d():
    fig = plt.figure()

    axes = fig.add_subplot(2, 2, 1, projection='3d')
    mesh = CylinderMesh(nr=32, nl=4)
    collections = triplot(mesh, axes=axes, boundary_kw={'colors': ['r', 'g']})
    assert collections
    legend = axes.legend()
    assert len(legend.get_texts()) == 2

    axes = fig.add_subplot(2, 2, 2, projection='3d')
    mesh = UnitIcosahedralSphereMesh(3)
    triplot(mesh, axes=axes, interior_kw={'edgecolors': 'white'})
    legend = axes.legend()
    assert len(legend.get_texts()) == 0

    axes = fig.add_subplot(2, 2, 3, projection='3d')
    mesh = UnitCubedSphereMesh(3)
    interior_kw = {'facecolors': 'tab:orange', 'alpha': 0.5}
    triplot(mesh, axes=axes, interior_kw=interior_kw)

    axes = fig.add_subplot(2, 2, 4, projection='3d')
    mesh = UnitCubeMesh(3, 3, 3)
    colors = ['red', 'green', 'blue', 'orange', 'yellow', 'purple']
    boundary_kw = {'facecolors': colors, 'alpha': 0.85, 'linewidths': 0.1}
    collections = triplot(mesh, axes=axes, boundary_kw=boundary_kw)
    assert collections
    legend = axes.legend(loc='upper right')
    assert len(legend.get_texts()) == 6


def test_trisurf():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] ** 2 + x[1] ** 2)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    assert isinstance(axes, Axes3D)
    collection = trisurf(f, axes=axes)
    assert collection is not None


def test_trisurf3d():
    mesh = UnitIcosahedralSphereMesh(2)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] * x[1] * x[2])

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    collection = trisurf(f, axes=axes)
    assert collection is not None


def test_trisurf3d_quad():
    mesh = UnitCubedSphereMesh(2)
    V = FunctionSpace(mesh, "CG", 2)
    f = Function(V)
    x = SpatialCoordinate(mesh)
    f.interpolate(x[0] * x[1] * x[2])

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    collection = trisurf(f, axes=axes)
    assert collection is not None


def test_tripcolor_movie():
    mesh = UnitSquareMesh(16, 16)
    Q = FunctionSpace(mesh, 'CG', 2)
    x = SpatialCoordinate(mesh)
    t = Constant(0)
    expr = sin(np.pi * (x[0] + 2 * x[1] + t))
    q = assemble(interpolate(expr, Q))

    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(q, num_sample_points=10, vmin=0.0, vmax=1.0, axes=axes)

    fn_plotter = FunctionPlotter(mesh, num_sample_points=10)

    def animate(time):
        t.assign(time)
        q.interpolate(expr)
        colors.set_array(fn_plotter(q))

    duration = 6
    fps = 24
    frames = np.linspace(0.0, duration, duration * fps)
    interval = 1e3 / fps
    movie = FuncAnimation(fig, animate, frames=frames, interval=interval)
    assert movie is not None

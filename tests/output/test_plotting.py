import pytest
from firedrake import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_plotting_1d():
    mesh = UnitIntervalMesh(32)
    CG = FunctionSpace(mesh, "CG", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    x, = SpatialCoordinate(mesh)

    u = interpolate(x * (1 - x), CG)
    v = project(x * (1 - x), DG)

    patches = plot(u, edgecolor='black', linestyle='--')
    assert patches is not None

    fig, axes = plt.subplots()
    plot(v, edgecolor='tab:green', label='v', axes=axes)
    legend = axes.legend(loc='upper right')
    assert len(legend.get_texts()) == 1


def test_plot_wrong_inputs():
    mesh = UnitSquareMesh(32, 32)
    with pytest.raises(TypeError):
        plot(mesh)

    Q = FunctionSpace(mesh, family='CG', degree=1)
    x, y = SpatialCoordinate(mesh)
    q = interpolate(x - y, Q)
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

    collection = tripcolor(f, axes=axes[2])
    assert collection is not None
    fig.colorbar(collection, ax=axes[2])


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

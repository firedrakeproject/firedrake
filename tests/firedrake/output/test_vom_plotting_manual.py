from firedrake import *
from firedrake.pyplot import *
import numpy as np
import matplotlib.pyplot as plt


def test_vom_plotting_2d_manual_examples():
    # [test_vom_plotting_2d_manual_examples 1]
    mesh = UnitSquareMesh(10, 10)
    coords = np.random.rand(50, 2)
    vom = VertexOnlyMesh(mesh, coords)

    # Plot the vertex-only mesh embedded in a 2D parent mesh
    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    scatter(vom, axes=axes)
    axes.set_aspect("equal")
    axes.legend()
    fig.show()

    # Define a scalar field on the vertex-only mesh and use it to colour points
    V = FunctionSpace(vom, "DG", 0)
    f = Function(V)
    f.dat.data_wo[:] = vom.coordinates.dat.data_ro[:, 0] # colour by x-coordinate

    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    sc = scatter(f, axes=axes, cmap="plasma")
    fig.colorbar(sc, ax=axes, label="x-coordinate")
    axes.set_aspect("equal")
    fig.show()
    # [test_vom_plotting_2d_manual_examples 2]

    # [test_vom_plotting_2d_manual_examples 3]
    # Define a Function on the parent mesh and plot the vertex-only mesh on top
    V_mesh = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u = Function(V_mesh)
    u.interpolate(sin(pi*x)*sin(pi*y))

    fig, axes = plt.subplots()
    tripcolor(u, axes=axes, cmap="viridis")
    scatter(vom, axes=axes, c="white", edgecolors="black", s=20)
    axes.set_aspect("equal")
    fig.show()
    # [test_vom_plotting_2d_manual_examples 4]

    # [test_vom_plotting_2d_manual_examples 5]
    # Define a vector field on the vertex-only mesh and plot its magnitude and direction at every point
    V_vec = VectorFunctionSpace(vom, "DG", 0, dim=2)
    v = Function(V_vec)
    v.dat.data_wo[:] = vom.coordinates.dat.data_ro - 0.5 # point away from the centre at (0.5, 0.5)

    fig, axes = plt.subplots()
    triplot(mesh, axes=axes)
    scatter(vom, axes=axes)
    quiver(v, axes=axes)
    fig.show()
    # [test_vom_plotting_2d_manual_examples 6]


def test_vom_plotting_3d_manual_examples():


    # [test_vom_plotting_3d_manual_examples 1]
    # Plot the vertex-only mesh embedded in a 2D parent mesh
    mesh_3d = UnitCubeMesh(5, 5, 5)
    coords_3d = np.random.rand(30, 3)
    vom_3d = VertexOnlyMesh(mesh_3d, coords_3d)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    # Reduce the opacity of the mesh interior facets
    triplot(mesh_3d, axes=axes, interior_kw={'alpha': 0.05}, boundary_kw={'alpha': 0.1})
    
    # Increase point size and disable depthshade so that all points are equally visible
    scatter(vom_3d, axes=axes, s=40, depthshade=False)
    axes.set_aspect("equal")
    fig.show()
    # [test_vom_plotting_3d_manual_examples 2]
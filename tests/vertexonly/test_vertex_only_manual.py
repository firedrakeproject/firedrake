from firedrake import *
import pytest
# Ensure that the code shown in the manual runs without error.


def test_vertex_only_mesh_manual_example():
    parent_mesh = UnitSquareMesh(10, 10)

    V = FunctionSpace(parent_mesh, "CG", 2)

    # Create a function f on the parent mesh to point evaluate
    x, y = SpatialCoordinate(parent_mesh)
    f = Function(V).interpolate(x**2 + y**2)

    # 3 points (i.e. vertices) at which to point evaluate f
    points = [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]

    vom = VertexOnlyMesh(parent_mesh, points)

    # P0DG is the only function space you can make on a vertex-only mesh
    P0DG = FunctionSpace(vom, "DG", 0)

    # Interpolation performs point evaluation
    f_at_points = interpolate(f, P0DG)

    print(f_at_points.dat.data)


def test_vom_manual_points_outside_domain():
    for i in [0]:
        parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)

        # point (1.1, 1.0) is outside the mesh
        points = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.0]]

    with pytest.raises(ValueError):
        # This will raise a ValueError
        vom = VertexOnlyMesh(parent_mesh, points, missing_points_behaviour='error')

    # This will generate a warning and the point will be lost
    vom = VertexOnlyMesh(parent_mesh, points, missing_points_behaviour='warn')

    # This will cause the point to be silently lost
    vom = VertexOnlyMesh(parent_mesh, points, missing_points_behaviour=None)

    assert vom  # Just here to shut up flake8 unused variable warning.


def test_vom_manual_keyword_arguments():
    omega = UnitSquareMesh(100, 100, quadrilateral=True)

    V = FunctionSpace(omega, "CG", 2)
    f = Function(V)

    points = [[0.1, 0.1], [0.2, 0.2], [1.0, 1.0]]
    # Create a vertex-only mesh at the points
    vom = VertexOnlyMesh(omega, points)

    # Create a P0DG function space on the vertex-only mesh
    P0DG = FunctionSpace(vom, "DG", 0)

    # Interpolating f into the P0DG space on the vertex-only mesh evaluates f at
    # the points
    expr = assemble(interpolate(f, P0DG)*dx)

    assert expr == 0.0


def test_mesh_tolerance():
    parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)

    # point (1.1, 1.0) is outside the mesh
    points = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.0]]

    # This prints 1.0 - points can be up to around 1 mesh cell width away from
    # the edge of the mesh and still be considered inside the domain.
    print(parent_mesh.tolerance)

    # This changes the tolerance and will cause the spatial index of the mesh
    # to be rebuilt when first performing point evaluation which can take some
    # time
    parent_mesh.tolerance = 20.0

    # This will now include the point (1.1, 1.0) in the mesh since each mesh
    # cell is 1.0/100.0 wide.
    vom = VertexOnlyMesh(parent_mesh, points)

    # Similarly .at will not generate an error
    V = FunctionSpace(parent_mesh, 'CG', 2)
    Function(V).at((1.1, 1.0))

    assert vom


def test_mesh_tolerance_change():
    parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)
    points = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.0]]
    V = FunctionSpace(parent_mesh, 'CG', 2)

    # The point (1.1, 1.0) will still be included in the vertex-only mesh
    vom = VertexOnlyMesh(parent_mesh, points, tolerance=30.0)

    # The tolerance property has been changed - this will print 30.0
    print(parent_mesh.tolerance)

    # This doesn't generate an error
    Function(V).at((1.1, 1.0), tolerance=20.0)

    # The tolerance property has been changed again - this will print 20.0
    print(parent_mesh.tolerance)

    try:
        # This generates an error
        Function(V).at((1.1, 1.0), tolerance=1.0)
    except PointNotInDomainError:
        # But the tolerance property has still been changed - this will print 1.0
        print(parent_mesh.tolerance)

    assert vom

from firedrake import *
from firedrake.__future__ import *
import pytest


def test_vertex_only_mesh_manual_example():
    # [test_vertex_only_mesh_manual_example 1]
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
    # [test_vertex_only_mesh_manual_example 2]
    f_at_points = assemble(interpolate(f, P0DG))

    print(f_at_points.dat.data_ro)
    # [test_vertex_only_mesh_manual_example 3]

    # [test_vertex_only_mesh_manual_example 4]
    # Make a P0DG function on the input ordering vertex-only mesh again
    P0DG_input_ordering = FunctionSpace(vom.input_ordering, "DG", 0)
    f_at_input_points = Function(P0DG_input_ordering)

    # We interpolate the other way this time
    f_at_input_points.interpolate(f_at_points)

    print(f_at_input_points.dat.data_ro)  # will print the values at the input points
    # [test_vertex_only_mesh_manual_example 5]

    # [test_vertex_only_mesh_manual_example 6]
    import numpy as np
    f_at_input_points.dat.data_wo[:] = np.nan
    f_at_input_points.interpolate(f_at_points)

    print(f_at_input_points.dat.data_ro)  # any points not found will be NaN
    # [test_vertex_only_mesh_manual_example 7]


def test_vom_manual_points_outside_domain():
    for i in [0]:
        parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)

        # [test_vom_manual_points_outside_domain 1]
        # point (1.1, 1.0) is outside the mesh
        points = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.0]]
        # [test_vom_manual_points_outside_domain 2]

    vom = True  # avoid flake8 unused variable warning
    with pytest.raises(VertexOnlyMeshMissingPointsError):
        # [test_vom_manual_points_outside_domain 3]
        # This will raise a VertexOnlyMeshMissingPointsError
        vom = VertexOnlyMesh(parent_mesh, points, missing_points_behaviour='error')
        # [test_vom_manual_points_outside_domain 4]

    def display_correct_indent():
        # [test_vom_manual_points_outside_domain 5]
        # This will generate a warning and the point will be lost
        vom = VertexOnlyMesh(parent_mesh, points, missing_points_behaviour='warn')

        # This will cause the point to be silently lost
        vom = VertexOnlyMesh(parent_mesh, points, missing_points_behaviour=None)
        # [test_vom_manual_points_outside_domain 6]

        assert vom  # Just here to shut up flake8 unused variable warning.

    assert vom  # here too
    display_correct_indent()


def test_vom_manual_keyword_arguments():
    # [test_vom_manual_keyword_arguments 1]
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
    f_at_points = assemble(interpolate(f, P0DG))
    expr = assemble(f_at_points * dx)
    # [test_vom_manual_keyword_arguments 2]

    assert expr == 0.0


def test_mesh_tolerance():
    # [test_mesh_tolerance 1]
    parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)

    # point (1.1, 1.0) is outside the mesh
    points = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.0]]

    # This prints 0.5 - points can be up to around half a mesh cell width away from
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
    # [test_mesh_tolerance 2]

    assert vom


def test_mesh_tolerance_change():
    parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)
    points = [[0.1, 0.1], [0.2, 0.2], [1.1, 1.0]]
    V = FunctionSpace(parent_mesh, 'CG', 2)

    # [test_mesh_tolerance_change 1]
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
    # [test_mesh_tolerance_change 2]

    assert vom


@pytest.mark.parallel
def test_input_ordering_input():
    parent_mesh = UnitSquareMesh(100, 100, quadrilateral=True)
    if parent_mesh.comm.rank == 0:
        point_locations_from_elsewhere = [[0.1, 0.1], [0.2, 0.2], [1.0, 1.0]]
        point_data_values_from_elsewhere = [1.0, 2.0, 3.0]
    elif parent_mesh.comm.rank == 1:
        point_locations_from_elsewhere = [[0.3, 0.3], [0.4, 0.4], [0.9, 0.9]]
        point_data_values_from_elsewhere = [4.0, 5.0, 6.0]
    else:
        import numpy as np
        point_locations_from_elsewhere = np.array([]).reshape(0, 2)
        point_data_values_from_elsewhere = []

    # [test_input_ordering_input 1]
    # We have a set of points with corresponding data from elsewhere which vary
    # from rank to rank
    vom = VertexOnlyMesh(parent_mesh, point_locations_from_elsewhere, redundant=False)
    P0DG = FunctionSpace(vom, "DG", 0)

    # Create a P0DG function on the input ordering vertex-only mesh
    P0DG_input_ordering = FunctionSpace(vom.input_ordering, "DG", 0)
    point_data_input_ordering = Function(P0DG_input_ordering)

    # We can safely set the values of this function, knowing that the data will
    # be in the same order and on the same MPI rank as point_locations_from_elsewhere
    point_data_input_ordering.dat.data_wo[:] = point_data_values_from_elsewhere

    # Interpolate puts this data onto the original vertex-only mesh
    point_data = assemble(interpolate(point_data_input_ordering, P0DG))
    # [test_input_ordering_input 2]

    assert vom
    if point_data:  # in case point data is None for some reason - apparently it can be in complex mode without an error occuring?
        assert point_data  # avoid flake8 unused variable warning

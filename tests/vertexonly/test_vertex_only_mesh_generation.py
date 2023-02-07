from firedrake import *
import pytest
import numpy as np
from mpi4py import MPI


# Utility Functions

def cell_midpoints(m):
    """Get the coordinates of the midpoints of every cell in mesh `m`.

    :param m: The mesh to generate cell midpoints for.

    :returns: A tuple of numpy arrays `(midpoints, local_midpoints)` where
    `midpoints` are the midpoints for the entire mesh even if the mesh is
    distributed and `local_midpoints` are the midpoints of only the
    rank-local non-ghost cells."""
    if isinstance(m.topology, mesh.ExtrudedMeshTopology):
        raise NotImplementedError("Extruded meshes are not supported")
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(SpatialCoordinate(m))
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = m.cell_set.size
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    local_midpoints = f.dat.data_ro
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.ufl_cell().geometric_dimension()), dtype=local_midpoints.dtype)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints, local_midpoints


@pytest.fixture(params=["interval",
                        "square",
                        pytest.param("extruded", marks=pytest.mark.xfail(reason="extruded meshes not supported")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        pytest.param("periodicrectangle"),
                        pytest.param("shiftedmesh", marks=pytest.mark.skip(reason="meshes with modified coordinate fields are not supported"))])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(1, 1), 1)
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(1, 1)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


def pseudo_random_coords(size):
    """
    Get an array of pseudo random coordinates with coordinate elements
    between -0.5 and 1.5. The random numbers are consistent for any
    given `size` since `numpy.random.seed(0)` is called each time this
    is used.
    """
    np.random.seed(0)
    a, b = -0.5, 1.5
    return (b - a) * np.random.random_sample(size=size) + a


# Mesh Generation Tests

def verify_vertexonly_mesh(m, vm, inputvertexcoords):
    """
    Check that VertexOnlyMesh `vm` immersed in parent mesh `m` with
    creation coordinates `inputvertexcoords` behaves as expected.
    `inputvertexcoords` should be the same for all MPI ranks to avoid
    hanging.
    """
    gdim = m.geometric_dimension()
    # Correct dims
    assert vm.geometric_dimension() == gdim
    assert vm.topological_dimension() == 0
    # Can initialise
    vm.init()
    # Find in-bounds and non-halo-region input coordinates
    in_bounds = []
    _, owned, _ = m.cell_set.sizes
    for i in range(len(inputvertexcoords)):
        cell_num = m.locate_cell(inputvertexcoords[i])
        if cell_num is not None and cell_num < owned:
            in_bounds.append(i)
    # Correct coordinates (though not guaranteed to be in same order)
    np.allclose(np.sort(vm.coordinates.dat.data_ro), np.sort(inputvertexcoords[in_bounds]))
    # Correct parent topology
    assert vm._parent_mesh is m
    assert vm.topology._parent_mesh is m.topology
    # Correct generic cell properties
    assert vm.cell_closure.shape == (len(inputvertexcoords[in_bounds]), 1)
    with pytest.raises(AttributeError):
        vm.exterior_facets()
    with pytest.raises(AttributeError):
        vm.interior_facets()
    with pytest.raises(AttributeError):
        vm.cell_to_facets
    assert vm.num_cells() == len(inputvertexcoords[in_bounds]) == vm.cell_set.size
    assert vm.num_facets() == 0
    assert vm.num_faces() == vm.num_entities(2) == 0
    assert vm.num_edges() == vm.num_entities(1) == 0
    assert vm.num_vertices() == vm.num_entities(0) == vm.num_cells()
    # Correct parent cell numbers
    stored_vertex_coords = np.copy(vm.topology_dm.getField("DMSwarmPIC_coor")).reshape((vm.num_cells(), gdim))
    vm.topology_dm.restoreField("DMSwarmPIC_coor")
    stored_parent_cell_nums = np.copy(vm.topology_dm.getField("parentcellnum"))
    vm.topology_dm.restoreField("parentcellnum")
    assert len(stored_vertex_coords) == len(stored_parent_cell_nums)
    for i in range(len(stored_vertex_coords)):
        assert m.locate_cell(stored_vertex_coords[i]) == stored_parent_cell_nums[i]


def test_generate_cell_midpoints(parentmesh):
    """
    Generate cell midpoints for mesh parentmesh and check they lie in
    the correct cells
    """
    inputcoords, inputcoordslocal = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, inputcoords)
    # Midpoints located in correct cells of parent mesh
    V = VectorFunctionSpace(parentmesh, "DG", 0)
    f = Function(V).interpolate(SpatialCoordinate(parentmesh))
    # Check size of biggest len(vm.coordinates.dat.data_ro) so
    # locate_cell can be called on every processor
    max_len = MPI.COMM_WORLD.allreduce(len(vm.coordinates.dat.data_ro), op=MPI.MAX)
    out_of_mesh_point = np.full((1, parentmesh.geometric_dimension()), np.inf)
    for i in range(max_len):
        if i < len(vm.coordinates.dat.data_ro):
            cell_num = parentmesh.locate_cell(vm.coordinates.dat.data_ro[i])
        else:
            cell_num = parentmesh.locate_cell(out_of_mesh_point)  # should return None
        if cell_num is not None:
            assert (f.dat.data_ro[cell_num] == vm.coordinates.dat.data_ro[i]).all()


@pytest.mark.parallel
def test_generate_cell_midpoints_parallel(parentmesh):
    test_generate_cell_midpoints(parentmesh)


def test_generate_random(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    verify_vertexonly_mesh(parentmesh, vm, vertexcoords)


@pytest.mark.parallel
def test_generate_random_parallel(parentmesh, vertexcoords):
    test_generate_random(parentmesh, vertexcoords)


@pytest.mark.xfail(raises=NotImplementedError)
def test_extrude(parentmesh):
    inputcoords, inputcoordslocal = cell_midpoints(parentmesh)
    vm = VertexOnlyMesh(parentmesh, inputcoords)
    ExtrudedMesh(vm, 1)


def test_point_tolerance():
    """Test the tolerance parameter of VertexOnlyMesh."""
    m = UnitSquareMesh(1, 1)
    assert m.tolerance == 1e-14
    assert m.tolerance == m.topology.tolerance
    # Make the mesh non-axis-aligned.
    m.coordinates.dat.data[1, :] = [1.1, 1]
    coords = [[1.0501, 0.5]]
    vm = VertexOnlyMesh(m, coords, tolerance=0.1)
    assert vm.cell_set.size == 1
    # check that the tolerance is passed through to the parent mesh
    assert m.tolerance == 0.1
    assert m.topology.tolerance == 0.1
    vm = VertexOnlyMesh(m, coords, tolerance=0.0)
    assert vm.cell_set.size == 0
    assert m.tolerance == 0.0
    assert m.topology.tolerance == 0.0
    # See if changing the tolerance on the parent mesh changes the tolerance
    # on the VertexOnlyMesh
    m.tolerance = 0.1
    vm = VertexOnlyMesh(m, coords)
    assert vm.cell_set.size == 1
    m.tolerance = 0.0
    vm = VertexOnlyMesh(m, coords)
    assert vm.cell_set.size == 0


def test_missing_points_behaviour(parentmesh):
    """
    Generate points outside of the parentmesh and check we get the expected
    error behaviour
    """
    inputcoord = np.full((1, parentmesh.geometric_dimension()), np.inf)
    assert len(inputcoord) == 1
    # No error by default
    vm = VertexOnlyMesh(parentmesh, inputcoord)
    assert vm.cell_set.size == 0
    with pytest.raises(ValueError):
        vm = VertexOnlyMesh(parentmesh, inputcoord, missing_points_behaviour='error')
    with pytest.warns(UserWarning):
        vm = VertexOnlyMesh(parentmesh, inputcoord, missing_points_behaviour='warn')


def test_outside_boundary_behaviour(parentmesh):
    """
    Generate points just outside the boundary of the parentmesh and
    check we get the expected behaviour. This is similar to the tolerance
    test but covers more meshes.
    """
    # This is just outside the boundary of the utility meshes in all cases
    inputcoord = np.full((1, parentmesh.geometric_dimension()), 0.0-1e-15)
    assert len(inputcoord) == 1
    # Tolerance is too small to pick up point
    vm = VertexOnlyMesh(parentmesh, inputcoord, tolerance=1e-16, missing_points_behaviour=None)
    assert vm.cell_set.size == 0
    # Tolerance is large enough to pick up point - note that we need to go up
    # by 2 orders of magnitude for this to work consistently
    vm = VertexOnlyMesh(parentmesh, inputcoord, tolerance=1e-13, missing_points_behaviour=None)
    assert vm.cell_set.size == 1


@pytest.mark.parallel(nprocs=2)  # nprocs == total number of mesh cells
def test_partition_behaviour_2d_2procs():
    test_partition_behaviour()


@pytest.mark.parallel(nprocs=3)  # nprocs > total number of mesh cells
def test_partition_behaviour_2d_3procs():
    test_partition_behaviour()


def test_partition_behaviour():
    parentmesh = UnitSquareMesh(1, 1)
    inputcoords = [[0.0-1e-15, 0.5],
                   [0.5, 0.0-1e-15],
                   [0.5, 1.0+1e-15],
                   [1.0+1e-15, 0.5],
                   [0.5, 0.5],
                   [0.5, 0.5],
                   [0.5+1e-15, 0.5],
                   [0.5, 0.5+1e-15]]
    npts = len(inputcoords)
    # Check that we get all the points with a big enough tolerance
    vm = VertexOnlyMesh(parentmesh, inputcoords, tolerance=1e-13, missing_points_behaviour='error')
    assert MPI.COMM_WORLD.allreduce(vm.cell_set.size, op=MPI.SUM) == npts
    # Check that we lose all but the last 4 points with a small tolerance
    with pytest.warns(UserWarning):
        vm = VertexOnlyMesh(parentmesh, inputcoords, tolerance=1e-16, missing_points_behaviour='warn')
    assert MPI.COMM_WORLD.allreduce(vm.cell_set.size, op=MPI.SUM) == 4


def test_inside_boundary_behaviour(parentmesh):
    """
    Generate points just inside the boundary of the parentmesh and
    check we get the expected behaviour. This is similar to the tolerance
    test but covers more meshes.
    """
    # This is just outside the boundary of the utility meshes in all cases
    inputcoord = np.full((1, parentmesh.geometric_dimension()), 0.0+1e-15)
    assert len(inputcoord) == 1
    # Tolerance is large enough to pick up point
    vm = VertexOnlyMesh(parentmesh, inputcoord, tolerance=1e-14, missing_points_behaviour=None)
    assert vm.cell_set.size == 1
    # Tolerance might be too small to pick up point, but it's not deterministic
    vm = VertexOnlyMesh(parentmesh, inputcoord, tolerance=1e-16, missing_points_behaviour=None)
    assert vm.cell_set.size == 0 or vm.cell_set.size == 1

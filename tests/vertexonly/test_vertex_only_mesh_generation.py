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
    m.init()
    V = VectorFunctionSpace(m, "DG", 0)
    f = Function(V).interpolate(SpatialCoordinate(m))
    # since mesh may be distributed, the number of cells on the MPI rank
    # may not be the same on all ranks (note we exclude ghost cells
    # hence using num_cells_local = m.cell_set.size). Below local means
    # MPI rank local.
    num_cells_local = len(f.dat.data_ro)
    num_cells = MPI.COMM_WORLD.allreduce(num_cells_local, op=MPI.SUM)
    # reshape is for 1D case where f.dat.data_ro has shape (num_cells_local,)
    local_midpoints = f.dat.data_ro.reshape(num_cells_local, m.ufl_cell().geometric_dimension())
    local_midpoints_size = np.array(local_midpoints.size)
    local_midpoints_sizes = np.empty(MPI.COMM_WORLD.size, dtype=int)
    MPI.COMM_WORLD.Allgatherv(local_midpoints_size, local_midpoints_sizes)
    midpoints = np.empty((num_cells, m.ufl_cell().geometric_dimension()), dtype=local_midpoints.dtype)
    MPI.COMM_WORLD.Allgatherv(local_midpoints, (midpoints, local_midpoints_sizes))
    assert len(np.unique(midpoints, axis=0)) == len(midpoints)
    return midpoints, local_midpoints


@pytest.fixture(params=["interval",
                        "square",
                        "extruded",
                        pytest.param("extrudedvariablelayers", marks=pytest.mark.skip(reason="Extruded meshes with variable layers not supported and will hang when created in parallel")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        "periodicrectangle",
                        "shiftedmesh"])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(2, 2), 3)
    elif request.param == "extrudedvariablelayers":
        return ExtrudedMesh(UnitIntervalMesh(3), np.array([[0, 3], [0, 3], [0, 2]]), np.array([3, 3, 2]))
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        return UnitIcosahedralSphereMesh()
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(10, 10)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=["redundant", "nonredundant"])
def redundant(request):
    if request.param == "redundant":
        return True
    else:
        return False


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

def verify_vertexonly_mesh(m, vm, inputvertexcoords, name):
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
    # has correct name
    assert vm.name == name
    # Find in-bounds and non-halo-region input coordinates
    in_bounds = []
    # this method of getting owned cells works for all mesh types
    owned_cells = len(Function(FunctionSpace(m, "DG", 0)).dat.data_ro)
    for i in range(len(inputvertexcoords)):
        cell_num = m.locate_cell(inputvertexcoords[i])
        if cell_num is not None and cell_num < owned_cells:
            in_bounds.append(i)
    # Correct coordinates (though not guaranteed to be in same order)
    np.allclose(np.sort(vm.coordinates.dat.data_ro), np.sort(inputvertexcoords[in_bounds]))
    # Correct parent topology
    assert vm._parent_mesh is m
    assert vm.topology._parent_mesh is m.topology
    # Correct generic cell properties
    if not skip_in_bounds_checks:
        assert vm.cell_closure.shape == (len(vm.coordinates.dat.data_ro_with_halos), 1)
    with pytest.raises(AttributeError):
        vm.exterior_facets()
    with pytest.raises(AttributeError):
        vm.interior_facets()
    with pytest.raises(AttributeError):
        vm.cell_to_facets
    if not skip_in_bounds_checks:
        assert vm.num_cells() == vm.cell_closure.shape[0] == len(vm.coordinates.dat.data_ro_with_halos) == vm.cell_set.total_size
        assert vm.cell_set.size == len(inputvertexcoords[in_bounds]) == len(vm.coordinates.dat.data_ro)
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
    # Input is correct (and includes points that were out of bounds)
    vm_input = vm.input_ordering
    assert vm_input.name == name + "_input_ordering"
    # We create vertex-only meshes using redundant=True by default so check
    # that vm_input has vertices on rank 0 only
    if MPI.COMM_WORLD.rank == 0:
        assert np.array_equal(vm_input.coordinates.dat.data_ro.reshape(inputvertexcoords.shape), inputvertexcoords)
    else:
        assert len(vm_input.coordinates.dat.data_ro) == 0


def test_generate_cell_midpoints(parentmesh, redundant):
    """
    Generate cell midpoints for mesh parentmesh and check they lie in
    the correct cells
    """
    inputcoords, inputcoordslocal = cell_midpoints(parentmesh)
    if redundant:
        # check redundant argument broadcasts from rank 0 by only supplying the
        # global cell midpoints only on rank 0. Note that this is the default
        # behaviour so it needn't be specified explicitly.
        if MPI.COMM_WORLD.rank == 0:
            vm = VertexOnlyMesh(parentmesh, inputcoords)
        else:
            # Otherwise we check redundant argument broadcasts from rank 0 by only
            # supplying the global cell midpoints only on rank 0. Note that this is
            # the default behaviour so it needn't be specified explicitly.
            if MPI.COMM_WORLD.rank == 0:
                vm = VertexOnlyMesh(parentmesh, inputcoords)
            else:
                vm = VertexOnlyMesh(parentmesh, np.empty(inputcoords.shape))
        # Check we can get original ordering back
        vm_input = vm.input_ordering
        if MPI.COMM_WORLD.rank == 0:
            assert np.array_equal(vm_input.coordinates.dat.data_ro.reshape(inputcoords.shape), inputcoords)
            vm_input.num_cells() == len(inputcoords)
        else:
            assert len(vm_input.coordinates.dat.data_ro) == 0
            vm_input.num_cells() == 0
    else:
        # When redundant == False we expect the same behaviour by only
        # supplying the local cell midpoints on each MPI ranks. Note that this
        # is not the default behaviour so it must be specified explicitly.
        vm = VertexOnlyMesh(parentmesh, inputcoordslocal, redundant=False)
        # Check we can get original ordering back
        vm_input = vm.input_ordering
        assert np.array_equal(vm_input.coordinates.dat.data_ro.reshape(inputcoordslocal.shape), inputcoordslocal)
        vm_input.num_cells() == len(inputcoordslocal)

    # Has correct name after not specifying one
    assert vm.name == parentmesh.name + "_immersed_vom"

    # More vm_input checks
    vm_input._parent_mesh is vm
    vm_input.input_ordering is None

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

    # Have correct pyop2 labels as implied by cell set sizes
    if parentmesh.extruded:
        layers = parentmesh.layers
        if parentmesh.variable_layers:
            # I think the below is correct but it's not actually tested...
            expected = tuple(size*(layer-1) for size, layer in zip(parentmesh.cell_set.sizes, layers))
            assert vm.cell_set.sizes == expected
        else:
            assert vm.cell_set.sizes == tuple(size*(layers-1) for size in parentmesh.cell_set.sizes)
    else:
        assert vm.cell_set.sizes == parentmesh.cell_set.sizes


@pytest.mark.parallel
def test_generate_cell_midpoints_parallel(parentmesh, redundant):
    test_generate_cell_midpoints(parentmesh, redundant)


def test_generate_random(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(
        parentmesh, vertexcoords, missing_points_behaviour=None, name="testvom"
    )
    verify_vertexonly_mesh(parentmesh, vm, vertexcoords, name="testvom")


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
    assert m.tolerance == 1.0
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
    # This is just outside the boundary of the utility meshes in all supported
    # cases
    edge_point = parentmesh.coordinates.dat.data_ro.min(axis=0, initial=np.inf)
    inputcoord = np.full((1, parentmesh.geometric_dimension()), edge_point-1e-15)
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
    # This is just inside the boundary of the utility meshes in all supported
    # cases
    edge_point = parentmesh.coordinates.dat.data_ro.min(axis=0, initial=np.inf)
    inputcoord = np.full((1, parentmesh.geometric_dimension()), edge_point+1e-15)
    assert len(inputcoord) == 1
    # Tolerance is large enough to pick up point
    vm = VertexOnlyMesh(parentmesh, inputcoord, tolerance=1e-14, missing_points_behaviour=None)
    assert vm.cell_set.size == 1
    # Tolerance might be too small to pick up point, but it's not deterministic
    vm = VertexOnlyMesh(parentmesh, inputcoord, tolerance=1e-16, missing_points_behaviour=None)
    assert vm.cell_set.size == 0 or vm.cell_set.size == 1


@pytest.mark.parallel(nprocs=2)
def test_pyop2_labelling():
    m = UnitIntervalMesh(4)
    # We inherit pyop2 labelling (owned, core and ghost) from the parent mesh
    # cell. Here we have one point per cell so can check directly
    points = np.asarray([[0.125], [0.375], [0.625], [0.875]])
    vm = VertexOnlyMesh(m, points, redundant=True)
    assert vm.cell_set.sizes == m.cell_set.sizes
    assert vm.cell_set.total_size == m.cell_set.total_size
    points = np.asarray([[0.125], [0.125], [0.375], [0.375], [0.625], [0.625], [0.875], [0.875]])
    vm = VertexOnlyMesh(m, points, redundant=True)
    assert vm.cell_set.total_size == 2*m.cell_set.total_size
    points = np.asarray([[-5.0]])
    vm = VertexOnlyMesh(m, points, redundant=False, missing_points_behaviour=None)
    assert vm.cell_set.total_size == 0

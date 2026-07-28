from firedrake import *
from firedrake.vertexonly_mutation import VertexOnlyMeshMutator
import pytest
from pytest_mpi.parallel_assert import parallel_assert
import numpy as np
from mpi4py import MPI


# Utility Functions

@pytest.fixture(params=["interval",
                        "square",
                        "squarequads",
                        "extruded",
                        pytest.param("extrudedvariablelayers", marks=pytest.mark.skip(reason="Extruded meshes with variable layers not supported and will hang when created in parallel")),
                        "cube",
                        "tetrahedron",
                        "immersedsphere",
                        "immersedsphereextruded",
                        "periodicrectangle",
                        "shiftedmesh"])
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "squarequads":
        return UnitSquareMesh(2, 2, quadrilateral=True)
    elif request.param == "extruded":
        return ExtrudedMesh(UnitSquareMesh(2, 2), 3)
    elif request.param == "extrudedvariablelayers":
        return ExtrudedMesh(UnitIntervalMesh(3), np.array([[0, 3], [0, 3], [0, 2]]), np.array([3, 3, 2]))
    elif request.param == "cube":
        return UnitCubeMesh(1, 1, 1)
    elif request.param == "tetrahedron":
        return UnitTetrahedronMesh()
    elif request.param == "immersedsphere":
        m = UnitIcosahedralSphereMesh(refinement_level=2, name='immersedsphere')
        m.init_cell_orientations(SpatialCoordinate(m))
        return m
    elif request.param == "immersedsphereextruded":
        m = UnitIcosahedralSphereMesh(refinement_level=2, name='immersedsphere')
        m.init_cell_orientations(SpatialCoordinate(m))
        return m
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(10, 10)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension)
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


# Function Space Generation Tests

def functionspace_tests(vm):
    # Prep
    num_cells = len(vm.coordinates.dat.data_ro)
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM)
    num_cells_halo = len(vm.coordinates.dat.data_ro_with_halos) - num_cells
    # Can create DG0 function space
    V = FunctionSpace(vm, "DG", 0)
    # Can't create with degree > 0
    with pytest.raises(ValueError):
        V = FunctionSpace(vm, "DG", 1)
    # Can create function on function spaces
    f = Function(V)
    g = Function(V)
    # Make expr which is x in 1D, x*y in 2D, x*y*z in 3D
    from functools import reduce
    from operator import mul
    expr = reduce(mul, SpatialCoordinate(vm))
    # Can interpolate and Galerkin project expressions onto functions
    f.interpolate(expr)
    g.project(expr)
    # Should have 1 DOF per cell so check DOF DataSet
    assert f.dof_dset.size == g.dof_dset.size == vm.cell_set.size == num_cells
    assert f.dof_dset.total_size == g.dof_dset.total_size == vm.cell_set.total_size == num_cells + num_cells_halo
    # The function should take on the value of the expression applied to
    # the vertex only mesh coordinates (with no change to coordinate ordering)
    # Reshaping because for all meshes, we want (-1, gdim) but
    # when gdim == 1 PyOP2 doesn't distinguish between dats with shape
    # () and shape (1,).
    assert np.allclose(f.dat.data_ro, np.prod(vm.coordinates.dat.data_ro.reshape(-1, vm.geometric_dimension), axis=1))
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times
    f.interpolate(Constant(2))
    assert np.isclose(assemble(f*dx), 2*num_cells_mpi_global)
    if "input_ordering" in vm.name:
        assert vm.input_ordering is None
        return
    # Can interpolate onto the input ordering VOM and we retain values from the
    # expresson on the main VOM
    W = FunctionSpace(vm.input_ordering, "DG", 0)
    h = Function(W)
    h.dat.data_wo_with_halos[:] = -1
    h.interpolate(g)
    # Exclude points which we know are missing - these should all be equal to -1
    input_ordering_parent_cell_nums = vm.input_ordering.topology_dm.getField("parentcellnum").ravel()
    vm.input_ordering.topology_dm.restoreField("parentcellnum")
    idxs_to_include = input_ordering_parent_cell_nums != -1
    assert np.allclose(h.dat.data_ro_with_halos[idxs_to_include], np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension), axis=1))
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == -1)
    # Using permutation matrix
    perm_mat = assemble(interpolate(TrialFunction(V), W), mat_type="aij")
    h2 = assemble(perm_mat @ g)
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], h.dat.data_ro_with_halos[idxs_to_include])
    h2 = assemble(interpolate(g, W))
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], h.dat.data_ro_with_halos[idxs_to_include])
    # check we can interpolate expressions
    h2 = Function(W)
    h2.interpolate(2*g)
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], 2*np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension), axis=1))
    # Check that the opposite works
    g.dat.data_wo_with_halos[:] = -1
    g.interpolate(h)
    assert np.allclose(g.dat.data_ro_with_halos, np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension), axis=1))

    h = assemble(interpolate(g, W))
    assert np.allclose(h.dat.data_ro_with_halos[idxs_to_include], np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension), axis=1))
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)
    h2 = assemble(interpolate(2*g, W))
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], 2*np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension), axis=1))

    h_star = h.riesz_representation(riesz_map="l2")
    g = assemble(interpolate(TestFunction(V), h_star))
    assert np.allclose(g.dat.data_ro_with_halos, np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension), axis=1))

    g2 = assemble(interpolate(2 * TestFunction(V), h_star))
    assert np.allclose(g2.dat.data_ro_with_halos, 2*np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension), axis=1))

    h_star = assemble(interpolate(TestFunction(W), g))
    h = h_star.riesz_representation(riesz_map="l2")
    assert np.allclose(h.dat.data_ro_with_halos[idxs_to_include], np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension), axis=1))
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)

    h2 = assemble(interpolate(2 * TestFunction(W), g))
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], 2*np.prod(vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include].reshape(-1, vm.input_ordering.geometric_dimension), axis=1))

    g = assemble(interpolate(h, V))
    assert np.allclose(g.dat.data_ro_with_halos, np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension), axis=1))
    g2 = assemble(interpolate(2 * h, V))
    assert np.allclose(g2.dat.data_ro_with_halos, 2*np.prod(vm.coordinates.dat.data_ro_with_halos.reshape(-1, vm.geometric_dimension), axis=1))


def vectorfunctionspace_tests(vm):
    # Prep
    gdim = vm.geometric_dimension
    num_cells = len(vm.coordinates.dat.data_ro)
    num_cells_mpi_global = MPI.COMM_WORLD.allreduce(num_cells, op=MPI.SUM)
    num_cells_halo = len(vm.coordinates.dat.data_ro_with_halos) - num_cells
    # Can create DG0 function space
    V = VectorFunctionSpace(vm, "DG", 0)
    # Can't create with degree > 0
    with pytest.raises(ValueError):
        V = VectorFunctionSpace(vm, "DG", 1)
    # Can create functions on function spaces
    f = Function(V)
    g = Function(V)
    # Can interpolate and Galerkin project onto functions
    x = SpatialCoordinate(vm)
    f.interpolate(2*x)
    g.project(2*x)
    # Should have 1 DOF per cell so check DOF DataSet
    assert f.dof_dset.size == g.dof_dset.size == vm.cell_set.size == num_cells
    assert f.dof_dset.total_size == g.dof_dset.total_size == vm.cell_set.total_size == num_cells + num_cells_halo
    # The function should take on the value of the expression applied to
    # the vertex only mesh coordinates (with no change to coordinate ordering)
    assert np.allclose(f.dat.data_ro, 2*vm.coordinates.dat.data_ro)
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)
    # Assembly works as expected - global assembly (integration) of a
    # constant on a vertex only mesh is evaluation of that constant
    # num_vertices (globally) times. Note that we get a vertex cell for
    # each geometric dimension so we have to sum over geometric
    # dimension too.
    R = VectorFunctionSpace(vm, "R", 0, dim=gdim)
    ones = Function(R).assign(1)
    f.interpolate(ones)
    assert np.isclose(assemble(inner(f, f)*dx), num_cells_mpi_global*gdim)
    if "input_ordering" in vm.name:
        assert vm.input_ordering is None
        return
    # Can interpolate onto the input ordering VOM and we retain values from the
    # expresson on the main VOM
    W = VectorFunctionSpace(vm.input_ordering, "DG", 0)
    h = Function(W)
    h.dat.data_wo_with_halos[:] = -1
    h.interpolate(g)
    # Exclude points which we know are missing - these should all be equal to -1
    input_ordering_parent_cell_nums = vm.input_ordering.topology_dm.getField("parentcellnum").ravel()
    vm.input_ordering.topology_dm.restoreField("parentcellnum")
    idxs_to_include = input_ordering_parent_cell_nums != -1
    assert np.allclose(h.dat.data_ro[idxs_to_include], 2*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == -1)
    # Using permutation matrix
    perm_mat = assemble(interpolate(TrialFunction(V), W), mat_type="aij")
    h2 = assemble(perm_mat @ g)
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], h.dat.data_ro_with_halos[idxs_to_include])
    # check other interpolation APIs work identically
    h2 = assemble(interpolate(g, W))
    assert np.allclose(h2.dat.data_ro_with_halos[idxs_to_include], h.dat.data_ro_with_halos[idxs_to_include])
    # check we can interpolate expressions
    h2 = Function(W)
    h2.interpolate(2*g)
    assert np.allclose(h2.dat.data_ro[idxs_to_include], 4*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    # Check that the opposite works
    g.dat.data_wo_with_halos[:] = -1
    g.interpolate(h)
    assert np.allclose(g.dat.data_ro_with_halos, 2*vm.coordinates.dat.data_ro_with_halos)

    h = assemble(interpolate(g, W))
    assert np.allclose(h.dat.data_ro[idxs_to_include], 2*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    assert np.all(h.dat.data_ro_with_halos[~idxs_to_include] == 0)
    h2 = assemble(interpolate(2*g, W))
    assert np.allclose(h2.dat.data_ro[idxs_to_include], 4*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])

    h_star = h.riesz_representation(riesz_map="l2")
    g = assemble(interpolate(TestFunction(V), h_star))
    assert np.allclose(g.dat.data_ro_with_halos, 2*vm.coordinates.dat.data_ro_with_halos)

    g2 = assemble(interpolate(2 * TestFunction(V), h_star))
    assert np.allclose(g2.dat.data_ro_with_halos, 4*vm.coordinates.dat.data_ro_with_halos)

    h_star = assemble(interpolate(TestFunction(W), g))
    assert np.allclose(h_star.dat.data_ro[idxs_to_include], 2*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])
    assert np.all(h_star.dat.data_ro_with_halos[~idxs_to_include] == 0)

    h2 = assemble(interpolate(2 * TestFunction(W), g))
    assert np.allclose(h2.dat.data_ro[idxs_to_include], 4*vm.input_ordering.coordinates.dat.data_ro_with_halos[idxs_to_include])

    h = h_star.riesz_representation(riesz_map="l2")
    g = assemble(interpolate(h, V))
    assert np.allclose(g.dat.data_ro_with_halos, 2*vm.coordinates.dat.data_ro_with_halos)
    g2 = assemble(interpolate(2*h, V))
    assert np.allclose(g2.dat.data_ro_with_halos, 4*vm.coordinates.dat.data_ro_with_halos)


@pytest.mark.parallel([1, 3])
def test_functionspaces(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour="ignore")
    functionspace_tests(vm)
    vectorfunctionspace_tests(vm)
    functionspace_tests(vm.input_ordering)
    vectorfunctionspace_tests(vm.input_ordering)


@pytest.mark.parallel(nprocs=2)
def test_simple_line():
    m = UnitIntervalMesh(4)
    points = np.asarray([[0.125], [0.375], [0.625]])
    vm = VertexOnlyMesh(m, points, redundant=True)
    V = FunctionSpace(vm, "DG", 0)
    f = Function(V)
    g = Function(V)
    x = SpatialCoordinate(vm)
    expr = x**2
    # Can interpolate and Galerkin project expressions onto functions
    f.interpolate(expr)
    g.project(expr)

    assert np.allclose(f.dat.data_ro, vm.coordinates.dat.data_ro**2)
    # Galerkin Projection of expression is the same as interpolation of
    # that expression since both exactly point evaluate the expression.
    assert np.allclose(f.dat.data_ro, g.dat.data_ro)


@pytest.mark.parallel(nprocs=2)
def test_input_ordering_missing_point():
    m = UnitIntervalMesh(4)
    points = np.asarray([[0.125], [0.375], [0.625], [5.0]])
    data = np.asarray([1.0, 2.0, 3.0, 4.0])
    vm = VertexOnlyMesh(m, points, missing_points_behaviour="ignore", redundant=True)

    # put data on the input ordering
    P0DG_input_ordering = FunctionSpace(vm.input_ordering, "DG", 0)
    data_input_ordering = Function(P0DG_input_ordering)

    if vm.comm.rank == 0:
        data_input_ordering.dat.data_wo[:] = data
        # Accessing data_ro [*here] is collective, hence this redundant call
        _ = len(data_input_ordering.dat.data_ro)
    else:
        data_input_ordering.dat.data_wo[:] = []
        # [*here]
        assert not len(data_input_ordering.dat.data_ro)

    # shouldn't have any halos
    assert np.array_equal(data_input_ordering.dat.data_ro_with_halos, data_input_ordering.dat.data_ro)

    # Interpolate it onto the immersed vertex-only mesh
    P0DG = FunctionSpace(vm, "DG", 0)
    data_on_vm = Function(P0DG).interpolate(data_input_ordering)

    # Check that the data is correct
    for data_at_point, point in zip(data_on_vm.dat.data_ro_with_halos, vm.coordinates.dat.data_ro_with_halos):
        assert data_at_point == data[points.flatten() == point]

    # change the data on the immersed vertex-only mesh
    data_on_vm.assign(2*data_on_vm)

    # interpolate it back onto the input ordering and make sure we get what we
    # expect and that the point which was missing still has it's original value
    data_input_ordering.interpolate(data_on_vm)
    if vm.comm.rank == 0:
        assert np.allclose(data_input_ordering.dat.data_ro[0:3], 2*data[0:3])
        # [*here]
        assert np.allclose(data_input_ordering.dat.data_ro[3], data[3])
    else:
        assert not len(data_input_ordering.dat.data_ro)
        # Accessing data_ro [*here] is collective, hence this redundant call
        _ = len(data_input_ordering.dat.data_ro)


@pytest.fixture(
    params=[
        ((2, 2), None),
        (None, True),
        ((), None),
        ((2, 3), None),
    ]
)
def tensorfs_and_expr(request):
    shape, symmetry = request.param
    np.random.seed(0)
    mesh = UnitSquareMesh(2, 2)
    coords = np.random.random_sample(size=(10, 2))
    vom = VertexOnlyMesh(mesh, coords)

    V = TensorFunctionSpace(vom, "DG", 0, shape=shape, symmetry=symmetry)
    W = TensorFunctionSpace(vom.input_ordering, "DG", 0, shape=shape, symmetry=symmetry)

    x = SpatialCoordinate(vom)
    if shape == ():
        expr = inner(x, x)
    elif shape is None or shape == (2, 2):
        expr = outer(x, x) + Identity(2)
    elif shape == (2, 3):
        a = as_vector([x[0], x[1]])
        b = as_vector([x[0], x[1], Constant(1.0)])
        expr = outer(a, b)

    return V, W, expr


@pytest.mark.parallel([1, 3])
def test_tensorfs_permutation(tensorfs_and_expr):
    V, W, expr = tensorfs_and_expr
    f = Function(V)
    f.interpolate(expr)
    f_in_W = assemble(interpolate(f, W))
    python_mat = assemble(interpolate(TrialFunction(V), W), mat_type="matfree")
    f_in_W_2 = assemble(python_mat @ f)
    assert np.allclose(f_in_W.dat.data_ro, f_in_W_2.dat.data_ro)
    petsc_mat = assemble(interpolate(TrialFunction(V), W), mat_type="aij")
    f_in_W_petsc = assemble(petsc_mat @ f)
    assert np.allclose(f_in_W.dat.data_ro, f_in_W_petsc.dat.data_ro)


# Function Space Mutation Tests

@pytest.fixture
def parent_mesh():
    return UnitSquareMesh(5, 5, quadrilateral=False)

@pytest.fixture
def vom(parent_mesh):
    points = cell_midpoints(parent_mesh, with_halos=False)
    return VertexOnlyMesh(parent_mesh, points, redundant=False)

@pytest.fixture(params=["scalar", "vector", "tensor"])
def vom_fs(request, vom):
    if request.param == "scalar":
        return FunctionSpace(vom, "DG", 0)

    if request.param == "vector":
        return VectorFunctionSpace(vom, "DG", 0, dim=vom.geometric_dimension)

    if request.param == "tensor":
        return TensorFunctionSpace(vom, "DG", 0, shape=(2, 2))

def cell_midpoints(mesh, with_halos=False):
    """
    Create deterministic physical point locations-one point at the midpoint of each mesh cell-returned in the cell order.
    Setting `with_halos=False` ensures we only get midpoints that are owned by each rank."""
    V = VectorFunctionSpace(mesh, "DG", 0)
    x = Function(V).interpolate(SpatialCoordinate(mesh))

    data = x.dat.data_ro_with_halos if with_halos else x.dat.data_ro
    cell_nodes = V.cell_node_list[:, 0]

    if not with_halos:
        cell_nodes = cell_nodes[:mesh.cell_set.size]

    return data[cell_nodes].copy()

def locate_points(mesh, points):
    """Get local data (reference coordinates + parent cells) from physical point coordinates."""
    parent_cells, refcoords, _ = mesh.locate_cells_ref_coords_and_dists(points)
    return np.asarray(parent_cells, dtype=int), np.asarray(refcoords, dtype=float)

@pytest.mark.parallel([1, 3])
def test_vom_fs_data_refreshes_after_vom_mutation(vom, vom_fs):
    """Check that FunctionSpace caches get invalidated and recomputed after a VOM topology change."""
    mutator = VertexOnlyMeshMutator(vom)
    
    T = vom_fs.topological
    
    # Read topology-dependent properties from the caches
    old_shared_data = T._shared_data
    old_dof_dset = T.dof_dset
    old_global_numbering = T.global_numbering
    old_dm = T.dm
    old_ises = T._ises
    old_cell_node_list = T.cell_node_list
    old_cell_node_map = T.cell_node_map()

    # Mutate the VOM topologically by removing every other point on each rank
    absorbed_local_indices = np.arange(0, vom.cell_set.size, 2, dtype=int)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_shared_data = T._shared_data
    new_dof_dset = T.dof_dset
    new_global_numbering = T.global_numbering
    new_dm = T.dm
    new_ises = T._ises
    new_cell_node_list = T.cell_node_list
    new_cell_node_map = T.cell_node_map()

    parallel_assert(new_shared_data is not old_shared_data)
    parallel_assert(new_dof_dset is not old_dof_dset)
    parallel_assert(new_global_numbering is not old_global_numbering)
    parallel_assert(new_dm is not old_dm)
    parallel_assert(new_ises is not old_ises)
    parallel_assert(new_cell_node_list is not old_cell_node_list)
    parallel_assert(new_cell_node_map is not old_cell_node_map)


@pytest.mark.parallel([1, 3])
def test_vom_fs_data_resizes_to_match_vom_topology(vom, vom_fs):
    T = vom_fs.topological
    mutator = VertexOnlyMeshMutator(vom)

    n_old_local_nodes = T.node_set.size # 1 node per VOM point
    n_old_global_dofs = T.dim() # total number of scalar DoFs across all ranks

    # Mutate the VOM topologically by removing every other point on each rank
    absorbed_local_indices = np.arange(0, vom.cell_set.size, 2, dtype=int)
    n_local_absorbed = len(absorbed_local_indices)
    n_global_absorbed = vom.comm.allreduce(n_local_absorbed, op=MPI.SUM)

    # Expected size post rebuild
    expected_n_local_nodes = n_old_local_nodes - n_local_absorbed
    expected_n_global_dofs = n_old_global_dofs - n_global_absorbed * T.block_size

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    new_n_global_dofs = T.dim()
    new_ises = T._ises

    # Check that the FS data structures are rebuilt appropriately to match the new VOM topology
    fs_layout_matches_vom_topology_post_rebuild = (
        vom.cell_set.size == expected_n_local_nodes
        and T.dof_dset.size == expected_n_local_nodes
        and T.node_set.size == expected_n_local_nodes
        and T.node_set.total_size == expected_n_local_nodes
        and T.node_count == expected_n_local_nodes
        and T.dof_count == expected_n_local_nodes * T.block_size
        and T.cell_node_list.shape == (vom.cell_set.size, 1)
        and T.global_numbering.getStorageSize() == expected_n_local_nodes
        and T.dm is T.dof_dset.dm
        and new_ises is T.dof_dset.field_ises
    )

    parallel_assert(
        fs_layout_matches_vom_topology_post_rebuild,
        "The FunctionSpace layout does not match the rebuilt VOM topology",
    )

    assert new_n_global_dofs == expected_n_global_dofs

@pytest.mark.parallel([1, 3])
def test_vom_fs_rebuilds_under_successive_vom_rebuilds(vom, vom_fs):
    T = vom_fs.topological
    mutator = VertexOnlyMeshMutator(vom)

    block_size = T.block_size # number of scalar DoFs per node
    v0 = vom.topology._topology_version

    shared_data_v0 = T._shared_data
    dof_dset_v0 = T.dof_dset
    dm_v0 = T.dm

    n_local_nodes_v0 = T.node_set.size
    n_global_dofs_v0 = T.dim()

    # First rebuild
    absorbed_v1 = np.arange(1, n_local_nodes_v0, 2, dtype=int)
    n_global_absorbed_v1 = vom.comm.allreduce(
        len(absorbed_v1),
        op=MPI.SUM,
    )

    expected_local_nodes_v1 = n_local_nodes_v0 - len(absorbed_v1)
    expected_global_dofs_v1 = n_global_dofs_v0 - n_global_absorbed_v1 * block_size

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_v1)

    # Access thee FS state after the first rebuild
    v1 = vom.topology._topology_version
    shared_data_v1 = T._shared_data
    dof_dset_v1 = T.dof_dset
    dm_v1 = T.dm
    n_global_dofs_v1 = T.dim()

    fs_layout_matches_vom_topology_post_rebuild_1 = (
        v1 == v0 + 1
        and shared_data_v1 is not shared_data_v0
        and dof_dset_v1 is not dof_dset_v0
        and dm_v1 is not dm_v0
        and vom.cell_set.size == expected_local_nodes_v1
        and T.node_set.size == expected_local_nodes_v1
        and T.node_set.total_size == expected_local_nodes_v1
        and T.node_count == expected_local_nodes_v1
        and T.dof_count == expected_local_nodes_v1 * block_size
        and T.cell_node_list.shape == (vom.cell_set.size, 1)
        and n_global_dofs_v1 == expected_global_dofs_v1
    )

    parallel_assert(
        fs_layout_matches_vom_topology_post_rebuild_1, 
        "The FunctionSpace was not rebuilt correctly after the first VOM mutation"
    )

    # Second rebuild
    absorbed_v2 = np.arange(1, vom.cell_set.size, 2, dtype=int)
    n_global_absorbed_v2 = vom.comm.allreduce(
        len(absorbed_v2),
        op=MPI.SUM,
    )

    expected_local_nodes_v2 = expected_local_nodes_v1 - len(absorbed_v2)
    expected_global_dofs_v2 = (
        expected_global_dofs_v1 - n_global_absorbed_v2 * block_size
    )

    mutator.rebuild_vom(absorbed_vom_indices=absorbed_v2)

    # Access the FS state after the second rebuild 
    v2 = vom.topology._topology_version
    shared_data_2 = T._shared_data
    dof_dset_2 = T.dof_dset
    dm_2 = T.dm
    n_global_dofs_2 = T.dim()

    fs_layout_matches_vom_topology_post_rebuild_2 = (
        v2 == v1 + 1
        and shared_data_2 is not shared_data_v1
        and dof_dset_2 is not dof_dset_v1
        and dm_2 is not dm_v1
        and vom.cell_set.size == expected_local_nodes_v2
        and T.node_set.size == expected_local_nodes_v2
        and T.node_set.total_size == expected_local_nodes_v2
        and T.node_count == expected_local_nodes_v2
        and T.dof_count == expected_local_nodes_v2 * block_size
        and T.cell_node_list.shape == (vom.cell_set.size, 1)
        and n_global_dofs_2 == expected_global_dofs_v2
    )

    parallel_assert(
        fs_layout_matches_vom_topology_post_rebuild_2,
        "The FunctionSpace was not rebuilt correctly after the second VOM mutation",
    )

@pytest.mark.parallel([1, 3])
def test_vom_fs_rebuils_to_matche_empty_vom(vom, vom_fs):
    T = vom_fs.topological

    # First ensure all topology-dependent structures exist and have been computed
    _ = T.dof_dset
    _ = T.global_numbering
    _ = T.dm
    _ = T._ises
    _ = T.cell_node_list
    _ = T.cell_node_map()

    # Mutate the VOM: remove all points from every rank
    mutator = VertexOnlyMeshMutator(vom)
    old_local_count = vom.cell_set.size
    absorbed_local_indices = np.arange(old_local_count, dtype=int)
    mutator.rebuild_vom(absorbed_vom_indices=absorbed_local_indices)

    # Access the new state post rebuild
    node_set = T.node_set
    dof_dset = T.dof_dset
    layout_vec = dof_dset.layout_vec
    global_numbering = T.global_numbering
    dm = T.dm
    ises = T._ises
    cell_node_list = T.cell_node_list
    cell_node_map = T.cell_node_map()
    global_dim = T.dim()

    # Check that all FS data structures are now of size 0
    fs_layout_is_empty = (
        vom.cell_set.size == 0
        and vom.cell_set.total_size == 0
        and node_set.size == 0
        and node_set.total_size == 0
        and dof_dset.size == 0
        and dof_dset.total_size == 0
        and T.node_count == 0
        and T.dof_count == 0
        and global_dim == 0
        and layout_vec.getLocalSize() == 0
        and layout_vec.getSize() == 0
        and global_numbering.getStorageSize() == 0
        and cell_node_list.shape == (0, 1)
        and cell_node_list.size == 0
        and cell_node_map.values.size == 0
        and cell_node_map.values_with_halo.size == 0
        and dm is dof_dset.dm
        and all(iset.getSize() == 0 for iset in ises)
    )

    parallel_assert(
        fs_layout_is_empty,
        "The FunctionSpace data structures are non-empty despite the VOM being empty"
    )

@pytest.mark.parallel(nprocs=3)
def test_vom_fs_rebuilds_under_parallel_migration(parent_mesh, vom, vom_fs):
    mutator = VertexOnlyMeshMutator(vom)
    T = vom_fs.topological

    old_global_dim = T.dim()
    old_local_nodes = T.node_set.size
    old_dof_dset = T.dof_dset
    old_ises = T._ises
    old_dm = T.dm

    # Migrate the first particle on rank 0 to a neighbouring rank
    midpoints = cell_midpoints(parent_mesh, with_halos=True)

    new_coords = vom.coordinates.dat.data_ro.copy()
    destination_rank = None

    ghost_cell_ids = np.arange(parent_mesh.cell_set.size, parent_mesh.cell_set.total_size, dtype=int)

    if parent_mesh.comm.rank == 0:
        # New point is the midpoint of the first ghost cell
        target_cell = ghost_cell_ids[0]
        new_coords[0] = midpoints[target_cell]

        _, sf_leaves, sf_remotes = parent_mesh.topology_dm.getPointSF().getGraph()

        leaf_owning_ranks = {
            leaf: remote[0]
            for leaf, remote in zip(sf_leaves, sf_remotes)
        }

        target_plex_cell = parent_mesh.topology.cell_closure[target_cell, -1] # Firedrake cell ID -> plex cell ID
        destination_rank = leaf_owning_ranks.get(target_plex_cell)

    destination_rank = parent_mesh.comm.bcast(destination_rank, root=0)
    
    parallel_assert(destination_rank is not None and destination_rank != 0, "Rank 0 failed to choose a destination rank to migrate its particle to")

    # Commit new state on every rank
    new_parent_cells, new_ref_coords = locate_points(parent_mesh, new_coords)
    vom.coordinates.dat.data_wo[:] = new_coords
    mutator.commit_reference_state(new_parent_cells, new_ref_coords)

    mutator.rebuild_vom()

    expected_local_nodes = old_local_nodes

    if parent_mesh.comm.rank == 0:
        expected_local_nodes -=1
    
    if parent_mesh.comm.rank == destination_rank:
        expected_local_nodes += 1

    # Check that the FS data structures are rebuilt appropriately to match the new VOM topology
    fs_layout_matches_migration = (
        vom.cell_set.size == expected_local_nodes
        and T.node_set.size == expected_local_nodes
        and T.node_set.total_size == expected_local_nodes # no ghost points by construction
        and T.dof_dset.size == expected_local_nodes
        and T.node_count == expected_local_nodes
        and T.dof_count == expected_local_nodes * T.block_size
        and T.cell_node_list.shape == (expected_local_nodes, 1)
        and T.global_numbering.getStorageSize() == expected_local_nodes
        and T.dof_dset.layout_vec.getLocalSize() == expected_local_nodes * T.block_size
        and T.dof_dset is not old_dof_dset
        and T.dm is not old_dm
        and T._ises is not old_ises
        and T._ises is T.dof_dset.field_ises
        and T.dim() == old_global_dim # no points absorbed
    )

    parallel_assert(fs_layout_matches_migration, "The FunctionSpace layout does not match the migrated VOM topology")

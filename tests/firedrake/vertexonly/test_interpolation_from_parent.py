from firedrake import *
from firedrake.__future__ import *
import pytest
import numpy as np
from functools import reduce
from operator import add
import subprocess


# Utility Functions and Fixtures

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
                        "shiftedmesh"],
                ids=lambda x: f"{x}-mesh")
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
        m = UnitIcosahedralSphereMesh()
        m.init_cell_orientations(SpatialCoordinate(m))
        m = ExtrudedMesh(m, 3, extrusion_type="radial")
        return m
    elif request.param == "periodicrectangle":
        return PeriodicRectangleMesh(3, 3, 1, 1)
    elif request.param == "shiftedmesh":
        m = UnitSquareMesh(10, 10)
        m.coordinates.dat.data[:] -= 0.5
        return m


@pytest.fixture(params=[0, 1, 100], ids=lambda x: f"{x}-coords")
def vertexcoords(request, parentmesh):
    size = (request.param, parentmesh.geometric_dimension())
    return pseudo_random_coords(size)


@pytest.fixture(params=[("CG", 2, FunctionSpace),
                        ("DG", 2, FunctionSpace)],
                ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})")
def fs(request):
    return request.param


@pytest.fixture(params=[("CG", 2, VectorFunctionSpace),
                        ("N1curl", 2, FunctionSpace),
                        ("N2curl", 2, FunctionSpace),
                        ("N1div", 2, FunctionSpace),
                        ("N2div", 2, FunctionSpace),
                        pytest.param(("RTCE", 2, FunctionSpace),
                                     marks=pytest.mark.xfail(raises=(subprocess.CalledProcessError, NotImplementedError),
                                                             reason="EnrichedElement dual basis not yet defined and FIAT duals don't have a point_dict")),
                        pytest.param(("RTCF", 2, FunctionSpace),
                                     marks=pytest.mark.xfail(raises=(subprocess.CalledProcessError, NotImplementedError),
                                                             reason="EnrichedElement dual basis not yet defined and FIAT duals don't have a point_dict"))],
                ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})")
def vfs(request, parentmesh):
    family = request.param[0]
    # skip where the element doesn't support the cell type
    if family != "CG":
        if parentmesh.ufl_cell().cellname() == "quadrilateral":
            if not (family == "RTCE" or family == "RTCF"):
                pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
        elif parentmesh.ufl_cell().cellname() == "triangle" or parentmesh.ufl_cell().cellname() == "tetrahedron":
            if (not (family == "N1curl" or family == "N2curl"
                     or family == "N1div" or family == "N2div")):
                pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
        else:
            pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
        if parentmesh.name == "immersedsphere":
            # See https://github.com/firedrakeproject/firedrake/issues/3089
            if family == "N1curl" or family == "N2curl":
                pytest.xfail(f"{family} does not give correct point evaluation results on immersed manifolds")
            elif family == "N1div" or family == "N2div":
                pytest.xfail(f"{family} cannot yet perform point evaluation on immersed manifolds")
    return request.param


@pytest.fixture(params=[("CG", 2, TensorFunctionSpace),
                        ("BDM", 2, VectorFunctionSpace),
                        ("Regge", 2, lambda *args: FunctionSpace(*args, variant="point"))],
                ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})")
def tfs(request, parentmesh):
    family = request.param[0]
    # skip where the element doesn't support the cell type
    if (family != "CG" and parentmesh.ufl_cell().cellname() != "triangle"
            and parentmesh.ufl_cell().cellname() != "tetrahedron"):
        pytest.skip(f"{family} does not support {parentmesh.ufl_cell()} cells")
    if parentmesh.name == "immersedsphere":
        # See https://github.com/firedrakeproject/firedrake/issues/3089
        if family == "Regge":
            pytest.xfail(f"{family} does not give correct point evaluation results on immersed manifolds")
        elif family == "BDM":
            pytest.xfail(f"{family} cannot yet perform point evaluation on immersed manifolds")
    return request.param


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


def allgather(comm, coords):
    """Gather all coordinates from all ranks."""
    coords = coords.copy()
    coords = comm.allgather(coords)
    coords = np.concatenate(coords)
    return coords


def immersed_sphere_vertexcoords(mesh, vertexcoords_old):
    # Need to pick points approximately on the surface of the sphere
    # to avoid interpolation errors in the tests. I just use the vertices of
    # the mesh itself. Correct projection behaviour (when the points are not
    # within cells) is tested elsewhere.
    if not len(vertexcoords_old):
        return vertexcoords_old
    else:
        # Get the coordinates of the vertices of the mesh
        meshvertexcoords = allgather(mesh.comm, mesh.coordinates.dat.data_ro)
        return meshvertexcoords[0:len(vertexcoords_old)]


# Tests

# NOTE: these _spatialcoordinate tests should be equivalent to some kind of
# interpolation from a CG1 VectorFunctionSpace (I think)
def test_scalar_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    # Reshaping because for all meshes, we want (-1, gdim) but
    # when gdim == 1 PyOP2 doesn't distinguish between dats with shape
    # () and shape (1,).
    vertexcoords = vm.coordinates.dat.data_ro.reshape(-1, parentmesh.geometric_dimension())
    W = FunctionSpace(vm, "DG", 0)
    expr = reduce(add, SpatialCoordinate(parentmesh))
    w_expr = assemble(interpolate(expr, W))
    assert np.allclose(w_expr.dat.data_ro, np.sum(vertexcoords, axis=1))


def test_scalar_function_interpolation(parentmesh, vertexcoords, fs):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    vertexcoords = vm.coordinates.dat.data_ro.reshape(-1, parentmesh.geometric_dimension())
    fs_fam, fs_deg, fs_typ = fs
    if (
        parentmesh.coordinates.function_space().ufl_element().family()
        == "Discontinuous Lagrange"
        and fs_fam == "CG"
    ):
        pytest.skip(f"Interpolating into f{fs_fam} on a periodic mesh is not well-defined.")

    V = fs_typ(parentmesh, fs_fam, fs_deg)
    W = FunctionSpace(vm, "DG", 0)
    expr = reduce(add, SpatialCoordinate(parentmesh))
    v = Function(V).interpolate(expr)
    w_v = assemble(interpolate(v, W))
    assert np.allclose(w_v.dat.data_ro, np.sum(vertexcoords, axis=1))
    # try and make reusable Interpolator from V to W
    A_w = Interpolator(TestFunction(V), W)
    w_v = Function(W)
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro, np.sum(vertexcoords, axis=1))
    # use it again for a different Function in V
    v = Function(V).assign(Constant(2))
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro, 2)


def test_vector_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    vertexcoords = vm.coordinates.dat.data_ro
    W = VectorFunctionSpace(vm, "DG", 0)
    expr = 2 * SpatialCoordinate(parentmesh)
    w_expr = assemble(interpolate(expr, W))
    assert np.allclose(w_expr.dat.data_ro, 2*np.asarray(vertexcoords))


def test_vector_function_interpolation(parentmesh, vertexcoords, vfs):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    vfs_fam, vfs_deg, vfs_typ = vfs
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    vertexcoords = vm.coordinates.dat.data_ro
    if (
        parentmesh.coordinates.function_space().ufl_element().family()
        == "Discontinuous Lagrange"
    ):
        pytest.skip(f"Interpolating into f{vfs_fam} on a periodic mesh is not well-defined.")
    V = vfs_typ(parentmesh, vfs_fam, vfs_deg)
    W = VectorFunctionSpace(vm, "DG", 0)
    expr = 2 * SpatialCoordinate(parentmesh)
    v = Function(V).interpolate(expr)
    w_v = assemble(interpolate(v, W))
    assert np.allclose(w_v.dat.data_ro, 2*np.asarray(vertexcoords))
    # try and make reusable Interpolator from V to W
    A_w = Interpolator(TestFunction(V), W)
    w_v = Function(W)
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro, 2*np.asarray(vertexcoords))
    # use it again for a different Function in V
    expr = 4 * SpatialCoordinate(parentmesh)
    v = Function(V).interpolate(expr)
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro, 4*np.asarray(vertexcoords))


def test_tensor_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    vertexcoords = vm.coordinates.dat.data_ro
    W = TensorFunctionSpace(vm, "DG", 0)
    x = SpatialCoordinate(parentmesh)
    gdim = parentmesh.geometric_dimension()
    expr = 2 * as_tensor([x]*gdim)
    assert W.shape == expr.ufl_shape
    w_expr = assemble(interpolate(expr, W))
    result = 2 * np.asarray([[vertexcoords[i]]*gdim for i in range(len(vertexcoords))])
    if len(result) == 0:
        result = result.reshape(vertexcoords.shape + (gdim,))
    assert np.allclose(w_expr.dat.data_ro.reshape(result.shape), result)


def test_tensor_function_interpolation(parentmesh, vertexcoords, tfs):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    tfs_fam, tfs_deg, tfs_typ = tfs
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    vertexcoords = vm.coordinates.dat.data_ro
    if (
        parentmesh.coordinates.function_space().ufl_element().family()
        == "Discontinuous Lagrange"
    ):
        pytest.skip(f"Interpolating into f{tfs_fam} on a periodic mesh is not well-defined.")
    V = tfs_typ(parentmesh, tfs_fam, tfs_deg)
    W = TensorFunctionSpace(vm, "DG", 0)
    x = SpatialCoordinate(parentmesh)
    # use outer product to check Regge works
    expr = outer(x, x)
    assert W.shape == expr.ufl_shape
    v = Function(V).interpolate(expr)
    result = np.asarray([np.outer(vertexcoords[i], vertexcoords[i]) for i in range(len(vertexcoords))])
    if len(result) == 0:
        result = result.reshape(vertexcoords.shape + (parentmesh.geometric_dimension(),))
    w_v = assemble(interpolate(v, W))
    assert np.allclose(w_v.dat.data_ro.reshape(result.shape), result)
    # try and make reusable Interpolator from V to W
    A_w = Interpolator(TestFunction(V), W)
    w_v = Function(W)
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro.reshape(result.shape), result)
    # use it again for a different Function in V
    expr = 2*outer(x, x)
    v = Function(V).interpolate(expr)
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro.reshape(result.shape), 2*result)


def test_mixed_function_interpolation(parentmesh, vertexcoords, tfs):
    if parentmesh.name == "immersedsphere":
        vertexcoords = immersed_sphere_vertexcoords(parentmesh, vertexcoords)
    tfs_fam, tfs_deg, tfs_typ = tfs

    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    vertexcoords = vm.coordinates.dat.data_ro.reshape(-1, parentmesh.geometric_dimension())
    if (
        parentmesh.coordinates.function_space().ufl_element().family()
        == "Discontinuous Lagrange"
    ):
        pytest.skip(f"Interpolating into f{tfs_fam} on a periodic mesh is not well-defined.")

    V1 = tfs_typ(parentmesh, tfs_fam, tfs_deg)
    V2 = FunctionSpace(parentmesh, "CG", 1)
    V = V1 * V2
    W1 = TensorFunctionSpace(vm, "DG", 0)
    W2 = FunctionSpace(vm, "DG", 0)
    W = W1 * W2
    x = SpatialCoordinate(parentmesh)
    v = Function(V)
    v1, v2 = v.subfunctions
    # Get Function in V1
    # use outer product to check Regge works
    expr1 = outer(x, x)
    assert W1.value_shape == expr1.ufl_shape
    v1.interpolate(expr1)
    result1 = np.asarray([np.outer(vertexcoords[i], vertexcoords[i]) for i in range(len(vertexcoords))])
    if len(result1) == 0:
        result1 = result1.reshape(vertexcoords.shape + (parentmesh.geometric_dimension(),))
    # Get Function in V2
    expr2 = reduce(add, SpatialCoordinate(parentmesh))
    v2.interpolate(expr2)
    result2 = np.sum(vertexcoords, axis=1)

    # Interpolate Function in V into W
    w_v = assemble(interpolate(v, W))

    # Split result and check
    w_v1, w_v2 = w_v.subfunctions
    assert np.allclose(w_v1.dat.data_ro.reshape(result1.shape), result1)
    assert np.allclose(w_v2.dat.data_ro.reshape(result2.shape), result2)


def test_scalar_real_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    W = FunctionSpace(vm, "DG", 0)
    V = FunctionSpace(parentmesh, "Real", 0)
    # Remove below when interpolating constant onto Real works for extruded
    if type(parentmesh.topology) is mesh.ExtrudedMeshTopology:
        with pytest.raises(ValueError):
            assemble(interpolate(Constant(1), V))
        return
    v = assemble(interpolate(Constant(1), V))
    w_v = assemble(interpolate(v, W))
    assert np.allclose(w_v.dat.data_ro, 1.)


def test_scalar_real_interpolator(parentmesh, vertexcoords):
    # try and make reusable Interpolator from V to W
    vm = VertexOnlyMesh(parentmesh, vertexcoords, missing_points_behaviour=None)
    W = FunctionSpace(vm, "DG", 0)
    V = FunctionSpace(parentmesh, "Real", 0)
    # Remove below when interpolating constant onto Real works for extruded
    if type(parentmesh.topology) is mesh.ExtrudedMeshTopology:
        with pytest.raises(ValueError):
            assemble(interpolate(Constant(1), V))
        return
    v = assemble(interpolate(Constant(1), V))
    A_w = Interpolator(TestFunction(V), W)
    w_v = Function(W)
    assemble(A_w.interpolate(v), tensor=w_v)
    assert np.allclose(w_v.dat.data_ro, 1.)


def test_extruded_cell_parent_cell_list():
    # If we make a function space that has 1 dof per cell, then we can use the
    # cell_parent_list directly to see if we get expected values. This is a
    # carbon copy of tests/regression/test_locate_cell.py

    ms = UnitSquareMesh(3, 3, quadrilateral=True)
    mx = ExtrudedMesh(UnitIntervalMesh(3), 3)

    # coords at locations from tests/regression/test_locate_cell.py - note that
    # we are not at the cell midpoints
    coords = np.array([[0.2, 0.1], [0.5, 0.2], [0.7, 0.1], [0.2, 0.4], [0.4, 0.4], [0.8, 0.5], [0.1, 0.7], [0.5, 0.9], [0.9, 0.8]])

    vms = VertexOnlyMesh(ms, coords, missing_points_behaviour=None)
    vmx = VertexOnlyMesh(mx, coords, missing_points_behaviour=None)
    assert vms.num_cells() == len(coords)
    assert vmx.num_cells() == len(coords)
    assert np.equal(vms.coordinates.dat.data_ro, coords[vms.topology._dm_renumbering]).all()
    assert np.equal(vmx.coordinates.dat.data_ro, coords[vmx.topology._dm_renumbering]).all()

    # set up test as in tests/regression/test_locate_cell.py - DG0 has 1 dof
    # per cell which is the expression evaluated at the cell midpoint.
    Vs = FunctionSpace(ms, 'DG', 0)
    Vx = FunctionSpace(mx, 'DG', 0)
    fs = Function(Vs)
    fx = Function(Vx)
    xs = SpatialCoordinate(ms)
    xx = SpatialCoordinate(mx)
    fs.interpolate(3*xs[0] + 9*xs[1] - 1)
    fx.interpolate(3*xx[0] + 9*xx[1] - 1)

    # expected values at coordinates from tests/regression/test_locate_cell.py
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert np.allclose(fs.at(coords), expected)
    assert np.allclose(fx.at(coords), expected)
    assert np.allclose(fs.dat.data[vms.cell_parent_cell_list], expected[vms.topology._dm_renumbering])
    assert np.allclose(fx.dat.data[vmx.cell_parent_cell_list], expected[vmx.topology._dm_renumbering])


@pytest.mark.parallel
def test_scalar_spatialcoordinate_interpolation_parallel(parentmesh, vertexcoords):
    test_scalar_spatialcoordinate_interpolation(parentmesh, vertexcoords)


@pytest.mark.parallel
def test_vector_spatialcoordinate_interpolation_parallel(parentmesh, vertexcoords):
    test_vector_spatialcoordinate_interpolation(parentmesh, vertexcoords)


@pytest.mark.parallel
def test_vector_function_interpolation_parallel(parentmesh, vertexcoords, vfs):
    test_vector_function_interpolation(parentmesh, vertexcoords, vfs)


@pytest.mark.parallel
def test_tensor_spatialcoordinate_interpolation_parallel(parentmesh, vertexcoords):
    test_tensor_spatialcoordinate_interpolation(parentmesh, vertexcoords)


@pytest.mark.parallel
def test_tensor_function_interpolation_parallel(parentmesh, vertexcoords, tfs):
    test_tensor_function_interpolation(parentmesh, vertexcoords, tfs)


@pytest.mark.parallel
def test_mixed_function_interpolation_parallel(parentmesh, vertexcoords, tfs):
    test_mixed_function_interpolation(parentmesh, vertexcoords, tfs)

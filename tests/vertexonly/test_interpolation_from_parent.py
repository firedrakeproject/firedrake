from firedrake import *
import pytest
import numpy as np


# Utility Functions and Fixtures

# NOTE we don't include interval mesh since many of the function spaces
# here are not defined on it.
@pytest.fixture(params=[pytest.param("interval", marks=pytest.mark.xfail(reason="swarm not implemented in 1d")),
                        "square",
                        "squarequads",
                        pytest.param("extruded", marks=pytest.mark.xfail(reason="extruded meshes not supported")),
                        "cube",
                        "tetrahedron",
                        pytest.param("immersedsphere", marks=pytest.mark.xfail(reason="immersed parent meshes not supported")),
                        pytest.param("periodicrectangle", marks=pytest.mark.xfail(reason="meshes made from coordinate fields are not supported"))],
                ids=lambda x: f"{x}-mesh")
def parentmesh(request):
    if request.param == "interval":
        return UnitIntervalMesh(1)
    elif request.param == "square":
        return UnitSquareMesh(1, 1)
    elif request.param == "squarequads":
        return UnitSquareMesh(2, 2, quadrilateral=True)
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
                        ("RTCE", 2, FunctionSpace),
                        ("RTCF", 2, FunctionSpace)],
                ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})")
def vfs(request):
    return request.param


@pytest.fixture(params=[("CG", 2, TensorFunctionSpace),
                        pytest.param(("BDM", 2, VectorFunctionSpace), marks=pytest.mark.skip("Interpolating an expression into BDM takes a very long time!")),
                        ("Regge", 2, FunctionSpace)],
                ids=lambda x: f"{x[2].__name__}({x[0]}{x[1]})")
def tfs(request):
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


# Tests

# NOTE: these _spatialcoordinate tests should be equivalent to some kind of
# interpolation from a CG1 VectorFunctionSpace (I think)
def test_scalar_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
    W = FunctionSpace(vm, "DG", 0)
    from functools import reduce
    from operator import add
    expr = reduce(add, SpatialCoordinate(parentmesh))
    w_expr = interpolate(expr, W)
    assert np.allclose(w_expr.dat.data_ro, np.sum(vertexcoords, axis=1))


def test_scalar_function_interpolation(parentmesh, vertexcoords, fs):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
    fs_fam, fs_deg, fs_typ = fs
    V = fs_typ(parentmesh, fs_fam, fs_deg)
    W = FunctionSpace(vm, "DG", 0)
    from functools import reduce
    from operator import add
    expr = reduce(add, SpatialCoordinate(parentmesh))
    v = Function(V).interpolate(expr)
    w_v = interpolate(v, W)
    assert np.allclose(w_v.dat.data_ro, np.sum(vertexcoords, axis=1))


# TODO: Add vector and tensor tests
def test_scalar_function_interpolator(parentmesh, vertexcoords, fs):
    # try and make reusable Interpolator from V to W
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
    fs_fam, fs_deg, fs_typ = fs
    V = fs_typ(parentmesh, fs_fam, fs_deg)
    W = FunctionSpace(vm, "DG", 0)
    from functools import reduce
    from operator import add
    expr = reduce(add, SpatialCoordinate(parentmesh))
    A_w = Interpolator(TestFunction(V), W)
    w_v = Function(W)
    v = Function(V).interpolate(expr)
    A_w.interpolate(v, output=w_v)
    assert np.allclose(w_v.dat.data_ro, np.sum(vertexcoords, axis=1))
    # use if again for a different Function in V
    v = Function(V).assign(Constant(2))
    A_w.interpolate(v, output=w_v)
    assert np.allclose(w_v.dat.data_ro, 2)


def test_vector_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
    W = VectorFunctionSpace(vm, "DG", 0)
    expr = 2 * SpatialCoordinate(parentmesh)
    w_expr = interpolate(expr, W)
    assert np.allclose(w_expr.dat.data_ro, 2*np.asarray(vertexcoords))


def test_vector_function_interpolation(parentmesh, vertexcoords, vfs):
    vfs_fam, vfs_deg, vfs_typ = vfs
    # skip where the element doesn't support the cell type
    if vfs_fam != "CG":
        if parentmesh.ufl_cell().cellname() == "quadrilateral":
            if not (vfs_fam == "RTCE" or vfs_fam == "RTCF"):
                pytest.skip(f"{vfs_fam} does not support {parentmesh.ufl_cell()} cells")
        elif parentmesh.ufl_cell().cellname() == "triangle" or parentmesh.ufl_cell().cellname() == "tetrahedron":
            if (not (vfs_fam == "N1curl" or vfs_fam == "N2curl"
                     or vfs_fam == "N1div" or vfs_fam == "N2div")):
                pytest.skip(f"{vfs_fam} does not support {parentmesh.ufl_cell()} cells")
            # TODO: Remove this else when fixed
            else:
                if parentmesh.ufl_cell().cellname() == "tetrahedron" and vfs_fam == "N2div":
                    pytest.skip("N2div on tetrahedron cells is broken upstream - causes hanging (try tests/regression/test_interpolation_nodes.py and see!)")
        else:
            pytest.skip(f"{vfs_fam} does not support {parentmesh.ufl_cell()} cells")
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
    V = vfs_typ(parentmesh, vfs_fam, vfs_deg)
    W = VectorFunctionSpace(vm, "DG", 0)
    expr = 2 * SpatialCoordinate(parentmesh)
    v = Function(V).interpolate(expr)
    w_v = interpolate(v, W)
    assert np.allclose(w_v.dat.data_ro, 2*np.asarray(vertexcoords))


def test_tensor_spatialcoordinate_interpolation(parentmesh, vertexcoords):
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
    W = TensorFunctionSpace(vm, "DG", 0)
    x = SpatialCoordinate(parentmesh)
    gdim = parentmesh.geometric_dimension()
    expr = 2 * as_tensor([x]*gdim)
    assert W.shape == expr.ufl_shape
    w_expr = interpolate(expr, W)
    result = 2 * np.asarray([[vertexcoords[i]]*gdim for i in range(len(vertexcoords))])
    if len(result) == 0:
        result = result.reshape(vertexcoords.shape + (gdim,))
    assert np.allclose(w_expr.dat.data_ro, result)


def test_tensor_function_interpolation(parentmesh, vertexcoords, tfs):
    tfs_fam, tfs_deg, tfs_typ = tfs
    # skip where the element doesn't support the cell type
    if (tfs_fam != "CG" and parentmesh.ufl_cell().cellname() != "triangle"
            and parentmesh.ufl_cell().cellname() != "tetrahedron"):
        pytest.skip(f"{tfs_fam} does not support {parentmesh.ufl_cell()} cells")
    vm = VertexOnlyMesh(parentmesh, vertexcoords)
    vertexcoords = vm.coordinates.dat.data_ro
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
    w_v = interpolate(v, W)
    assert np.allclose(w_v.dat.data_ro, result)

# TODO: Add parallel tests
# @pytest.mark.parallel
# def test_scalar_spatialcoordinate_interpolation_parallel(parentmesh, vertexcoords):
#     test_scalar_spatialcoordinate_interpolation(parentmesh, vertexcoords)

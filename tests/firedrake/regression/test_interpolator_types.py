from firedrake import *
from firedrake.interpolation import (
    MixedInterpolator, SameMeshInterpolator, CrossMeshInterpolator,
    get_interpolator, VomOntoVomInterpolator,
)
from firedrake.matrix import ImplicitMatrix, Matrix
import pytest


def params():
    params = []
    for mat_type in [None, "aij", "matfree"]:
        params.append(pytest.param(mat_type, None, id=f"mat_type={mat_type}"))
    for sub_mat_type in [None, "aij", "baij"]:
        params.append(pytest.param("nest", sub_mat_type, id=f"nest_sub_mat_type={sub_mat_type}"))
    return params


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij", "baij", "matfree"], ids=lambda v: f"mat_type={v}")
@pytest.mark.parametrize("mode", ["forward", "adjoint"], ids=lambda v: f"mode={v}")
def test_same_mesh_mattype(value_shape, mat_type, mode):
    if COMM_WORLD.size > 1:
        prefix = "mpi"
    else:
        prefix = "seq"

    mesh = UnitSquareMesh(10, 10)
    x, y = SpatialCoordinate(mesh)
    if value_shape == "scalar":
        fs_type = FunctionSpace
        expr = x**2 + y**2
    else:
        fs_type = VectorFunctionSpace
        expr = as_vector([x**2, y**2])

    V1 = fs_type(mesh, "CG", 1)
    V2 = fs_type(mesh, "CG", 2)

    if mode == "forward":
        exact = Function(V1).interpolate(expr)
        interp = interpolate(TrialFunction(V2), V1)  # V1 x V2^* -> R
        f = Function(V2).interpolate(expr)
    elif mode == "adjoint":
        v1 = TestFunction(V1)
        exact = assemble(inner(1, sum(v1) if value_shape == "vector" else v1) * dx)
        interp = interpolate(TestFunction(V1), TrialFunction(V2.dual()))  # V2^* x V1 -> R
        v2 = TestFunction(V2)
        f = inner(1, sum(v2) if value_shape == "vector" else v2) * dx

    assert isinstance(get_interpolator(interp), SameMeshInterpolator)

    I_mat = assemble(interp, mat_type=mat_type)
    assert isinstance(I_mat, ImplicitMatrix if mat_type == "matfree" else Matrix)

    if mat_type == "matfree":
        assert I_mat.petscmat.type == "python"
    elif value_shape == "scalar":
        # Always seqaij for scalar
        assert I_mat.petscmat.type == prefix + "aij"
    else:
        # Defaults to seqaij
        assert I_mat.petscmat.type == prefix + (mat_type if mat_type else "aij")

    res = assemble(action(I_mat, f))
    assert np.allclose(res.dat.data, exact.dat.data)

    with pytest.raises(NotImplementedError):
        # MatNest only implemented for interpolation between MixedFunctionSpaces
        assemble(interp, mat_type="nest")


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij", "matfree"], ids=lambda v: f"mat_type={v}")
def test_cross_mesh_mattype(value_shape, mat_type):
    mesh1 = UnitSquareMesh(8, 8)
    x1, y1 = SpatialCoordinate(mesh1)
    mesh2 = UnitSquareMesh(2, 2)
    x2, y2 = SpatialCoordinate(mesh2)
    if value_shape == "scalar":
        fs_type = FunctionSpace
        expr1 = x1**2 + y1**2
        expr2 = x2**2 + y2**2
    else:
        fs_type = VectorFunctionSpace
        expr1 = as_vector([x1**2, y1**2])
        expr2 = as_vector([x2**2, y2**2])
    V1 = fs_type(mesh1, "CG", 1)
    V2 = fs_type(mesh2, "CG", 1)

    interp = interpolate(TrialFunction(V1), V2)
    assert isinstance(get_interpolator(interp), CrossMeshInterpolator)

    I_mat = assemble(interp, mat_type=mat_type)
    assert isinstance(I_mat, ImplicitMatrix if mat_type == "matfree" else Matrix)
    assert I_mat.petscmat.type == "python" if mat_type == "matfree" else "seqaij"

    f = Function(V1).interpolate(expr1)
    res = assemble(action(I_mat, f))
    exact = Function(V2).interpolate(expr2)
    assert np.allclose(res.dat.data, exact.dat.data)


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij", "baij", "matfree"], ids=lambda v: f"mat_type={v}")
def test_vomtovom_mattype(value_shape, mat_type):
    mesh = UnitSquareMesh(1, 1)
    points = [[0.1, 0.1]]
    vom = VertexOnlyMesh(mesh, points)
    if value_shape == "scalar":
        fs_type = FunctionSpace
    else:
        fs_type = VectorFunctionSpace
    P0DG = fs_type(vom, "DG", 0)
    P0DG_io = fs_type(vom.input_ordering, "DG", 0)

    interp = interpolate(TrialFunction(P0DG), P0DG_io)
    assert isinstance(get_interpolator(interp), VomOntoVomInterpolator)

    res = assemble(interp, mat_type=mat_type)
    assert isinstance(res, ImplicitMatrix if mat_type == "matfree" else Matrix)

    if not mat_type or mat_type == "matfree":
        assert res.petscmat.type == "python"
    else:
        if value_shape == "scalar":
            assert res.petscmat.type == "seqaij"
        else:
            # Defaults to seqaij
            assert res.petscmat.type == "seq" + (mat_type if mat_type else "aij")


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij", "baij", "matfree"], ids=lambda v: f"mat_type={v}")
def test_point_eval_mattype(value_shape, mat_type):
    mesh = UnitSquareMesh(1, 1)
    points = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
    vom = VertexOnlyMesh(mesh, points)
    if value_shape == "scalar":
        fs_type = FunctionSpace
    else:
        fs_type = VectorFunctionSpace
    P0DG = fs_type(vom, "DG", 0)
    V = fs_type(mesh, "CG", 1)

    interp = interpolate(TrialFunction(V), P0DG)
    assert isinstance(get_interpolator(interp), SameMeshInterpolator)
    res = assemble(interp, mat_type=mat_type)
    assert isinstance(res, ImplicitMatrix if mat_type == "matfree" else Matrix)

    if mat_type == "matfree":
        assert res.petscmat.type == "python"
    elif value_shape == "scalar":
        assert res.petscmat.type == "seqaij"
    else:
        # Defaults to seqaij
        assert res.petscmat.type == "seq" + (mat_type if mat_type else "aij")


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type,sub_mat_type", params())
def test_mixed_same_mesh_mattype(value_shape, mat_type, sub_mat_type):
    mesh = UnitSquareMesh(5, 5)
    x, y = SpatialCoordinate(mesh)
    if value_shape == "scalar":
        fs_type = FunctionSpace
    else:
        fs_type = VectorFunctionSpace
    V1 = fs_type(mesh, "CG", 1)
    V2 = fs_type(mesh, "CG", 2)
    V3 = fs_type(mesh, "CG", 3)
    V4 = fs_type(mesh, "CG", 4)

    W = V1 * V2
    U = V3 * V4

    if value_shape == "scalar":
        expr = as_vector([x**2, y**2])
    else:
        expr = as_vector([x**2, x**2, y**2, y**2])

    interp = interpolate(TrialFunction(U), W)
    assert isinstance(get_interpolator(interp), MixedInterpolator)
    I_mat = assemble(interp, mat_type=mat_type, sub_mat_type=sub_mat_type)
    assert isinstance(I_mat, ImplicitMatrix if mat_type == "matfree" else Matrix)

    if mat_type == "matfree":
        assert I_mat.petscmat.type == "python"
    elif not mat_type or mat_type == "aij":
        assert I_mat.petscmat.type == "seqaij"
    else:
        assert I_mat.petscmat.type == "nest"
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            sub_mat = I_mat.petscmat.getNestSubMatrix(i, j)
            if i != j:
                assert not sub_mat
                continue
            if value_shape == "scalar":
                # Always seqaij for scalar
                assert sub_mat.type == "seqaij"
            else:
                # matnest sub_mat_type defaults to aij
                assert sub_mat.type == "seq" + (sub_mat_type if sub_mat_type else "aij")

    f = Function(U).interpolate(expr)
    exact = Function(W).interpolate(expr)
    res = assemble(action(I_mat, f))
    for resi, exi in zip(res.subfunctions, exact.subfunctions):
        assert np.allclose(resi.dat.data, exi.dat.data)

    with pytest.raises(NotImplementedError):
        assemble(interp, mat_type="baij")

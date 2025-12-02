from firedrake import *
from firedrake.interpolation import (
    MixedInterpolator, SameMeshInterpolator, CrossMeshInterpolator,
    get_interpolator, VomOntoVomInterpolator,
)
import pytest


def params():
    params = []
    for mat_type in [None, "aij"]:
        params.append(pytest.param(mat_type, None, id=f"mat_type={mat_type}"))
    for sub_mat_type in [None, "aij", "baij"]:
        params.append(pytest.param("nest", sub_mat_type, id=f"nest_sub_mat_type={sub_mat_type}"))
    return params


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij", "baij"], ids=lambda v: f"mat_type={v}")
def test_same_mesh_mattype(value_shape, mat_type):
    if COMM_WORLD.size > 1:
        prefix = "mpi"
    else:
        prefix = "seq"
    mesh = UnitSquareMesh(4, 4)
    if value_shape == "scalar":
        fs_type = FunctionSpace
    else:
        fs_type = VectorFunctionSpace
    V1 = fs_type(mesh, "CG", 1)
    V2 = fs_type(mesh, "CG", 2)

    u = TrialFunction(V1)

    interp = interpolate(u, V2)
    assert isinstance(get_interpolator(interp), SameMeshInterpolator)
    res = assemble(interp, mat_type=mat_type)

    if value_shape == "scalar":
        # Always seqaij for scalar
        assert res.petscmat.type == prefix + "aij"
    else:
        # Defaults to seqaij
        assert res.petscmat.type == prefix + (mat_type if mat_type else "aij")

    with pytest.raises(NotImplementedError):
        # MatNest only implemented for interpolation between MixedFunctionSpaces
        assemble(interp, mat_type="nest")


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij"], ids=lambda v: f"mat_type={v}")
def test_cross_mesh_mattype(value_shape, mat_type):
    mesh1 = UnitSquareMesh(1, 1)
    mesh2 = UnitSquareMesh(1, 1)
    if value_shape == "scalar":
        fs_type = FunctionSpace
    else:
        fs_type = VectorFunctionSpace
    V1 = fs_type(mesh1, "CG", 1)
    V2 = fs_type(mesh2, "CG", 1)

    u = TrialFunction(V1)

    interp = interpolate(u, V2)
    assert isinstance(get_interpolator(interp), CrossMeshInterpolator)
    res = assemble(interp, mat_type=mat_type)

    # only aij for cross-mesh
    assert res.petscmat.type == "seqaij"


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

    u = TrialFunction(P0DG)
    interp = interpolate(u, P0DG_io)
    assert isinstance(get_interpolator(interp), VomOntoVomInterpolator)
    res = assemble(interp, mat_type=mat_type)
    if not mat_type or mat_type == "aij":
        assert res.petscmat.type == "seqaij"
    elif mat_type == "matfree":
        assert res.petscmat.type == "python"
    else:
        if value_shape == "scalar":
            # Always seqaij for scalar
            assert res.petscmat.type == "seqaij"
        else:
            # Defaults to seqaij
            assert res.petscmat.type == "seq" + (mat_type if mat_type else "aij")


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type", [None, "aij", "baij"], ids=lambda v: f"mat_type={v}")
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

    u = TrialFunction(V)
    interp = interpolate(u, P0DG)
    assert isinstance(get_interpolator(interp), SameMeshInterpolator)
    res = assemble(interp, mat_type=mat_type)

    if value_shape == "scalar":
        # Always seqaij for scalar
        assert res.petscmat.type == "seqaij"
    else:
        # Defaults to seqaij
        assert res.petscmat.type == "seq" + (mat_type if mat_type else "aij")


@pytest.mark.parametrize("value_shape", ["scalar", "vector"], ids=lambda v: f"fs_type={v}")
@pytest.mark.parametrize("mat_type,sub_mat_type", params())
def test_mixed_same_mesh_mattype(value_shape, mat_type, sub_mat_type):
    mesh = UnitSquareMesh(1, 1)
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

    w = TrialFunction(W)
    w0, w1 = split(w)
    if value_shape == "scalar":
        expr = as_vector([w0 + w1, w0 + w1])
    else:
        w00, w01 = split(w0)
        w10, w11 = split(w1)
        expr = as_vector([w00 + w10, w00 + w10, w01 + w11, w01 + w11])
    interp = interpolate(expr, U)
    assert isinstance(get_interpolator(interp), MixedInterpolator)
    res = assemble(interp, mat_type=mat_type, sub_mat_type=sub_mat_type)
    if not mat_type or mat_type == "aij":
        # Defaults to seqaij
        assert res.petscmat.type == "seqaij"
    else:
        assert res.petscmat.type == "nest"
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            sub_mat = res.petscmat.getNestSubMatrix(i, j)
            if value_shape == "scalar":
                # Always seqaij for scalar
                assert sub_mat.type == "seqaij"
            else:
                # matnest sub_mat_type defaults to baij
                assert sub_mat.type == "seq" + (sub_mat_type if sub_mat_type else "baij")

    with pytest.raises(NotImplementedError):
        assemble(interp, mat_type="baij")

    with pytest.raises(NotImplementedError):
        assemble(interp, mat_type="matfree")

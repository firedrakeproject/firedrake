from firedrake import *
from firedrake.interpolation import (
    SameMeshInterpolator, CrossMeshInterpolator,
    get_interpolator, VomOntoVomInterpolator,
)
from firedrake.matrix import ImplicitMatrix, Matrix
import pytest
import numpy as np
import ufl
from firedrake.assemble import BaseFormAssembler


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
        forward_interp = interpolate(TrialFunction(V1), V2)  # V1 x V2^* -> R
        interp = interpolate(TestFunction(V1), TrialFunction(V2.dual()))  # V2^* x V1 -> R
        v2 = TestFunction(V2)
        f = inner(1, sum(v2) if value_shape == "vector" else v2) * dx

    assert isinstance(get_interpolator(interp), SameMeshInterpolator)

    I_mat = assemble(interp, mat_type=mat_type)
    assert isinstance(I_mat, ImplicitMatrix if mat_type == "matfree" else Matrix)

    if mat_type == "matfree":
        assert I_mat.petscmat.type == "python"
    else:
        # Defaults to seqaij
        assert I_mat.petscmat.type == prefix + (mat_type if mat_type else "aij")

    res = assemble(action(I_mat, f))
    assert np.allclose(res.dat.data, exact.dat.data)

    if mode == "adjoint":
        forward_I_mat = assemble(forward_interp, mat_type=mat_type)
        res2 = assemble(action(adjoint(forward_I_mat), f))
        assert np.allclose(res2.dat.data, exact.dat.data)

    # A nest holds the one block that these unmixed spaces make, just as it
    # does for a Form.
    nest_mat = assemble(interp, mat_type="nest")
    assert nest_mat.petscmat.type == "nest"
    assert nest_mat.petscmat.getNestSubMatrix(0, 0).type == prefix + "aij"
    res = assemble(action(nest_mat, f))
    assert np.allclose(res.dat.data, exact.dat.data)


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


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("mode", ["test", "trial", "both"])
@pytest.mark.parametrize("assemble_matrix", [False, True], ids=["action", "matrix"])
def test_cross_mesh_arguments_in_integrals(mode, assemble_matrix):
    source = UnitSquareMesh(3, 3)
    target = UnitSquareMesh(2, 2)
    V = FunctionSpace(source, "CG", 1)
    Q = FunctionSpace(target, "CG", 1)
    v = TestFunction(V if mode in ("test", "both") else Q)
    u = TrialFunction(V if mode in ("trial", "both") else Q)
    Iv = interpolate(v, Q) if mode in ("test", "both") else v
    Iu = interpolate(u, Q) if mode in ("trial", "both") else u
    a = inner(Iu, Iv) * dx(domain=target)
    A = assemble(a, mat_type="aij") if assemble_matrix else BaseFormAssembler.preprocess_base_form(a)

    # Compare the assembled operator against independent forward and adjoint
    # interpolation of a coefficient and the resulting linear functional.
    x, y = SpatialCoordinate(u.function_space().mesh())
    f = Function(u.function_space()).interpolate(x + 2*y)
    fq = assemble(interpolate(f, Q)) if mode in ("trial", "both") else f
    load = inner(fq, TestFunction(Q)) * dx
    expected = assemble(interpolate(v, load)) if mode in ("test", "both") else assemble(load)
    result = assemble(action(A, f))
    assert np.allclose(result.dat.data_ro, expected.dat.data_ro)

    lowered = BaseFormAssembler.preprocess_base_form(a)
    assert lowered.arguments() == a.arguments()
    assert isinstance(lowered, ufl.Action)
    assert BaseFormAssembler.preprocess_base_form(lowered) == lowered


@pytest.mark.parallel([1, 2])
def test_cross_mesh_sum_of_interpolated_arguments():
    source = UnitSquareMesh(3, 3)
    target = UnitSquareMesh(2, 2)
    V = VectorFunctionSpace(source, "CG", 1)
    Q = VectorFunctionSpace(target, "CG", 1)
    P = VectorFunctionSpace(target, "CG", 2)
    v = TestFunction(V)
    x, y = SpatialCoordinate(target)
    f = Function(P).interpolate(as_vector((x**2, y**2)))
    F = (inner(f, interpolate(v, Q)) + inner(f, interpolate(2*v, P))) * dx(domain=target)
    expected = assemble(interpolate(v, inner(f, TestFunction(Q)) * dx))
    expected += assemble(interpolate(2*v, inner(f, TestFunction(P)) * dx))
    result = assemble(F)
    assert np.allclose(result.dat.data_ro, expected.dat.data_ro)


@pytest.mark.parallel([1, 2])
@pytest.mark.parametrize("assemble_dual", [False, True], ids=["form", "cofunction"])
def test_cross_mesh_mixed_adjoint_interpolation(assemble_dual):
    target = UnitSquareMesh(2, 2)
    source = UnitSquareMesh(3, 3)
    Q = FunctionSpace(target, "CG", 1)
    R = FunctionSpace(source, "CG", 1)
    Z = Q * R
    _, w = TestFunctions(Z)
    x, y = SpatialCoordinate(target)
    load = inner(1 + x + 2*y, TestFunction(Q)) * dx
    dual = assemble(load) if assemble_dual else load
    interpolation = interpolate(w, dual)
    expected = assemble(interpolate(TestFunction(R), dual))

    for tensor in (None, Cofunction(Z.dual()).assign(99)):
        result = assemble(interpolation, tensor=tensor)
        assert result.function_space() == Z.dual()
        if tensor is not None:
            assert result is tensor
        assert np.allclose(result.subfunctions[0].dat.data_ro, 0)
        assert np.allclose(result.subfunctions[1].dat.data_ro, expected.dat.data_ro)


@pytest.mark.parallel([1, 2])
def test_cross_mesh_mixed_jump_derivative():
    mesh = UnitSquareMesh(3, 3)
    other = UnitSquareMesh(2, 2)
    Q = FunctionSpace(mesh, "CG", 1)
    R = FunctionSpace(other, "CG", 1)
    Z = Q * R
    z = Function(Z)
    z.sub(0).interpolate(SpatialCoordinate(mesh)[0])
    z.sub(1).interpolate(2*SpatialCoordinate(other)[1])
    t, s = split(z)
    v, w = TestFunctions(Z)
    F = inner(t - interpolate(s, Q), v - interpolate(w, Q)) * ds(domain=mesh)
    F += inner(s, w) * dx(domain=other)
    J = assemble(derivative(F, z), mat_type="aij")
    # This residual is linear, so applying its Jacobian to its coefficient
    # must recover the residual, including the un-interpolated contribution.
    result = assemble(action(J, z))
    expected = assemble(F)
    assert np.allclose(result.dat.data_ro, expected.dat.data_ro)


def test_same_mesh_argument_interpolation_stays_fused():
    mesh = UnitSquareMesh(2, 2)
    V = FunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    a = inner(interpolate(TrialFunction(V), Q), TestFunction(Q)) * dx
    lowered = BaseFormAssembler.preprocess_base_form(a)
    assert isinstance(lowered, ufl.Form)
    assert lowered.base_form_operators()


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
    assert isinstance(get_interpolator(interp), SameMeshInterpolator)
    I_mat = assemble(interp, mat_type=mat_type, sub_mat_type=sub_mat_type)
    assert isinstance(I_mat, ImplicitMatrix if mat_type == "matfree" else Matrix)

    if mat_type == "matfree":
        assert I_mat.petscmat.type == "python"
    elif not mat_type or mat_type == "aij":
        assert I_mat.petscmat.type == "seqaij"
    else:
        assert I_mat.petscmat.type == "nest"
        # matnest sub_mat_type defaults to aij
        sub_type = "seq" + (sub_mat_type if sub_mat_type else "aij")
        for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            # Every block is assembled, as it is for a Form.  The components
            # do not mix, so the off-diagonal ones assemble to zero.
            sub_mat = I_mat.petscmat.getNestSubMatrix(i, j)
            assert sub_mat.type == sub_type
            if i != j:
                assert sub_mat.norm() == 0.0

    f = Function(U).interpolate(expr)
    exact = Function(W).interpolate(expr)
    res = assemble(action(I_mat, f))
    for resi, exi in zip(res.subfunctions, exact.subfunctions):
        assert np.allclose(resi.dat.data, exi.dat.data)

    with pytest.raises(ValueError, match="BAIJ matrix type makes no sense"):
        # A mixed space has no block structure to give BAIJ, as for a Form.
        assemble(interp, mat_type="baij")

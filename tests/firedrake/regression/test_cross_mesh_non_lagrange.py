from firedrake import *
import pytest
import numpy as np
from finat.quadrature import QuadratureRule
from functools import partial

from ufl.compound_expressions import deviatoric_expr_2x2


def fs_shape(V):
    shape = V.ufl_function_space().value_shape
    if len(shape) == 1:
        fs_type = partial(VectorFunctionSpace, dim=shape[0])
    elif len(shape) == 2:
        fs_type = partial(TensorFunctionSpace, shape=shape)
    else:
        raise ValueError("Invalid function space shape")
    return fs_type


def make_quadrature_space(V):
    """Builds a quadrature space on the target mesh.

    This space has point evaluation dofs at the quadrature PointSet
    of the target space's element

    """
    fe = V.finat_element
    _, ps = fe.dual_basis
    wts = np.full(len(ps.points), np.nan)  # These can be any number since we never integrate
    scheme = QuadratureRule(ps, wts, ref_el=fe.cell)
    element = FiniteElement("Quadrature", degree=fe.degree, quad_scheme=scheme)
    return fs_shape(V)(V.mesh(), element)


@pytest.fixture(params=[("RT", 2), ("RT", 3), ("RT", 4), ("BDM", 1), ("BDM", 2), ("BDM", 3),
                        ("BDFM", 2), ("HHJ", 2), ("N1curl", 2), ("N1curl", 3), ("N1curl", 4),
                        ("N2curl", 1), ("N2curl", 2), ("N2curl", 3), ("GLS", 2), ("GLS", 3),
                        ("GLS", 4), ("GLS2", 1), ("GLS2", 2), ("GLS2", 3)],
                ids=lambda x: f"{x[0]}_{x[1]}")
def V(request):
    element, degree = request.param
    mesh = UnitSquareMesh(8, 8)
    return FunctionSpace(mesh, element, degree)


def test_cross_mesh_oneform(V):
    mesh1 = UnitSquareMesh(5, 5)
    mesh2 = V.mesh()
    x, y = SpatialCoordinate(mesh1)
    x1, y1 = SpatialCoordinate(mesh2)

    shape = V.ufl_function_space().value_shape
    if len(shape) == 1:
        fs_type = partial(VectorFunctionSpace, dim=shape[0])
        expr1 = as_vector([x, y])
        expr2 = as_vector([x1, y1])
    elif len(shape) == 2:
        fs_type = partial(TensorFunctionSpace, shape=shape)
        expr1 = as_tensor([[x, x*y], [x*y, y]])
        expr2 = as_tensor([[x1, x1*y1], [x1*y1, y1]])
    else:
        raise ValueError("Unsupported target space shape")

    V_source = fs_type(mesh1, "CG", 2)
    f_source = Function(V_source).interpolate(expr1)

    f_target = assemble(interpolate(f_source, V))
    f_direct = Function(V).interpolate(expr2)

    assert np.allclose(f_target.dat.data_ro, f_direct.dat.data_ro)


def test_cross_mesh_twoform(V):
    mesh1 = UnitSquareMesh(5, 5)
    mesh2 = V.mesh()
    x, y = SpatialCoordinate(mesh1)
    x1, y1 = SpatialCoordinate(mesh2)

    shape = V.ufl_function_space().value_shape
    if len(shape) == 1:
        fs_type = partial(VectorFunctionSpace, dim=shape[0])
        expr1 = as_vector([x, y])
        expr2 = as_vector([x1, y1])
    elif len(shape) == 2:
        fs_type = partial(TensorFunctionSpace, shape=shape)
        expr1 = as_tensor([[x, x*y], [x*y, y]])
        expr2 = as_tensor([[x1, x1*y1], [x1*y1, y1]])
    else:
        raise ValueError("Unsupported target space shape")

    V_source = fs_type(mesh1, "CG", 2)

    I = assemble(interpolate(TrialFunction(V_source), V))  # V_source x V^* -> R

    f_source = Function(V_source).interpolate(expr1)
    f_direct = Function(V).interpolate(expr2)

    f_interpolated = assemble(action(I, f_source))
    assert np.allclose(f_interpolated.dat.data_ro, f_direct.dat.data_ro)


def test_cross_mesh_oneform_adjoint(V):
    # Can already do Lagrange -> RT adjoint
    # V^* -> Q^* -> V_target^*
    mesh1 = UnitSquareMesh(2, 2)
    x1 = SpatialCoordinate(mesh1)
    V_target = fs_shape(V)(mesh1, "CG", 1)

    mesh2 = V.mesh()
    x2 = SpatialCoordinate(mesh2)

    if len(V.value_shape) > 1:
        expr = outer(x2, x2)
        target_expr = outer(x1, x1)
        if V.ufl_element().mapping() == "covariant contravariant Piola":
            expr = deviatoric_expr_2x2(expr)
            target_expr = deviatoric_expr_2x2(target_expr)
    else:
        expr = x2
        target_expr = x1

    oneform_V = inner(expr, TestFunction(V)) * dx

    # Q_target = make_quadrature_space(V)

    # # Interp V^* -> Q^*
    # I1_adj = interpolate(TestFunction(Q_target), oneform_V)  # SameMesh
    # cofunc_Q = assemble(I1_adj)

    # # Interp Q^* -> V_target^*
    # I2_adj = interpolate(TestFunction(V_target), cofunc_Q)  # CrossMesh
    # cofunc_V = assemble(I2_adj)

    cofunc_V = assemble(interpolate(TestFunction(V_target), oneform_V))  # V^* -> V_target^*

    cofunc_V_direct = assemble(inner(target_expr, TestFunction(V_target)) * dx)
    assert np.allclose(cofunc_V.dat.data_ro, cofunc_V_direct.dat.data_ro)


def test_cross_mesh_twoform_adjoint(V):
    # V^* -> Q^* -> V_target^*
    mesh1 = UnitSquareMesh(2, 2)
    x1 = SpatialCoordinate(mesh1)
    V_target = fs_shape(V)(mesh1, "CG", 1)
    mesh2 = V.mesh()
    x2 = SpatialCoordinate(mesh2)

    if len(V.value_shape) > 1:
        expr = outer(x2, x2)
        target_expr = outer(x1, x1)
        if V.ufl_element().mapping() == "covariant contravariant Piola":
            expr = deviatoric_expr_2x2(expr)
            target_expr = deviatoric_expr_2x2(target_expr)
    else:
        expr = x2
        target_expr = x1

    oneform_V = inner(expr, TestFunction(V)) * dx

    I = assemble(interpolate(TestFunction(V_target), V))  # V^* x V_target -> R
    assert I.arguments() == (TestFunction(V_target), TrialFunction(V.dual()))

    cofunc_V = assemble(action(I, oneform_V))
    cofunc_V_direct = assemble(inner(target_expr, TestFunction(V_target)) * dx)

    assert np.allclose(cofunc_V.dat.data_ro, cofunc_V_direct.dat.data_ro)


if __name__ == "__main__":
    pytest.main([__file__ + "::test_cross_mesh_oneform_adjoint[RT_2]"])

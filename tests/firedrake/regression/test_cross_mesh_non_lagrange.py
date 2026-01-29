from firedrake import *
import pytest
import numpy as np
from finat.quadrature import QuadratureRule
from functools import partial


def mat_equals(a, b) -> bool:
    """Check that two Matrices are equal."""
    a = a.petscmat.copy()
    a.axpy(-1.0, b.petscmat)
    return a.norm(norm_type=PETSc.NormType.NORM_FROBENIUS) < 1e-14


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


@pytest.fixture(params=[("RT", 1), ("RT", 2), ("RT", 3), ("RT", 4), ("BDM", 1), ("BDM", 2), ("BDM", 3),
                        ("BDFM", 2), ("HHJ", 2), ("N1curl", 1), ("N1curl", 2), ("N1curl", 3), ("N1curl", 4),
                        ("N2curl", 1), ("N2curl", 2), ("N2curl", 3), ("GLS", 1), ("GLS", 2), ("GLS", 3),
                        ("GLS", 4), ("GLS2", 1), ("GLS2", 2), ("GLS2", 3)],
                ids=lambda x: f"{x[0]}_{x[1]}")
def V(request):
    element, degree = request.param
    mesh = UnitSquareMesh(8, 8)
    return FunctionSpace(mesh, element, degree)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("rank", [1, 2])
def test_cross_mesh(V, rank):
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
    f_direct = Function(V).interpolate(expr2)

    Q = make_quadrature_space(V)

    if rank == 2:
        # Assemble the operator
        I1 = interpolate(TrialFunction(V_source), Q)  # V_source x Q_target^* -> R
        I2 = interpolate(TrialFunction(Q), V)  # Q_target x V^* -> R
        I_manual = assemble(action(I2, I1))  # V_source x V^* -> R
        assert I_manual.arguments() == (TestFunction(V.dual()), TrialFunction(V_source))
        # Direct assembly
        I_direct = assemble(interpolate(TrialFunction(V_source), V))  # V_source
        assert I_direct.arguments() == (TestFunction(V.dual()), TrialFunction(V_source))
        assert mat_equals(I_manual, I_direct)

        f_interpolated_manual = assemble(action(I_manual, f_source))
        assert np.allclose(f_interpolated_manual.dat.data_ro, f_direct.dat.data_ro)
        f_interpolated_direct = assemble(action(I_direct, f_source))
        assert np.allclose(f_interpolated_direct.dat.data_ro, f_direct.dat.data_ro)
    elif rank == 1:
        # Interp V_source -> Q
        I1 = interpolate(f_source, Q)  # SameMesh
        f_quadrature = assemble(I1)
        # Interp Q -> V
        I2 = interpolate(f_quadrature, V)  # CrossMesh
        f_interpolated_manual = assemble(I2)
        assert f_interpolated_manual.function_space() == V
        assert np.allclose(f_interpolated_manual.dat.data_ro, f_direct.dat.data_ro)

        f_interpolated_direct = assemble(interpolate(f_source, V))
        assert f_interpolated_direct.function_space() == V
        assert np.allclose(f_interpolated_direct.dat.data_ro, f_direct.dat.data_ro)


@pytest.mark.parallel([1, 3])
@pytest.mark.parametrize("rank", [0, 1, 2])
def test_cross_mesh_adjoint(V, rank):
    # Can already do Lagrange -> RT adjoint
    # V^* -> Q^* -> V_target^*
    name = V.ufl_element()._short_name
    deg = V.ufl_element().degree()
    if name in ["N1curl", "GLS", "RT"] and deg == 1:
        if name == "RT" and rank == 0:
            pass
        else:
            pytest.skip(f"Not exact for degree {deg} {name} elements")

    mesh1 = UnitSquareMesh(2, 2)
    x1 = SpatialCoordinate(mesh1)
    V_target = fs_shape(V)(mesh1, "CG", 1)

    mesh2 = V.mesh()
    x2 = SpatialCoordinate(mesh2)

    if len(V.value_shape) > 1:
        expr = outer(x2, x2)
        target_expr = outer(x1, x1)
        if V.ufl_element().mapping() == "covariant contravariant Piola":
            expr = dev(expr)
            target_expr = dev(target_expr)
    else:
        expr = x2
        target_expr = x1

    oneform_V = inner(expr, TestFunction(V)) * dx  # V^*
    cofunc_Vtarget_direct = assemble(inner(target_expr, TestFunction(V_target)) * dx)

    Q = make_quadrature_space(V)

    if rank == 2:
        # Assemble the operator
        I1 = interpolate(TestFunction(Q), V)  # V^* x Q -> R
        I2 = interpolate(TestFunction(V_target), Q)  # Q^* x V_target -> R
        I_manual = assemble(action(I2, I1))  # V^* x V_target -> R
        assert I_manual.arguments() == (TestFunction(V_target), TrialFunction(V.dual()))
        # Direct assembly
        I_direct = assemble(interpolate(TestFunction(V_target), V))  # V^* x V_target -> R
        assert I_direct.arguments() == (TestFunction(V_target), TrialFunction(V.dual()))
        assert mat_equals(I_manual, I_direct)

        cofunc_Vtarget_manual = assemble(action(I_manual, oneform_V))
        assert np.allclose(cofunc_Vtarget_manual.dat.data_ro, cofunc_Vtarget_direct.dat.data_ro)
        cofunc_Vtarget = assemble(action(I_direct, oneform_V))
        assert np.allclose(cofunc_Vtarget.dat.data_ro, cofunc_Vtarget_direct.dat.data_ro)
    elif rank == 1:
        # Interp V^* -> Q^*
        I1_adj = interpolate(TestFunction(Q), oneform_V)  # SameMesh
        cofunc_Q = assemble(I1_adj)

        # Interp Q^* -> V_target^*
        I2_adj = interpolate(TestFunction(V_target), cofunc_Q)  # CrossMesh
        cofunc_Vtarget_manual = assemble(I2_adj)
        assert np.allclose(cofunc_Vtarget_manual.dat.data_ro, cofunc_Vtarget_direct.dat.data_ro)

        cofunc_Vtarget = assemble(interpolate(TestFunction(V_target), oneform_V))  # V^* -> V_target^*
        assert np.allclose(cofunc_Vtarget.dat.data_ro, cofunc_Vtarget_direct.dat.data_ro)
    elif rank == 0:
        if name == "GLS2" and deg == 1:
            pytest.skip(f"Not exact for degree {deg} {name} elements")
        res = assemble(interpolate(target_expr, oneform_V))
        actual = assemble(inner(expr, expr) * dx)
        assert np.isclose(res, actual)

import pytest
import numpy as np
import ufl
from firedrake import *
from ufl.algorithms.comparison_checker import ComplexComparisonError


@pytest.mark.skipreal
@pytest.mark.parametrize("ncell",
                         [1, 4, 10])
def test_conditional(ncell):
    mesh = UnitIntervalMesh(ncell)
    V = FunctionSpace(mesh, "DG", 0)
    u = Function(V)
    du = TrialFunction(V)
    v = TestFunction(V)
    bhp = Constant(2)
    u.dat.data[...] = range(ncell)
    cond = conditional(ge(real(u-bhp), 0.0), u-bhp, 0.0)
    Fc = inner(cond, v) * dx

    A = assemble(derivative(Fc, u, du)).M.values
    expect = np.zeros_like(A)
    for i in range(2, ncell):
        expect[i, i] = 1.0/ncell

    assert np.allclose(A, expect)
    with pytest.raises(ComplexComparisonError):
        cond = conditional(ge(u-bhp, 0.0), u-bhp, 0.0)
        Fc = inner(cond, v) * dx

        A = assemble(derivative(Fc, u, du)).M.values


@pytest.mark.skipif(utils.complex_mode, reason="Differentiation of conditional unlikely to work in complex.")
def test_conditional_nan():
    # Test case courtesy of Marco Morandini:
    # https://github.com/firedrakeproject/tsfc/issues/183

    def crossTensor(V):
        return as_tensor([
            [0.0, V[2], -V[1]],
            [-V[2], 0.0, V[0]],
            [V[1], -V[0], 0.0]])

    def coefa(phi2):
        return conditional(le(phi2, 1.e-6),
                           phi2,
                           sin(phi2)/phi2)

    def RotDiffTensor(phi):
        PhiCross = crossTensor(phi)
        phi2 = dot(phi, phi)
        a = coefa(phi2)
        return PhiCross * a

    mesh = UnitIntervalMesh(8)
    V = VectorFunctionSpace(mesh, "CG", 1, dim=3)
    u_phi = Function(V)
    v_phi = TestFunction(V)

    Gamma = RotDiffTensor(u_phi)

    beta = dot(Gamma, grad(u_phi))
    delta_beta = ufl.derivative(beta, u_phi, v_phi)

    L = inner(delta_beta, grad(u_phi)) * dx
    result = assemble(L).dat.data

    # Check whether there are any NaNs in the result
    assert not np.isnan(result).any()

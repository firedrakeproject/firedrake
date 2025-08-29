import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, minimize,
    pause_annotation)
from firedrake.adjoint.transformed_functional import L2TransformedFunctional
import numpy as np
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
from pyadjoint.tape import set_working_tape
import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_tape():
    with set_working_tape():
        pause_annotation()
        yield
    pause_annotation()


def test_transformed_functional_mass_inverse():
    mesh = fd.UnitSquareMesh(5, 5, diagonal="crossed")
    x, y = fd.SpatialCoordinate(mesh)
    space = fd.FunctionSpace(mesh, "Lagrange", 1)

    m_ref = fd.Function(space, name="m_ref").interpolate(
        fd.exp(x) * fd.sin(fd.pi * x) * fd.cos(fd.pi * x))

    def forward(m):
        return fd.assemble(fd.inner(m - m_ref, m - m_ref) * fd.dx)

    continue_annotation()
    m_0 = fd.Function(space, name="m_0")
    J = forward(m_0)
    pause_annotation()
    c = Control(m_0, riesz_map="l2")

    class MinimizeCallback:
        def __init__(self):
            self._ncalls = 0

        @property
        def ncalls(self):
            return self._ncalls

        def __call__(self, xk):
            self._ncalls += 1

    J_hat = ReducedFunctional(J, c)
    cb = MinimizeCallback()
    m_opt = minimize(J_hat, method="L-BFGS-B",
                     callback=cb,
                     options={"ftol": 0,
                              "gtol": 1e-6})
    assert fd.norm(m_opt - m_ref, "L2") < 1e-4
    assert cb.ncalls > 10  # == 13

    J_hat = L2TransformedFunctional(J, c, alpha=1)
    cb = MinimizeCallback()
    m_opt = minimize(ReducedFunctionalNumPy(J_hat), method="L-BFGS-B",
                     callback=cb,
                     options={"ftol": 0,
                              "gtol": 1e-6})
    m_opt = J_hat.map_result(m_opt)
    assert fd.norm(m_opt - m_ref, "L2") < 1e-10
    assert cb.ncalls == 2

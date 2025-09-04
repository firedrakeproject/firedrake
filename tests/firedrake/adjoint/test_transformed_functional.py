from collections.abc import Sequence
from functools import partial

import firedrake as fd
from firedrake.adjoint import (
    Control, ReducedFunctional, continue_annotation, minimize,
    pause_annotation)
from firedrake.adjoint.transformed_functional import L2TransformedFunctional
import numpy as np
from pyadjoint import MinimizationProblem, TAOSolver
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy
from pyadjoint.tape import set_working_tape
import pytest
import ufl


@pytest.fixture(scope="module", autouse=True)
def setup_tape():
    with set_working_tape():
        pause_annotation()
        yield
    pause_annotation()


class ReducedFunctional(ReducedFunctional):
    def __init__(self, *args, **kwargs):
        self._test_transformed_functional__ncalls = 0
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._test_transformed_functional__ncalls += 1
        return super().__call__(*args, **kwargs)


class L2TransformedFunctional(L2TransformedFunctional):
    def __init__(self, *args, **kwargs):
        self._test_transformed_functional__ncalls = 0
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self._test_transformed_functional__ncalls += 1
        return super().__call__(*args, **kwargs)


class MinimizeCallback(Sequence):
    def __init__(self, m_0, error_norm):
        self._space = m_0.function_space()
        self._error_norm = error_norm
        self._data = []

        self(np.asarray(m_0._ad_to_list(m_0)))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __call__(self, xk):
        k = len(self)
        if ufl.duals.is_primal(self._space):
            m_k = fd.Function(self._space, name="m_k")
        elif ufl.duals.is_dual(self._space):
            m_k = fd.Cofunction(self._space, name="m_k")
        else:
            raise ValueError("space is neither primal nor dual")
        m_k._ad_assign_numpy(m_k, xk, 0)
        error_norm = self._error_norm(m_k)
        print(f"{k=} {error_norm=:6g}")
        self._data.append(error_norm)


@pytest.mark.parametrize("family", ("Lagrange", "Discontinuous Lagrange"))
def test_transformed_functional_mass_inverse(family):
    mesh = fd.UnitSquareMesh(5, 5, diagonal="crossed")
    x, y = fd.SpatialCoordinate(mesh)
    space = fd.FunctionSpace(mesh, family, 1, variant="equispaced")

    def forward(m):
        return fd.assemble(fd.inner(m - m_ref, m - m_ref) * fd.dx)

    m_ref = fd.Function(space, name="m_ref").interpolate(
        fd.exp(x) * fd.sin(fd.pi * x) * fd.cos(fd.pi * y))

    continue_annotation()
    m_0 = fd.Function(space, name="m_0")
    J = forward(m_0)
    pause_annotation()
    c = Control(m_0, riesz_map="l2")

    J_hat = ReducedFunctional(J, c)

    def error_norm(m):
        return fd.norm(m - m_ref, norm_type="L2")

    cb = MinimizeCallback(m_0, error_norm)
    _ = minimize(J_hat, method="L-BFGS-B",
                 callback=cb,
                 options={"ftol": 0,
                          "gtol": 1e-6})
    assert 1e-6 < cb[-1] < 1e-5
    if family == "Lagrange":
        assert len(cb) > 12  # == 15
        assert J_hat._test_transformed_functional__ncalls > 12  # == 15
    elif family == "Discontinuous Lagrange":
        assert len(cb) == 5
        assert J_hat._test_transformed_functional__ncalls == 6
    else:
        raise ValueError(f"Invalid element family: '{family}'")

    J_hat = L2TransformedFunctional(J, c, alpha=1)

    def error_norm(m):
        m = J_hat.map_result(m)
        return fd.norm(m - m_ref, norm_type="L2")

    cb = MinimizeCallback(J_hat.controls[0].control, error_norm)
    _ = minimize(ReducedFunctionalNumPy(J_hat), method="L-BFGS-B",
                 callback=cb,
                 options={"ftol": 0,
                          "gtol": 1e-6})
    assert cb[-1] < 1e-8
    assert len(cb) == 3
    assert J_hat._test_transformed_functional__ncalls == 3


def test_transformed_functional_poisson():
    mesh = fd.UnitSquareMesh(5, 5, diagonal="crossed")
    x, y = fd.SpatialCoordinate(mesh)
    space = fd.FunctionSpace(mesh, "Lagrange", 1)
    test = fd.TestFunction(space)
    trial = fd.TrialFunction(space)
    bc = fd.DirichletBC(space, 0, "on_boundary")

    def pre_process(m):
        m_0 = fd.Function(space, name="m_0").assign(m)
        bc.apply(m_0)
        m_1 = fd.Function(space, name="m_1").assign(m - m_0)
        return m_0, m_1

    def forward(m):
        m_0, m_1 = pre_process(m)
        u = fd.Function(space, name="u")
        fd.solve(fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
                 == fd.inner(m_0, test) * fd.dx,
                 u, bc)
        return m_0, m_1, u

    def forward_J(m, u_ref, alpha):
        _, m_1, u = forward(m)
        return fd.assemble(fd.inner(u - u_ref, u - u_ref) * fd.dx
                           + fd.Constant(alpha ** 2) * fd.inner(m_1, m_1) * fd.ds)

    m_ref = fd.Function(space, name="m_ref").interpolate(
        fd.exp(x) * fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
    m_ref, _, u_ref = forward(m_ref)
    forward_J = partial(forward_J, u_ref=u_ref, alpha=1)

    continue_annotation()
    m_0 = fd.Function(space, name="m_0")
    J = forward_J(m_0)
    pause_annotation()
    c = Control(m_0, riesz_map="l2")

    J_hat = ReducedFunctional(J, c)

    def error_norm(m):
        m, _ = pre_process(m)
        return fd.norm(m - m_ref, norm_type="L2")

    cb = MinimizeCallback(m_0, error_norm)
    _ = minimize(J_hat, method="L-BFGS-B",
                 callback=cb,
                 options={"ftol": 0,
                          "gtol": 1e-10})
    assert 1e-2 < cb[-1] < 5e-2
    assert len(cb) > 80  # == 85
    assert J_hat._test_transformed_functional__ncalls > 90  # == 95

    J_hat = L2TransformedFunctional(J, c, alpha=1e-5)

    def error_norm(m):
        m = J_hat.map_result(m)
        m, _ = pre_process(m)
        return fd.norm(m - m_ref, norm_type="L2")

    cb = MinimizeCallback(J_hat.controls[0].control, error_norm)
    _ = minimize(ReducedFunctionalNumPy(J_hat), method="L-BFGS-B",
                 callback=cb,
                 options={"ftol": 0,
                          "gtol": 1e-10})
    assert 1e-4 < cb[-1] < 5e-4
    assert len(cb) < 55  # == 50
    assert J_hat._test_transformed_functional__ncalls < 55  # == 51


def test_transformed_functional_poisson_tao_nls():
    mesh = fd.UnitSquareMesh(5, 5, diagonal="crossed")
    x, y = fd.SpatialCoordinate(mesh)
    space = fd.FunctionSpace(mesh, "Lagrange", 1)
    test = fd.TestFunction(space)
    trial = fd.TrialFunction(space)
    bc = fd.DirichletBC(space, 0, "on_boundary")

    def pre_process(m):
        m_0 = fd.Function(space, name="m_0").assign(m)
        bc.apply(m_0)
        m_1 = fd.Function(space, name="m_1").assign(m - m_0)
        return m_0, m_1

    def forward(m):
        m_0, m_1 = pre_process(m)
        u = fd.Function(space, name="u")
        fd.solve(fd.inner(fd.grad(trial), fd.grad(test)) * fd.dx
                 == fd.inner(m_0, test) * fd.dx,
                 u, bc)
        return m_0, m_1, u

    def forward_J(m, u_ref, alpha):
        _, m_1, u = forward(m)
        return fd.assemble(fd.inner(u - u_ref, u - u_ref) * fd.dx
                           + fd.Constant(alpha ** 2) * fd.inner(m_1, m_1) * fd.ds)

    m_ref = fd.Function(space, name="m_ref").interpolate(
        fd.exp(x) * fd.sin(fd.pi * x) * fd.sin(fd.pi * y))
    m_ref, _, u_ref = forward(m_ref)
    forward_J = partial(forward_J, u_ref=u_ref, alpha=1)

    continue_annotation()
    m_0 = fd.Function(space, name="m_0")
    J = forward_J(m_0)
    pause_annotation()
    c = Control(m_0)

    J_hat = ReducedFunctional(J, c)

    def error_norm(m):
        m, _ = pre_process(m)
        return fd.norm(m - m_ref, norm_type="L2")

    problem = MinimizationProblem(J_hat)
    solver = TAOSolver(problem, {"tao_type": "nls",
                                 "tao_monitor": None,
                                 "tao_converged_reason": None,
                                 "tao_gatol": 1.0e-5,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 1.0e-6,
                                 "tao_monitor": None})
    m_opt = solver.solve()
    error_norm_opt = error_norm(m_opt)
    print(f"{error_norm_opt=:.6g}")
    assert 1e-2 < error_norm_opt < 5e-2
    assert J_hat._test_transformed_functional__ncalls > 22  # == 24

    J_hat = L2TransformedFunctional(J, c, alpha=1e-5)

    def error_norm(m):
        m = J_hat.map_result(m)
        m, _ = pre_process(m)
        return fd.norm(m - m_ref, norm_type="L2")

    problem = MinimizationProblem(J_hat)
    solver = TAOSolver(problem, {"tao_type": "nls",
                                 "tao_monitor": None,
                                 "tao_converged_reason": None,
                                 "tao_gatol": 1.0e-5,
                                 "tao_grtol": 0.0,
                                 "tao_gttol": 1.0e-6,
                                 "tao_monitor": None})
    m_opt = solver.solve()
    error_norm_opt = error_norm(m_opt)
    print(f"{error_norm_opt=:.6g}")
    assert 1e-3 < error_norm_opt < 1e-2
    assert J_hat._test_transformed_functional__ncalls < 18  # == 16

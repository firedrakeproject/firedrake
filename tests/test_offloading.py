import pytest
from firedrake import (set_offloading_backend,
                       offloading, solve, FunctionSpace, TestFunction,
                       TrialFunction, Function, UnitSquareMesh,
                       SpatialCoordinate, inner, grad, dx, norm, pi, cos,
                       assemble)
import firedrake_configuration
from pyop2.backends.cpu import cpu_backend


AVAILABLE_BACKENDS = [cpu_backend]

if firedrake_configuration.get_config()["options"].get("cuda"):
    from pyop2.backends.cuda import cuda_backend
    AVAILABLE_BACKENDS.append(cuda_backend)


def allclose(a, b, rtol=1e-05, atol=1e-08):
    """
    Prefer this routine over np.allclose(...) to allow pycuda/pyopencl arrays
    """
    return bool(abs(a - b) < (atol + rtol * abs(b)))


@pytest.mark.parametrize("offloading_backend", AVAILABLE_BACKENDS)
def test_nonlinear_variational_solver(offloading_backend):
    set_offloading_backend(offloading_backend)
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(mesh)

    a = (inner(grad(u), grad(v)) + inner(u, v)) * dx
    f = Function(V)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    L = inner(f, v) * dx
    fem_soln = Function(V)
    sp = {"mat_type": "matfree",
          "ksp_monitor_true_residual": None,
          "ksp_converged_reason": None}
    with offloading():
        solve(a == L, fem_soln, solver_parameters=sp)

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))

    assert norm(fem_soln-f) < 1e-2

    with offloading():
        assert norm(fem_soln-f) < 1e-2


@pytest.mark.parametrize("offloading_backend", AVAILABLE_BACKENDS)
def test_linear_variational_solver(offloading_backend):
    set_offloading_backend(offloading_backend)
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    L = assemble(inner(f, v) * dx)
    fem_soln = Function(V)

    with offloading():

        a = assemble((inner(grad(u), grad(v)) + inner(u, v)) * dx,
                     mat_type="matfree")
        solve(a, fem_soln, L,
              solver_parameters={"pc_type": "none",
                                 "ksp_type": "cg",
                                 "ksp_monitor": None})

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))

    assert norm(fem_soln-f) < 1e-2

    with offloading():
        assert norm(fem_soln-f) < 1e-2


@pytest.mark.parametrize("offloading_backend", AVAILABLE_BACKENDS)
def test_data_manipulation_on_host(offloading_backend):
    set_offloading_backend(offloading_backend)

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    L = assemble(inner(f, v) * dx)
    fem_soln = Function(V)

    with offloading():

        a = assemble((inner(grad(u), grad(v)) + inner(u, v)) * dx,
                     mat_type="matfree")
        solve(a, fem_soln, L,
              solver_parameters={"pc_type": "none",
                                 "ksp_type": "cg",
                                 "ksp_monitor": None})

    old_norm = norm(fem_soln)
    kappa = 2.0
    fem_soln.dat.data[:] *= kappa  # update data on host

    with offloading():
        new_norm = norm(fem_soln)

    allclose(kappa*old_norm, new_norm)

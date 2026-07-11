import pytest
import warnings

from firedrake import *
from firedrake.adjoint import *


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.fixture()
def forcing():
    mesh = UnitSquareMesh(20,20)
    U = FunctionSpace(mesh, "CG", 1)
    return Function(U)


@pytest.fixture()
def nvp(forcing):
    U = forcing.function_space()
    u = Function(U)
    v = TestFunction(U)

    F = inner(grad(u), grad(v))*dx - inner(forcing, v)*dx

    bcs = [DirichletBC(U, Constant(1), (4,)),
        DirichletBC(U, Constant(0), (1, 2, 3))]

    return NonlinearVariationalProblem(F, u, bcs=bcs)


# Spike the adjoint solve so we can check that the options are actually
# set.
sp_lu = {
        "pc_type": "none",
        "ksp_type": "cg",
        "ksp_max_it": 1,
        "ksp_view": None
        }

#@pytest.mark.parametrize("mode", ("solver", "solve", "both", "none", "free"))


@pytest.mark.skipcomplex
def test_adj_kwargs_solver(nvp, forcing):
    nvs = NonlinearVariationalSolver(nvp,
                                     adj_kwargs={"solver_parameters": sp_lu})
    nvs.solve()
    u = nvp.u

    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, Control(forcing))

    with pytest.raises(ConvergenceError):
        Jhat.derivative()
       

@pytest.mark.skipcomplex
def test_adj_kwargs_none(nvp, forcing):
    # Don't pass adj_kwargs anywhere. Should succeed.
    nvs = NonlinearVariationalSolver(nvp)
    nvs.solve()
    u=nvp.u
    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, Control(forcing))
    Jhat.derivative()


@pytest.mark.skipcomplex
def test_adj_kwargs_both(nvp):
    nvs = NonlinearVariationalSolver(nvp,
                                     adj_kwargs={"solver_parameters": sp_lu})
    # Passing adj_kwargs to both the solver and solve() is an error.
    with pytest.raises(TypeError):
        nvs.solve(adj_kwargs={"solver_parameters": sp_lu})


@pytest.mark.skipcomplex
def test_adj_kwargs_solve(nvp, forcing):
    nvs = NonlinearVariationalSolver(nvp)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        nvs.solve(adj_kwargs={"solver_parameters": sp_lu})
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, FutureWarning)

    u = nvp.u
    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, Control(forcing))

    with pytest.raises(ConvergenceError):
        Jhat.derivative()


# Unclear why this doesn't work.
@pytest.mark.xfail
@pytest.mark.skipcomplex
def test_adj_kwargs_solve_free_function(nvp, forcing):
    u = nvp.u

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Should not trigger warnings.
        solve(nvp.F == 0, u, adj_kwargs={"solver_parameters": sp_lu})
        # Verify no warning raised.
        assert not w

    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, Control(forcing))

    with pytest.raises(ConvergenceError):
        Jhat.derivative()

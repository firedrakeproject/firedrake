import pytest

from firedrake import *
from firedrake.adjoint import *
from numpy.testing import assert_approx_equal


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


@pytest.mark.skipcomplex
def test_adj_kwargs():

    mesh = UnitSquareMesh(20,20)
    U = FunctionSpace(mesh, "CG", 1)
    u = Function(U)
    v = TestFunction(U)
    f = Function(U)

    F = inner(grad(u),grad(v))*dx - inner(f,v)*dx

    bcs = [DirichletBC(U, Constant(1), (4,)),
        DirichletBC(U, Constant(0), (1, 2, 3))]

    # Spike the adjoint solve so we can check that the options are actually
    # set.
    sp_lu = {
            "pc_type": "none",
            "ksp_type": "cg",
            "ksp_max_it": 1,
            }

    nvp = NonlinearVariationalProblem(F, u, bcs=bcs)
    # This errors
    nvs = NonlinearVariationalSolver(nvp,
                                     adj_kwargs={"solver_parameters": sp_lu})
    nvs.solve()
    J = assemble(inner(u, u)*dx)
    Jhat = ReducedFunctional(J, Control(f))
    with pytest.raises(ConvergenceError):
        Jhat.derivative()

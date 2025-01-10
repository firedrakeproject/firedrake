import pytest
import functools
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint.tape import get_working_tape, pause_annotation


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake.adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Ensure annotation is paused when we finish.
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_translation_operator_inverse_problem():

    class TranslationOperator(AbstractExternalOperator):

        def __init__(self, *operands, function_space, **kwargs):
            AbstractExternalOperator.__init__(self, *operands, function_space=function_space, **kwargs)

        @assemble_method(0, (0,))
        def assemble_N(self, *args, **kwargs):
            f, f0 = self.ufl_operands
            N = assemble(f - f0)
            return N

        @assemble_method((1, 0), (None, 0))
        def assemble_Jacobian_adjoint_action(self, *args, **kwargs):
            y, _ = self.argument_slots()
            return y

    mesh = UnitSquareMesh(50, 50)
    V = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)
    u_exact = Function(V).interpolate(sin(pi * x) * sin(pi * y))
    f_exact = Function(V).interpolate((2 * pi ** 2 + 1) * u_exact)

    # Set observed data
    u_obs = Function(V).assign(u_exact)
    # Add noise
    scale_noise = 5e-2
    noise = scale_noise * np.random.rand(V.node_count)
    u_obs.dat.data[:] += noise

    alpha = 1e-2

    f_0 = Function(V).assign(f_exact)

    u = Function(V)
    f = Function(V)
    v = TestFunction(V)
    bcs = DirichletBC(V, 0, 'on_boundary')

    R = functools.partial(TranslationOperator, function_space=V)

    def J(f):
        F = (inner(grad(u), grad(v)) + inner(u, v) - inner(f, v)) * dx
        solve(F == 0, u, bcs=bcs)
        return assemble(0.5 * (u - u_obs) ** 2 * dx + 0.5 * alpha * R(f, f_0) ** 2 * dx)

    c = Control(f)
    Jhat = ReducedFunctional(J(f), c)

    f_opt = minimize(Jhat, tol=1e-6, method="BFGS")

    assert assemble((f_exact - f_opt)**2 * dx) / assemble(f_exact**2 * dx) < 1e-5

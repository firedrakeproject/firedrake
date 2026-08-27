import pytest
import functools
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from pyadjoint.tape import get_working_tape, pause_annotation

try:
    import torch
    from torch.nn import Linear, Module
    from torch.nn.functional import softplus
    from firedrake.ml.pytorch import ml_operator

    class Diffusivity(Module):
        """Toy model for a strictly positive diffusivity ``1 + softplus(w u + b)``.

        The parameters are initialised deterministically so that the gradient
        verification in :func:`test_ml_operator_parameters_gradient` is reproducible.
        """

        def __init__(self):
            super().__init__()
            self.layer = Linear(1, 1)
            with torch.no_grad():
                self.layer.weight.fill_(0.5)
                self.layer.bias.fill_(0.1)

        def forward(self, x):
            return 1.0 + softplus(self.layer(x.view(-1, 1))).flatten()

except ImportError:
    # PyTorch is not installed
    pass


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

    f_opt = minimize(Jhat, tol=1e-4, method="BFGS")

    assert assemble((f_exact - f_opt)**2 * dx) / assemble(f_exact**2 * dx) < 1e-5


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_ml_operator_parameters_gradient():
    """Differentiate a PDE-constrained functional w.r.t. an ML operator's parameters.

    A :class:`~.PytorchOperator` carrying a trailing operand that acts as a handle on the
    model's trainable parameters is embedded in the residual of a nonlinear diffusion
    problem. Differentiating the reduced functional runs a reverse pass through the model,
    accumulating gradients into the PyTorch parameters -- this is what makes the embedded
    model trainable. Those gradients are checked against central finite differences of the
    functional, which is the property the whole training path hinges on.
    """
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)
    DG = FunctionSpace(mesh, "DG", 0)
    R = FunctionSpace(mesh, "R", 0)
    x, y = SpatialCoordinate(mesh)

    model = Diffusivity()
    model.double()
    kappa = ml_operator(model, function_space=DG)
    # Handle on the model's trainable parameters: the trailing operand.
    theta = Function(R).assign(0.0)
    f = Function(V).interpolate(sin(pi * x) * sin(pi * y))

    def functional():
        u, v = Function(V), TestFunction(V)
        F = (inner(kappa(interpolate(u, DG), theta) * grad(u), grad(v)) * dx
             + inner(u, v) * dx - inner(f, v) * dx)
        solve(F == 0, u, solver_parameters={"snes_rtol": 1e-14, "snes_stol": 0.0})
        return assemble(inner(u, u) * dx)

    model.zero_grad()
    with set_working_tape():
        ReducedFunctional(functional(), [Control(theta)]).derivative()

    parameters = list(model.parameters())
    assert all(p.grad is not None for p in parameters)
    gradient = torch.cat([p.grad.reshape(-1) for p in parameters])
    assert torch.all(torch.isfinite(gradient))
    # The reverse pass must actually reach the parameters.
    assert float(gradient.abs().sum()) > 0.0

    direction = torch.ones_like(gradient) / gradient.numel() ** 0.5
    analytic = float(torch.dot(gradient, direction))

    def perturb(step):
        with torch.no_grad():
            offset = 0
            for p in parameters:
                n = p.numel()
                p.add_(step * direction[offset:offset + n].view_as(p).double())
                offset += n

    eps = 1e-6
    with stop_annotating():
        perturb(eps)
        J_plus = float(functional())
        perturb(-2 * eps)
        J_minus = float(functional())
        perturb(eps)
    finite_difference = (J_plus - J_minus) / (2 * eps)

    assert np.isclose(analytic, finite_difference, rtol=1e-6)

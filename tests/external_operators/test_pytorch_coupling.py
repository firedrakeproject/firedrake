import pytest

import torch
import torch.nn.functional as torch_func
from torch.nn import Module, Flatten, Linear

from firedrake import *
from firedrake_adjoint import *
from firedrake.external_operators.neural_networks.backends import get_backend
from pyadjoint.tape import get_working_tape, pause_annotation


@pytest.fixture(autouse=True)
def handle_taping():
    yield
    tape = get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    from firedrake_adjoint import annotate_tape, continue_annotation
    if not annotate_tape():
        continue_annotation()
    yield
    # Since importing firedrake_adjoint modifies a global variable, we need to
    # pause annotations at the end of the module
    annotate = annotate_tape()
    if annotate:
        pause_annotation()


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture(scope='module')
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture
def f_exact(V, mesh):
    x, y = SpatialCoordinate(mesh)
    return Function(V).interpolate(sin(pi * x) * sin(pi * y))


class EncoderDecoder(Module):
    """Build a simple toy model"""

    def __init__(self, n):
        super(EncoderDecoder, self).__init__()
        self.n1 = n
        self.n2 = int(n/2)
        self.flatten = Flatten()
        self.encoder_1 = Linear(self.n1, self.n2)
        self.decoder_1 = Linear(self.n2, self.n1)

    def encode(self, x):
        return self.encoder_1(x)

    def decode(self, x):
        return self.decoder_1(x)

    def forward(self, x):
        # [batch_size, n]
        x = self.flatten(x)
        # [batch_size, n2]
        encoded = self.encode(x)
        hidden = torch_func.relu(encoded)
        # [batch_size, n]
        decoded = self.decode(hidden)
        return torch_func.relu(decoded)


# Set of Firedrake operations that will be composed with PyTorch operations
def poisson_residual(u, f, V):
    """Assemble the residual of a Poisson problem"""
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) + inner(u, v) - inner(f, v)) * dx
    return assemble(F)


# Set of Firedrake operations that will be composed with PyTorch operations
def solve_poisson(f, V):
    """Solve Poisson problem"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) + inner(u, v) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
    # Solve PDE
    solve(F == 0, u, bcs=bcs)
    # Assemble Firedrake loss
    return assemble(u ** 2 * dx)


@pytest.fixture(params=['poisson_residual', 'solve_poisson'])
def firedrake_operator(request, f_exact, V):
    # Return firedrake operator and the corresponding non-control arguments
    if request.param == 'poisson_residual':
        return poisson_residual, (f_exact, V)
    elif request.param == 'solve_poisson':
        return solve_poisson, (V,)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_pytorch_loss_backward(V, f_exact):
    """Add doc """

    # Instantiate model
    model = EncoderDecoder(V.dim())

    # Set double precision
    model.double()

    # Check that gradients are initially set to None
    assert all([θi.grad is None for θi in model.parameters()])

    # Get machine learning backend (default: PyTorch)
    pytorch_backend = get_backend()

    # Model input
    f = Function(V)

    # Convert f to torch.Tensor
    f_P = pytorch_backend.to_ml_backend(f)

    # Forward pass
    y_P = model(f_P)

    # Set control
    u_F = Function(V)
    c = Control(u_F)

    # Set reduced functional which expresses the Firedrake operations in terms of the control
    Jhat = ReducedFunctional(poisson_residual(u_F, f_exact, V), c)

    # Construct the HybridOperator that takes a callable representing the Firedrake operations
    G = HybridOperator(Jhat)

    # Compute Poisson residual in Firedrake using HybridOperator: `residual_P` is a torch.Tensor
    residual_P = G(y_P)

    # Compute PyTorch loss
    loss = (residual_P ** 2).sum()

    # -- Check backpropagation API -- #
    loss.backward()

    # Check that gradients were propagated to model parameters
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_hybrid_operator`
    assert all([θi.grad is not None for θi in model.parameters()])

    # -- Check forward operator -- #
    y_F = pytorch_backend.from_ml_backend(y_P, V)
    residual_F = poisson_residual(y_F, f_exact, V)
    residual_P_exact = pytorch_backend.to_ml_backend(residual_F)

    assert (residual_P - residual_P_exact).detach().norm() < 1e-10


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_firedrake_loss_backward(V, f_exact):
    """Add doc """

    # Instantiate model
    model = EncoderDecoder(V.dim())

    # Set double precision
    model.double()

    # Check that gradients are initially set to None
    assert all([θi.grad is None for θi in model.parameters()])

    # Get machine learning backend (default: PyTorch)
    pytorch_backend = get_backend()

    # Model input
    λ = Function(V)

    # Convert f to torch.Tensor
    λ_P = pytorch_backend.to_ml_backend(λ)

    # Forward pass
    f_P = model(λ_P)

    # Set control
    f = Function(V)
    c = Control(f)

    # Set reduced functional which expresses the Firedrake operations in terms of the control
    Jhat = ReducedFunctional(solve_poisson(f, V), c)

    # Construct the HybridOperator that takes a callable representing the Firedrake operations
    G = HybridOperator(Jhat)

    # Solve Poisson problem and compute the loss defined as the L2-norm of the solution
    # -> `loss_P` is a torch.Tensor
    loss_P = G(f_P)

    # -- Check backpropagation API -- #
    loss_P.backward()

    # Check that gradients were propagated to model parameters
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_hybrid_operator`
    assert all([θi.grad is not None for θi in model.parameters()])

    # -- Check forward operator -- #
    f_F = pytorch_backend.from_ml_backend(f_P, V)
    loss_F = solve_poisson(f_F, V)
    loss_P_exact = pytorch_backend.to_ml_backend(loss_F)

    assert (loss_P - loss_P_exact).detach().norm() < 1e-10


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_taylor_hybrid_operator(firedrake_operator, V):
    # Control value
    ω = Function(V)
    # Get Firedrake operator and other operator arguments
    fd_op, args = firedrake_operator
    # Set reduced functional
    Jhat = ReducedFunctional(fd_op(ω, *args), Control(ω))
    # Define the hybrid operator
    G = HybridOperator(Jhat)
    # `gradcheck` is likey to fail if the inputs are not double precision (cf. https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html)
    x_P = torch.rand(V.dim(), dtype=torch.double, requires_grad=True)
    # Taylor test (`eps` is the perturbation)
    torch.autograd.gradcheck(G, x_P, eps=1e-6)

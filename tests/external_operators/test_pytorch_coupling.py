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


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_loss_backward(V, f_exact):

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

    # Construct the HybridOperator that takes a callable representing the Firedrake operations
    G = HybridOperator(poisson_residual, control_space=V)

    # Compute Poisson residual in Firedrake using HybridOperator: `residual_P` is a torch.Tensor
    residual_P = G(y_P, f_exact, V)

    # Compute PyTorch loss
    loss = (residual_P ** 2).sum()

    # Check backpropagation API
    loss.backward()

    # Check that gradients were propagated to model parameters
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_hybrid_operator`
    assert all([θi.grad is not None for θi in model.parameters()])


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
def test_taylor_hybrid_operator(V, f_exact):
    G = HybridOperator(poisson_residual, control_space=V)
    # Make callable for taylor test
    Ghat = lambda x: G(x, f_exact, V)
    # `gradcheck` is likey to fail if the inputs are not double precision (cf. https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html)
    x_P = torch.rand(V.dim(), dtype=torch.double, requires_grad=True)
    # Taylor test
    torch.autograd.gradcheck(Ghat, x_P)

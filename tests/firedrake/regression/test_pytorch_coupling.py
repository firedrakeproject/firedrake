import pytest

from firedrake import *
from pyadjoint.tape import get_working_tape, pause_annotation


try:
    from firedrake.ml.pytorch import *
    import torch
    import torch.nn.functional as torch_func
    from torch.nn import Module, Flatten, Linear

    class EncoderDecoder(Module):
        """Build a simple toy model"""

        def __init__(self, n):
            super(EncoderDecoder, self).__init__()
            self.n = n
            self.m = int(n/2)
            self.flatten = Flatten()
            self.linear_encoder = Linear(self.n, self.m)
            self.linear_decoder = Linear(self.m, self.n)

        def encode(self, x):
            return torch_func.relu(self.linear_encoder(x))

        def decode(self, x):
            return torch_func.relu(self.linear_decoder(x))

        def forward(self, x):
            # [batch_size, n]
            x = self.flatten(x)
            # [batch_size, m]
            hidden = self.encode(x)
            # [batch_size, n]
            return self.decode(hidden)
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


@pytest.fixture(scope="module")
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture(scope="module")
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture
def f_exact(V, mesh):
    x, y = SpatialCoordinate(mesh)
    return Function(V).interpolate(sin(pi * x) * sin(pi * y))


# Set of Firedrake operations that will be composed with PyTorch operations
def poisson_residual(u, f, V):
    """Assemble the residual of a Poisson problem"""
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) + inner(u, v) - inner(f, v)) * dx
    return assemble(F)


# Set of Firedrake operations that will be composed with PyTorch operations
def solve_poisson(f, V):
    """Solve Poisson problem with homogeneous Dirichlet boundary conditions"""
    u = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) + inner(u, v) - inner(f, v)) * dx
    bcs = [DirichletBC(V, Constant(1.0), "on_boundary")]
    # Solve PDE
    solve(F == 0, u, bcs=bcs)
    # Assemble Firedrake loss
    return assemble(u ** 2 * dx)


@pytest.fixture(params=["poisson_residual", "solve_poisson"])
def firedrake_operator(request, f_exact, V):
    # Return firedrake operator and the corresponding non-control arguments
    if request.param == "poisson_residual":
        return poisson_residual, (f_exact, V)
    elif request.param == "solve_poisson":
        return solve_poisson, (V,)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_pytorch_loss_backward(V, f_exact):
    """Test backpropagation through a vector-valued Firedrake operator"""

    from firedrake.adjoint import ReducedFunctional, Control

    # Instantiate model
    model = EncoderDecoder(V.dim())

    # Set double precision
    model.double()

    # Check that gradients are initially set to None
    assert all([pi.grad is None for pi in model.parameters()])

    # Convert f_exact to torch.Tensor
    f_P = to_torch(f_exact)

    # Forward pass
    u_P = model(f_P)

    # Set control
    u = Function(V)
    c = Control(u)

    # Set reduced functional which expresses the Firedrake operations in terms of the control
    Jhat = ReducedFunctional(poisson_residual(u, f_exact, V), c)

    # Construct the torch operator that takes a callable representing the Firedrake operations
    G = fem_operator(Jhat)

    # Compute Poisson residual in Firedrake using the torch operator: `residual_P` is a torch.Tensor
    residual_P = G(u_P)

    # Compute PyTorch loss
    loss = (residual_P ** 2).sum()

    # -- Check backpropagation API -- #
    loss.backward()

    # Check that gradients were propagated to model parameters
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_fem_operator`
    assert all([pi.grad is not None for pi in model.parameters()])

    # -- Check forward operator -- #
    u = from_torch(u_P, V)
    residual = poisson_residual(u, f_exact, V)
    residual_P_exact = to_torch(residual)

    assert (residual_P - residual_P_exact).detach().norm() < 1e-10


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_firedrake_loss_backward(V):
    """Test backpropagation through a scalar-valued Firedrake operator"""

    from firedrake.adjoint import ReducedFunctional, Control

    # Instantiate model
    model = EncoderDecoder(V.dim())

    # Set double precision
    model.double()

    # Check that gradients are initially set to None
    assert all([pi.grad is None for pi in model.parameters()])

    # Model input
    u = Function(V)

    # Convert f to torch.Tensor
    u_P = to_torch(u)

    # Forward pass
    f_P = model(u_P)

    # Set control
    f = Function(V)
    c = Control(f)

    # Set reduced functional which expresses the Firedrake operations in terms of the control
    Jhat = ReducedFunctional(solve_poisson(f, V), c)

    # Construct the torch operator that takes a callable representing the Firedrake operations
    G = fem_operator(Jhat)

    # Solve Poisson problem and compute the loss defined as the L2-norm of the solution
    # -> `loss_P` is a torch.Tensor
    loss_P = G(f_P)

    # -- Check backpropagation API -- #
    loss_P.backward()

    # Check that gradients were propagated to model parameters
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_fem_operator`
    assert all([pi.grad is not None for pi in model.parameters()])

    # -- Check forward operator -- #
    f = from_torch(f_P, V)
    loss = solve_poisson(f, V)
    loss_P_exact = to_torch(loss)

    assert (loss_P - loss_P_exact).detach().norm() < 1e-10


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_taylor_fem_operator(firedrake_operator, V):
    """Taylor test for the torch operator"""

    from firedrake.adjoint import ReducedFunctional, Control

    # Control value
    w = Function(V)
    # Get Firedrake operator and other operator arguments
    fd_op, args = firedrake_operator
    # Set reduced functional
    Jhat = ReducedFunctional(fd_op(w, *args), Control(w))
    # Define the torch operator
    G = fem_operator(Jhat)
    # `gradcheck` is likely to fail if the inputs are not double precision (cf. https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html)
    x_P = torch.rand(V.dim(), dtype=torch.double, requires_grad=True)
    # Taylor test (`eps` is the perturbation)
    assert torch.autograd.gradcheck(G, x_P, eps=1e-6)

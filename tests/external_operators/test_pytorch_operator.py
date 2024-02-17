import pytest

from ufl.algorithms.ad import expand_derivatives

from firedrake import *

try:
    from firedrake.ml.pytorch import *
    import torch
    import torch.nn.functional as F
    import torch.autograd.functional as torch_func
    from torch.nn import Module, Linear

    class EncoderDecoder(Module):
        """Build a simple toy model"""

        def __init__(self, n):
            super(EncoderDecoder, self).__init__()
            self.n1 = n
            self.n2 = int(2*n/3)
            self.n3 = int(n/2)
            # Encoder/decoder layers
            self.encoder_1 = Linear(self.n1, self.n2)
            self.encoder_2 = Linear(self.n2, self.n3)
            self.decoder_1 = Linear(self.n3, self.n2)
            self.decoder_2 = Linear(self.n2, self.n1)

        def encode(self, x):
            return self.encoder_2(F.relu(self.encoder_1(x)))

        def decode(self, x):
            return self.decoder_2(F.relu(self.decoder_1(x)))

        def forward(self, x):
            # x: [batch_size, n]
            encoded = self.encode(x)
            # encoded: [batch_size, n3]
            hidden = F.relu(encoded)
            decoded = self.decode(hidden)
            # decoded: [batch_size, n]
            return F.relu(decoded)

except ImportError:
    # PyTorch is not installed
    pass


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(10, 10)


@pytest.fixture(scope='module')
def V(mesh):
    return FunctionSpace(mesh, "CG", 1)


@pytest.fixture(scope='module')
def u(mesh, V):
    x, y = SpatialCoordinate(mesh)
    return Function(V).interpolate(sin(pi * x) * sin(pi * y))


@pytest.fixture
def nn(model, V):
    # What should we do for inputs_format?
    return ml_operator(model, function_space=V)


@pytest.fixture(params=['linear', 'encoder_decoder'])
def model(request, V):
    n = V.dim()
    if request.param == 'linear':
        return Linear(n, n)
    elif request.param == 'encoder_decoder':
        f = EncoderDecoder(n)
        f.double()
        return f


@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_forward(u, nn):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble NeuralNet operator
    assembled_N = assemble(N)

    # Convert from Firedrake to PyTorch
    x_P = to_torch(u)
    # Forward pass
    y_P = model(x_P)
    # Convert from PyTorch to Firedrake
    y_F = from_torch(y_P, u.function_space())

    # Check
    assert np.allclose(y_F.dat.data_ro, assembled_N.dat.data_ro)


@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_jvp(u, nn):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model
    # Set δu
    V = N.function_space()
    δu = Function(V)
    δu.vector()[:] = np.random.rand(V.dim())

    # Symbolic compute: <∂N/∂u, δu>
    dN = action(derivative(N, u), δu)
    # Assemble
    dN = assemble(dN)

    # Convert from Firedrake to PyTorch
    δu_P = to_torch(δu)
    u_P = to_torch(u)
    # Compute Jacobian-vector product with PyTorch
    _, jvp_exact = torch_func.jvp(lambda x: model(x), u_P, δu_P)

    # Check
    assert np.allclose(dN.dat.data_ro, jvp_exact.numpy())


@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_vjp(u, nn):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model
    # Set δN
    V = N.function_space()
    δN = Cofunction(V.dual())
    δN.vector()[:] = np.random.rand(V.dim())

    # Symbolic compute: <(∂N/∂u)*, δN>
    dNdu = expand_derivatives(derivative(N, u))
    dNdu = action(adjoint(dNdu), δN)
    # Assemble
    dN_adj = assemble(dNdu)
    # TODO: Fix above so that can directly write: dN_adj = assemble(action(adjoint(derivative(N, u)), δN))

    # Convert from Firedrake to PyTorch
    δN_P = to_torch(δN)
    u_P = to_torch(u)
    # Compute vector-Jacobian product with PyTorch
    _, vjp_exact = torch_func.vjp(lambda x: model(x), u_P, δN_P)

    # Check
    assert np.allclose(dN_adj.dat.data_ro, vjp_exact.numpy())


@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_jacobian(u, nn):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble Jacobian of N
    dN = assemble(derivative(N, u))

    # Convert from Firedrake to PyTorch
    u_P = to_torch(u, batched=False)
    # Compute Jacobian with PyTorch
    J = torch_func.jacobian(lambda x: model(x), u_P)

    # Check
    assert np.allclose(dN.petscmat[:, :], J.numpy())


@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_jacobian_adjoint(u, nn):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble Jacobian adjoint of N
    dNdu = expand_derivatives(derivative(N, u))
    dNdu = adjoint(dNdu)
    dN_adj = assemble(dNdu)

    # Convert from Firedrake to PyTorch
    u_P = to_torch(u, batched=False)
    # Compute Jacobian with PyTorch
    J = torch_func.jacobian(lambda x: model(x), u_P)
    # Take Hermitian transpose
    J_adj = J.H

    # Check
    assert np.allclose(dN_adj.petscmat[:, :], J_adj.numpy())


@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_solve(mesh, V):

    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V)
    u = Function(V)
    f = Function(V).interpolate(cos(x)*sin(y))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    # Define the identity model
    n = V.dim()
    model = Linear(n, n)
    model.weight.data = torch.eye(n)
    model.bias.data = torch.zeros(n)

    u2 = Function(V)
    p = ml_operator(model, function_space=V, inputs_format=1)
    tau2 = p(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09

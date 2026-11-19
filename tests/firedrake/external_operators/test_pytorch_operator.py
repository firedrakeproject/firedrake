import pytest
import numpy as np
import ufl

from ufl.algorithms.ad import expand_derivatives
from ufl.algorithms.restructure_base_form import restructure_base_form
from ufl.core.external_operator import ExternalOperator

from firedrake import *
from firedrake.utils import ScalarType


@pytest.fixture
def rg():
    return RandomGenerator(PCG64(seed=1234))


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


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_forward(u, nn):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble NeuralNet operator
    assembled_N = assemble(N)

    assert isinstance(assembled_N, Function)

    # Convert from Firedrake to PyTorch
    x_P = to_torch(u)
    # Forward pass
    y_P = model(x_P)
    # Convert from PyTorch to Firedrake
    y_F = from_torch(y_P, u.function_space())

    # Check
    assert np.allclose(y_F.dat.data_ro, assembled_N.dat.data_ro)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_forward_mixed(V, nn):

    W = V * V
    u = Function(W)
    u1, u2 = u.subfunctions
    x, y = SpatialCoordinate(V.mesh())
    u1.interpolate(sin(pi * x) * sin(pi * y))
    u2.interpolate(sin(2 * pi * x) * sin(2 * pi * y))

    # Set PytorchOperator
    n = W.dim()
    model = Linear(n, n)

    N = ml_operator(model, function_space=W)(u)
    # Get model
    model = N.model

    # Assemble NeuralNet
    assembled_N = assemble(N)

    assert isinstance(assembled_N, Function)

    # Convert from Firedrake to PyTorch
    x_P = to_torch(u)
    # Forward pass
    y_P = model(x_P)
    # Convert from PyTorch to Firedrake
    y_F = from_torch(y_P, u.function_space())

    # Check
    assert np.allclose(y_F.dat.data_ro, assembled_N.dat.data_ro)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_jvp(u, nn, rg):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model
    # Set δu
    V = N.function_space()
    delta_u = rg.uniform(V)

    # Symbolic compute: <∂N/∂u, δu>
    dN = action(derivative(N, u), delta_u)
    # Assemble
    dN = assemble(dN)

    assert isinstance(dN, Function)

    # Convert from Firedrake to PyTorch
    delta_u_P = to_torch(delta_u)
    u_P = to_torch(u)
    # Compute Jacobian-vector product with PyTorch
    _, jvp_exact = torch_func.jvp(lambda x: model(x), u_P, delta_u_P)

    # Check
    assert np.allclose(dN.dat.data_ro, jvp_exact.numpy())


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
def test_vjp(u, nn, rg):
    # Set PytorchOperator
    N = nn(u)
    # Get model
    model = N.model
    # Set δN
    V = N.function_space()
    delta_N = rg.uniform(V.dual())

    # Symbolic compute: <(∂N/∂u)*, δN>
    dNdu = expand_derivatives(derivative(N, u))
    dNdu = action(adjoint(dNdu), delta_N)
    # Assemble
    dN_adj = assemble(dNdu)

    assert isinstance(dN_adj, Cofunction)

    # Convert from Firedrake to PyTorch
    delta_N_P = to_torch(delta_N)
    u_P = to_torch(u)
    # Compute vector-Jacobian product with PyTorch
    _, vjp_exact = torch_func.vjp(lambda x: model(x), u_P, delta_N_P)

    # Check
    assert np.allclose(dN_adj.dat.data_ro, vjp_exact.numpy())


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
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


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
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


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
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

    F2 = inner(grad(u2), grad(w))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09


@pytest.mark.skipcomplex  # grad can be implicitly created only for real scalar outputs but got torch.complex128
@pytest.mark.skiptorch  # Skip if PyTorch is not installed
@pytest.mark.xfail(reason="I think this might be illegal")  # see the equivalent jax test for reasoning
def test_mixed_space_bcs():
    mesh = UnitIntervalMesh(4)
    V = FunctionSpace(mesh, "CG", 1)
    W = V * V

    test = TestFunction(W)
    bcs = [DirichletBC(W.sub(0), Constant(1), 1),
           DirichletBC(W.sub(1), Constant(2), 1)]

    model = Linear(W.dim(), V.dim())
    I = torch.eye(V.dim())
    model.weight.data = torch.cat([I, I], dim=1)
    model.bias.data = torch.zeros(V.dim())
    model.eval()

    p1 = ml_operator(model, function_space=V, inputs_format=1)
    p2 = sum

    results = []
    for p in (p1, p2):
        w = Function(W)
        F = inner(w, test)*dx + inner(p(w), sum(test))*dx
        solve(F == 0, w, bcs=bcs)
        results.append(np.ravel(w.dat.data))

    result, expected = results
    assert np.allclose(result, expected)


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("complex_parameters", [False, True])
def test_ml_parameters_storage_and_snapshots(complex_parameters: bool) -> None:
    """Keep model and optimiser references valid when restoring parameter values."""
    dtype = torch.complex128 if complex_parameters else torch.float64
    model = Linear(2, 1).to(dtype=dtype)
    parameters = MLParameters(model.parameters())
    tensors = tuple(model.parameters())
    pointers = tuple(t.data_ptr() for t in tensors)
    assert all(t is p for t, p in zip(parameters.tensors, tensors))
    assert parameters.ufl_shape == (3,)
    assert parameters.ufl_domains() == ()

    saved = parameters.copy()
    assert all(not t.requires_grad for t in saved.tensors)
    assert all(t.data_ptr() != p.data_ptr() for t, p in zip(saved.tensors, tensors))
    optimiser = torch.optim.SGD(parameters.tensors, lr=0.1)
    model(torch.ones(2, dtype=dtype)).real.sum().backward()
    optimiser.step()
    for value, original in zip(parameters.tensors, saved.tensors):
        torch.testing.assert_close(value, original - 0.1)

    assert parameters.assign(saved) is parameters
    assert tuple(t.data_ptr() for t in parameters.tensors) == pointers
    assert all(t is p for t, p in zip(parameters.tensors, model.parameters()))
    for value, original in zip(parameters.tensors, saved.tensors):
        torch.testing.assert_close(value, original)


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("interpolated", [False, True])
@pytest.mark.parametrize("size", [2, 10000])
def test_ml_parameters_symbolic_derivative(size: int, interpolated: bool) -> None:
    """Preserve the parameter space in the pullback through an ML input."""
    mesh = UnitIntervalMesh(2)
    V = FunctionSpace(mesh, "CG", 1)
    u = Function(V)
    parameters = MLParameters([torch.nn.Parameter(torch.ones(size))])
    operand = interpolate(u, V) if interpolated else u
    N = ExternalOperator(operand, parameters, function_space=V)
    linearization = expand_derivatives(derivative(N, parameters))
    assert linearization.derivatives == (0, 1)
    assert linearization.arguments()[-1].ufl_function_space() == parameters.ufl_function_space()
    form = inner(N, TestFunction(V)) * dx
    assert form.domain_numbering() == {mesh: 0}
    assert isinstance(form.signature(), str)
    pullback = restructure_base_form(action(adjoint(derivative(form, parameters)), u))
    argument, = pullback.arguments()
    assert argument.number() == 0
    assert argument.ufl_function_space() == parameters.ufl_function_space()
    assert parameters.ufl_shape == (size,)
    assert parameters.ufl_domains() == ()

    saved = parameters.copy()
    direction = linearization.arguments()[-1]
    assert expand_derivatives(derivative(N, parameters, direction)) == linearization
    directional = restructure_base_form(action(linearization, saved))
    assert directional.argument_slots()[-1] is saved
    replaced = ufl.replace(N, {parameters: saved})
    assert replaced.ufl_operands[-1] is saved
    assert type(replaced.ufl_operands[-1]) is MLParameters
    assert saved.ufl_function_space() == parameters.ufl_function_space()
    assert saved != parameters


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
def test_ml_parameters_on_two_meshes() -> None:
    """Differentiate residuals on two meshes with respect to the same weights."""
    dtype = torch.from_numpy(np.empty((), dtype=ScalarType)).dtype
    model = Linear(1, 1, dtype=dtype)
    theta = MLParameters(model.parameters())
    spaces = (FunctionSpace(UnitIntervalMesh(2), "CG", 1),
              FunctionSpace(UnitSquareMesh(2, 2), "DG", 0))
    states = tuple(Function(V) for V in spaces)
    contributions = []
    for u in states:
        V = u.function_space()
        N = ml_operator(model, V, inputs_format=1)
        residual = inner(N(u, theta), TestFunction(V)) * dx
        assert residual.domain_numbering() == {V.mesh(): 0}
        contributions.append(restructure_base_form(action(adjoint(derivative(residual, theta)), u)))

    total = sum(contributions)
    direction, = total.arguments()
    assert direction.ufl_function_space() == theta.ufl_function_space()
    assert set(total.coefficients()) == {theta, *states}


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("mismatch", ["shape", "dtype", "count"])
def test_ml_parameters_assign_mismatch(mismatch: str) -> None:
    """Reject incompatible snapshots before modifying any parameter."""
    parameters = MLParameters([torch.ones(2), torch.ones(3)])
    replacement = [torch.full((2,), 9.0), torch.zeros(3)]
    if mismatch == "shape":
        replacement[1] = torch.zeros(4)
    elif mismatch == "dtype":
        replacement[1] = replacement[1].double()
    else:
        replacement.pop()
    with pytest.raises(ValueError, match="matching shapes, dtypes and devices"):
        parameters.assign(MLParameters(replacement))
    for tensor in parameters.tensors:
        torch.testing.assert_close(tensor, torch.ones_like(tensor))


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
@pytest.mark.parametrize("invalid", ["empty", "non_tensor", "integer"])
def test_ml_parameters_invalid_values(invalid: str) -> None:
    """Require a nonempty vector of differentiable scalar values."""
    if invalid == "empty":
        values, error = [], ValueError
    elif invalid == "non_tensor":
        values, error = [1.0], TypeError
    else:
        values, error = [torch.ones(2, dtype=torch.int64)], TypeError
    with pytest.raises(error):
        MLParameters(values)


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
def test_ml_parameters_training_step() -> None:
    """Two unit-volume meshes contribute (3, 2) to the same affine model."""
    dtype = torch.from_numpy(np.empty((), dtype=ScalarType)).dtype
    model = Linear(1, 1, dtype=dtype)
    with torch.no_grad():
        model.weight.fill_(2)
        model.bias.fill_(1)
    theta = MLParameters(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    spaces = (FunctionSpace(UnitIntervalMesh(4), "CG", 1),
              FunctionSpace(UnitSquareMesh(2, 2), "DG", 0))
    losses = []
    for i, V in enumerate(spaces, start=1):
        u = Function(V).assign(i)
        N = ml_operator(model, V, inputs_format=1)
        value = N(theta, u) if i == 1 else N(u, theta)
        losses.append(inner(1, value) * dx)
        jac = derivative(value, u)
        tangent = Function(V).assign(1)
        assert np.allclose(assemble(action(jac, tangent)).dat.data_ro, 2)
        matrix = assemble(jac)
        assert np.allclose(assemble(action(matrix, tangent)).dat.data_ro, 2)
        seed = assemble(inner(1, TestFunction(V)) * dx)
        state_gradient = assemble(action(adjoint(jac), seed))
        assert np.allclose(state_gradient.dat.data_ro, 2 * seed.dat.data_ro)

    def loss_value():
        return sum(assemble(loss) for loss in losses).real

    assert np.isclose(loss_value(), 8)
    derivative_sum = sum(derivative(loss, theta) for loss in losses)
    for _ in range(2):
        gradient = assemble(derivative_sum)
        assert gradient.parameters is theta
        assert gradient.ufl_function_space() == theta.ufl_function_space().dual()
        for g, expected in zip(gradient.tensors, (3, 2), strict=True):
            torch.testing.assert_close(g, torch.full_like(g, expected))
        assert all(p.grad is None for p in model.parameters())

    # Check the directional derivative in the (1, -1) direction.
    saved = theta.copy()
    with torch.no_grad():
        model.weight.add_(1e-5)
        model.bias.sub_(1e-5)
    plus = loss_value()
    theta.assign(saved)
    with torch.no_grad():
        model.weight.sub_(1e-5)
        model.bias.add_(1e-5)
    minus = loss_value()
    theta.assign(saved)
    assert np.isclose((plus - minus) / 2e-5, 1)
    torch.autograd.backward(theta.tensors, gradient.tensors)
    optimizer.step()
    assert np.isclose(loss_value(), 8 - 0.1 * (3**2 + 2**2))


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
def test_ml_parameters_binding() -> None:
    """A handle can identify only live parameters of this model."""
    model = Linear(2, 1)
    V = FunctionSpace(UnitIntervalMesh(2), "CG", 1)
    N = ml_operator(model, V)
    u = Function(V)
    theta = MLParameters(model.parameters())
    with pytest.raises(ValueError, match="at most one"):
        N(theta, u, theta)
    with pytest.raises(ValueError, match="distinct parameters"):
        N(u, MLParameters(Linear(2, 1).parameters()))


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
def test_ml_parameters_multiple_fields() -> None:
    """Select the derivative slot independently of the weight operand's position."""
    dtype = torch.from_numpy(np.empty((), dtype=ScalarType)).dtype
    model = Linear(2, 1, dtype=dtype)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2, 3]], dtype=dtype))
        model.bias.fill_(1)
    theta = MLParameters(model.parameters())
    V = FunctionSpace(UnitIntervalMesh(4), "CG", 1)
    u, v = Function(V).assign(4), Function(V).assign(5)
    op = ml_operator(model, V, inputs_format=1)(u, theta, v)
    assert np.allclose(assemble(op).dat.data_ro, 24)

    du = Function(V).assign(1)
    seed = assemble(inner(1, TestFunction(V)) * dx)
    for field, factor in ((u, 2), (v, 3)):
        jac = derivative(op, field)
        assert np.allclose(assemble(action(jac, du)).dat.data_ro, factor)
        assert np.allclose(assemble(action(adjoint(jac), seed)).dat.data_ro, factor * seed.dat.data_ro)

    direction = MLParameters(torch.ones_like(t) for t in theta.tensors)
    jac = derivative(op, theta)
    assert np.allclose(assemble(action(jac, direction)).dat.data_ro, 10)
    gradient = assemble(action(adjoint(jac), seed))
    torch.testing.assert_close(gradient.tensors[0], torch.tensor([[4, 5]], dtype=dtype))
    torch.testing.assert_close(gradient.tensors[1], torch.ones(1, dtype=dtype))


@pytest.mark.skiptorch
@pytest.mark.parallel(nprocs=[1, 2])
def test_ml_parameters_global_model() -> None:
    """A global network couples DoFs across ranks and returns a replicated gradient."""
    class GlobalSum(Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

        def forward(self, x):
            return (self.scale * x.sum()).expand_as(x)

    model = GlobalSum()
    V = FunctionSpace(UnitIntervalMesh(6), "CG", 1)
    u = Function(V).assign(1)
    theta = MLParameters(model.parameters())
    op = ml_operator(model, V)(u, theta)
    assert np.allclose(assemble(op).dat.data_ro, 2 * V.dim())
    jac = derivative(op, u)
    assert np.allclose(assemble(action(assemble(jac), u)).dat.data_ro, 2 * V.dim())
    seed = assemble(inner(1, TestFunction(V)) * dx)
    assert np.allclose(assemble(action(adjoint(jac), seed)).dat.data_ro, 2)
    gradient = assemble(derivative(inner(1, op) * dx, theta))
    assert gradient.tensors[0].item() == pytest.approx(V.dim())

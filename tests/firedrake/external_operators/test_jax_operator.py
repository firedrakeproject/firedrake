import pytest
import numpy as np

from ufl.algorithms.ad import expand_derivatives

from firedrake import *

try:
    from firedrake.ml.jax import *
    import jax
    import jax.numpy as jnp

    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)

    # -- Construct standard models using native JAX -- #

    class Linear():
        """Linear layer: y = Wx + b"""
        def __init__(self, n):
            # Randomly initialise weights and biases
            w_key, b_key = jax.random.split(key)
            self.weight = jax.random.normal(w_key, (n, n))
            self.bias = jax.random.normal(b_key, (n,))

        def __call__(self, x):
            return jnp.dot(self.weight, x) + self.bias

    class MLP():
        """Simple multi-layer perceptron (MLP) model with ReLU activation."""
        def __init__(self, n):
            # Define layer sizes
            sizes = [n, n // 2, n // 2, n]
            # Initialize all layers for a fully-connected neural network with 3 layers
            nlayers = len(sizes) - 1
            keys = jax.random.split(key, nlayers)
            params = []
            for i, k in enumerate(keys):
                w_key, b_key = jax.random.split(k)
                W = jax.random.normal(w_key, (sizes[i+1], sizes[i]))
                b = jax.random.normal(b_key, (sizes[i+1],))
                params.append((W, b))
            self.params = params

        def __call__(self, x):
            activations = x
            for W, b in self.params:
                outputs = jnp.dot(W, activations) + b
                activations = jax.nn.relu(outputs)
            return activations
except ImportError:
    # JAX is not installed
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


@pytest.fixture(params=['jax_linear', 'jax_mlp'])
def model(request, V):
    n = V.dim()
    if request.param == 'jax_linear':
        return Linear(n)
    elif request.param == 'jax_mlp':
        return MLP(n)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_forward(u, nn):
    # Set JaxOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble NeuralNet operator
    assembled_N = assemble(N)

    # Convert from Firedrake to JAX
    x_P = to_jax(u)
    # Forward pass
    y_P = model(x_P)
    # Convert from JAX to Firedrake
    y_F = from_jax(y_P, u.function_space())

    # Check
    assert np.allclose(y_F.dat.data_ro, assembled_N.dat.data_ro)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_jvp(u, nn):
    # Set JaxOperator
    N = nn(u)
    # Get model
    model = N.model
    # Set δu
    V = N.function_space()
    delta_u = Function(V)
    delta_u.vector()[:] = np.random.rand(V.dim())

    # Symbolic compute: <∂N/∂u, δu>
    dN = action(derivative(N, u), delta_u)
    # Assemble
    dN = assemble(dN)

    # Convert from Firedrake to JAX
    delta_u_P = to_jax(delta_u)
    u_P = to_jax(u)
    # Compute Jacobian-vector product with JAX
    _, jvp_exact = jax.jvp(lambda x: model(x), (u_P,), (delta_u_P,))

    # Check
    assert np.allclose(dN.dat.data_ro, jvp_exact)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_vjp(u, nn):
    # Set JaxOperator
    N = nn(u)
    # Get model
    model = N.model
    # Set δN
    V = N.function_space()
    delta_N = Cofunction(V.dual())
    delta_N.vector()[:] = np.random.rand(V.dim())

    # Symbolic compute: <(∂N/∂u)*, δN>
    dNdu = expand_derivatives(derivative(N, u))
    dNdu = action(adjoint(dNdu), delta_N)
    # Assemble
    dN_adj = assemble(dNdu)

    # Convert from Firedrake to JAX
    delta_N_P = to_jax(delta_N)
    u_P = to_jax(u)
    # Compute vector-Jacobian product with JAX
    _, vjp_func = jax.vjp(lambda x: model(x), u_P)
    vjp_exact, = vjp_func(delta_N_P)

    # Check
    assert np.allclose(dN_adj.dat.data_ro, vjp_exact)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_jacobian(u, nn):
    # Set JaxOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble Jacobian of N
    dN = assemble(derivative(N, u))

    # Convert from Firedrake to JAX
    u_P = to_jax(u, batched=False)
    # Compute Jacobian with JAX
    J = jax.jacobian(lambda x: model(x))(u_P)

    # Check
    assert np.allclose(dN.petscmat[:, :], J)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_jacobian_adjoint(u, nn):
    # Set JaxOperator
    N = nn(u)
    # Get model
    model = N.model

    # Assemble Jacobian adjoint of N
    dNdu = expand_derivatives(derivative(N, u))
    dNdu = adjoint(dNdu)
    dN_adj = assemble(dNdu)

    # Convert from Firedrake to JAX
    u_P = to_jax(u, batched=False)
    # Compute Jacobian with JAX
    J = jax.jacobian(lambda x: model(x))(u_P)
    # Take Hermitian transpose
    J_adj = J.T.conj()

    # Check
    assert np.allclose(dN_adj.petscmat[:, :], J_adj)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_solve(mesh, V):

    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V)
    u = Function(V)
    f = Function(V).interpolate(cos(x)*sin(y))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    # Define the identity model
    n = V.dim()
    model = Linear(n)
    model.weight = jnp.eye(n)
    model.bias = jnp.zeros(n)

    u2 = Function(V)
    p = ml_operator(model, function_space=V, inputs_format=1)
    tau2 = p(u2)

    F2 = inner(grad(u2), grad(w))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09

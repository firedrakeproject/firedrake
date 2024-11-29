import pytest

from firedrake import *
from pyadjoint.tape import get_working_tape, pause_annotation


try:
    from firedrake.ml.jax import *
    import jax
    import jax.numpy as jnp
    from jax.test_util import check_grads

    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(0)

    # -- Construct standard MLP model using native JAX -- #

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


# Set of Firedrake operations that will be composed with JAX operations
def poisson_residual(u, f, V):
    """Assemble the residual of a Poisson problem"""
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) + inner(u, v) - inner(f, v)) * dx
    return assemble(F)


# Set of Firedrake operations that will be composed with JAX operations
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
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_jax_loss_backward(V, f_exact):
    """Test backpropagation through a vector-valued Firedrake operator"""

    from firedrake.adjoint import ReducedFunctional, Control

    # Instantiate model
    model = MLP(V.dim())

    # Convert f_exact to jax.Array
    f_P = to_jax(f_exact)

    # Forward pass
    u_P = model(f_P)

    # Set control
    u = Function(V)
    c = Control(u)

    # Set reduced functional which expresses the Firedrake operations in terms of the control
    Jhat = ReducedFunctional(poisson_residual(u, f_exact, V), c)

    # Construct the jax operator that takes a callable representing the Firedrake operations
    G = fem_operator(Jhat)

    # Compute Poisson residual in Firedrake using the jax operator: `residual_P` is a jax.Array
    residual_P = G(u_P)

    # Compute JAX loss
    loss = lambda x: (x ** 2).sum()

    # -- Check backpropagation API -- #
    grad_loss = jax.grad(loss)(residual_P)

    # Check that gradients were calculated.
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_fem_operator`
    assert grad_loss is not None
    assert grad_loss.shape == residual_P.shape

    # -- Check forward operator -- #
    u = from_jax(u_P, V)
    residual = poisson_residual(u, f_exact, V)
    residual_P_exact = to_jax(residual)

    assert np.allclose(residual_P, residual_P_exact)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_firedrake_loss_backward(V):
    """Test backpropagation through a scalar-valued Firedrake operator"""

    from firedrake.adjoint import ReducedFunctional, Control

    # Instantiate model
    model = MLP(V.dim())

    # Model input
    u = Function(V)

    # Convert f to jax.Array
    u_P = to_jax(u)

    # Forward pass
    f_P = model(u_P)

    # Set control
    f = Function(V)
    c = Control(f)

    # Set reduced functional which expresses the Firedrake operations in terms of the control
    Jhat = ReducedFunctional(solve_poisson(f, V), c)

    # Construct the jax operator that takes a callable representing the Firedrake operations
    G = fem_operator(Jhat)

    # Solve Poisson problem and compute the loss defined as the L2-norm of the solution
    # -> `loss_P` is a jax.Array
    loss_P = G(f_P)

    # -- Check backpropagation API -- #
    grad_loss = jax.grad(G)(f_P)

    # Check that gradients were calculated.
    # This test doesn't check the correctness of these gradients
    # -> This is checked in `test_taylor_fem_operator`
    assert grad_loss is not None
    assert grad_loss.shape == f_P.shape

    # -- Check forward operator -- #
    f = from_jax(f_P, V)
    loss = solve_poisson(f, V)
    loss_P_exact = to_jax(loss)

    assert np.allclose(loss_P, loss_P_exact)


@pytest.mark.skipcomplex  # Taping for complex-valued 0-forms not yet done
@pytest.mark.skipjax  # Skip if JAX is not installed
def test_taylor_fem_operator(firedrake_operator, V):
    """Taylor test for the jax operator"""

    from firedrake.adjoint import ReducedFunctional, Control

    # Control value
    w = Function(V)
    # Get Firedrake operator and other operator arguments
    fd_op, args = firedrake_operator
    # Set reduced functional
    Jhat = ReducedFunctional(fd_op(w, *args), Control(w))
    # Define the jax operator
    G = fem_operator(Jhat)
    # `check_grads` is likely to fail if the inputs are not double precision
    x_P = jax.random.normal(key, (V.dim(),)).astype(jnp.float64)
    # Taylor test for first-order derivative using reverse-move AD (`eps` is the perturbation)
    check_grads(G, (x_P,), order=1, modes=('rev',), eps=1e-06, atol=1e-05, rtol=0.001)

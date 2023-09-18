import pytest
from firedrake import *
try:
    import torch
    from firedrake.ml.neural_network_operators import neuralnet
except ImportError:
    raise ImportError("Try: pip install torch")

import torch.nn.functional as F


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


@pytest.fixture(scope='module')
def model():
    return torch.nn.Linear(1, 1)


@pytest.fixture(scope='module')
def model_Identity():
    # Make an Identity model mapping model_Identity: x |--> x (for sake of testing)
    Id = torch.nn.Linear(1, 1)
    Id.weight.data[:] = 1
    Id.bias.data[:] = 0
    return Id


def test_PyTorch_operator_model_attribute(mesh, model):
    V = FunctionSpace(mesh, "CG", 1)

    x, y = SpatialCoordinate(mesh)

    dtype = torch.float64
    x_target = torch.tensor([[0.5]], dtype=dtype)
    y_target = torch.tensor([[2.]], dtype=dtype)

    c1 = float(4.)
    u = Function(V).assign(0.5)

    def get_batch(batch_size=32):
        # Builds a batch i.e. (x, f(x)) pair.
        random = torch.randn(batch_size, dtype=dtype)
        d = random[0]
        d = torch.unsqueeze(d, 0)
        for i in range(1, batch_size):
            f = random[i]
            f = torch.unsqueeze(f, 0)
            d = torch.cat((d, f), 0)
        x = d
        # Approximated function
        y = c1*random[0]
        y = torch.unsqueeze(y, 0)
        for i in range(1, batch_size):
            f = c1*random[i]
            f = torch.unsqueeze(f, 0)
            y = torch.cat((y, f), 0)

        return x, y

    # Define model
    fc = model
    nP = neuralnet(fc, function_space=V, inputs_format=1)
    nP2 = nP(u)

    assert nP2.framework == 'PyTorch'
    assert fc == nP2.model

    for batch_idx in range(500):
        # Get data
        batch_x, batch_y = get_batch()
        batch_x = batch_x.unsqueeze(1)
        batch_y = batch_y.unsqueeze(1)

        # Reset gradients
        nP2.model.zero_grad()

        # Forward pass
        output = F.smooth_l1_loss(nP2.model(batch_x), batch_y)
        loss = output.item()

        # Backward pass
        output.backward()

        # Apply gradients
        for param in nP2.model.parameters():
            param.data.add_(-0.1 * param.grad.data)

        # Stop criterion
        if loss < 1e-4:
            break

    print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
    print("\n x_target : ", x_target, "\n y_target : ", y_target, "\n learning output : ", nP2.model(x_target))

    sol = Function(V)
    sol.dat.data[:] = fc(x_target).detach().numpy()
    error = ((nP2-sol)**2)*dx
    assert assemble(error) < 1.0e-9


"""
TODO: ncontrols is a deprecated feature
def test_pointwise_neuralnet_PyTorch_control(mesh, model):
    V = FunctionSpace(mesh, "CG", 1)

    x, y = SpatialCoordinate(mesh)

    u = Function(V)
    g = Function(V)

    # Define model
    fc = model
    #  By default ncontrols = 1
    nP = neuralnet(fc, function_space=V, ncontrols=1)
    # The ncontrols first operands are taken as controls
    nP2 = nP(g, u)

    assert nP2.framework == 'PyTorch'
    assert fc == nP2.model

    from ufl.algorithms.apply_derivatives import apply_derivatives
    dnp2_du = apply_derivatives(diff(nP2, u))
    assemble(dnp2_du*dx)

    dnp2_dg = apply_derivatives(diff(nP2, g))
    assemble(dnp2_dg*dx)
"""


def test_scalar_check_equality(mesh, model_Identity):

    V1 = FunctionSpace(mesh, "CG", 1)
    x, y = SpatialCoordinate(mesh)

    w = TestFunction(V1)
    u = Function(V1)
    f = Function(V1).interpolate(cos(x)*sin(y))

    F = inner(grad(w), grad(u))*dx + inner(u, w)*dx - inner(f, w)*dx
    solve(F == 0, u)

    u2 = Function(V1)
    p = neuralnet(model_Identity, function_space=V1, inputs_format=1)
    tau2 = p(u2)

    F2 = inner(grad(w), grad(u2))*dx + inner(tau2, w)*dx - inner(f, w)*dx
    solve(F2 == 0, u2)

    err_point_expr = assemble((u-u2)**2*dx)/assemble(u**2*dx)
    assert err_point_expr < 1.0e-09

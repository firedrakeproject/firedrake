import pytest
from firedrake import *
import torch
import torch.nn.functional as F


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_pointwise_neuralnet_PyTorch(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    dtype = torch.float64
    x_target = torch.tensor([[0.5]], dtype=dtype)
    y_target = torch.tensor([[2.]], dtype=dtype)

    c1 = float(4.)
    u = Function(V).assign(0.5)

    def get_batch(batch_size=32):
        #Builds a batch i.e. (x, f(x)) pair.
        random = torch.randn(batch_size)
        d = torch.tensor(random[0], dtype=dtype)
        d = torch.unsqueeze(d, 0)
        for i in range(1, batch_size):
            f = torch.tensor(random[i], dtype=dtype)
            f = torch.unsqueeze(f, 0)
            d = torch.cat((d,f), 0)
        x = d
        # Approximated function
        y = torch.tensor(c1*random[0], dtype=dtype)
        y = torch.unsqueeze(y, 0)
        for i in range(1, batch_size):
            f = torch.tensor(c1*random[i], dtype=dtype)
            f = torch.unsqueeze(f, 0)
            y = torch.cat((y,f), 0)

        return x, y


    # Define model
    fc = torch.nn.Linear(1, 1)
    nP = neuralnet(fc, function_space=V)
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
    print("\n x_target : ",x_target," \n y_target : ",y_target,"\n learning output : ",nP2.model(x_target))

    sol = Function(V)
    sol.dat.data[:] = fc(x_target).detach().numpy()
    error = ((nP2-sol)**2)*dx
    assert assemble(error) < 1.0e-9

def test_pointwise_neuralnet_PyTorch_control(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    u = Function(V)
    g = Function(V) # g will be the control

    # Define model
    fc = torch.nn.Linear(1, 1)
    nP = neuralnet(fc, function_space=V, ncontrols=1)  #  By default ncontrols = 1
    nP2 = nP(g, u) # The ncontrols first operands are taken as controls

    assert nP2.framework == 'PyTorch'
    assert fc == nP2.model

    from ufl.algorithms.apply_derivatives import apply_derivatives
    dnp2_du = apply_derivatives(diff(nP2,u))
    assemble(dnp2_du*dx)
    print("\n FIRST ASSEMBLE DONE")
    #import ipdb; ipdb.set_trace()
    
    dnp2_dg = apply_derivatives(diff(nP2,g))
    assemble(dnp2_dg*dx)

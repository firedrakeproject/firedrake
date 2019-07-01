import pytest
from firedrake import *
import numpy as np
import torch
import torch.nn.functional as F


@pytest.fixture(scope='module')
def mesh():
    return UnitSquareMesh(5, 5)


def test_pointwise_operator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: x*y)
    p2 = p(u, v, function_space=P)
    a2 = p2*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0)
    assert p2.ufl_shape == ()
    assert p2.expr(u, v) == u*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-3  # Not evaluate on the same space whence the lack of precision
    u2 = Function(V)
    u3 = Function(V)
    g = Function(V).interpolate(cos(x))
    v = TestFunction(V)

    f = Function(V).interpolate(cos(x)*sin(y))
    p = point_expr(lambda x:x**2+1)
    p2 = p(g, function_space=V)
    #f = Function(V).interpolate(cos(x)*sin(y))

    F = (dot(grad(p2*u),grad(v)) + u*v)*dx - f*v*dx
    solve(F == 0, u)

    F2 = (dot(grad((g**2+1)*u2),grad(v)) + u2*v)*dx - f*v*dx
    solve(F2 == 0,u2)
    
    p = point_expr(lambda x:x**2+1)
    p2 = p(g, function_space=V)
    f = Function(V).interpolate(cos(x)*sin(y))
    F = (dot(grad(p2*u3),grad(v)) + u3*v)*dx - f*v*dx
    problem = NonlinearVariationalProblem(F, u3)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()

    a1 = assemble(u*dx)
    a2 = assemble(u2*dx)
    a3 = assemble(u3*dx)
    err = (a1-a2)**2 + (a2-a3)**2
    assert err < 1.0e-9


def test_pointwise_solver(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    uexact = Function(P).assign(1)

    m = uexact
    a1 = m*dx

    a = Function(V).assign(0)
    b = Function(V).assign(1)

    # Conflict with ufl if we use directly cos()
    p = point_solve(lambda x, y, m1, m2: (1-m1)*(1-x)**2 + 100*m2*(y-x**2)**2)
    p2 = p(b, a, b, function_space=P)  # Rosenbrock function for (m1,m2) = (0,1), the global minimum is reached in (1,1)
    a2 = p2*dx

    assert p2.ufl_operands == (b, a, b)
    assert p2._ufl_function_space == P
    assert p2.derivatives == (0, 0, 0)
    assert p2.ufl_shape == ()

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-7
    assemble(p2*dx)

    u = Function(V)
    u2 = Function(V)
    u3 = Function(V)
    v = TestFunction(V)
    g = Function(V).assign(1.)

    f = Function(V).interpolate(cos(x)*sin(y))
    p = point_solve(lambda x,y:x**3-y)
    p2 = p(g, function_space=V)
    #f = Function(V).interpolate(cos(x)*sin(y))

    
    F = (dot(grad(p2*u),grad(v)) + u*v)*dx - f*v*dx
    solve(F == 0, u)

    F = (dot(grad((g**2)*u2),grad(v)) + u2*v)*dx - f*v*dx
    solve(F == 0, u2)

    F = (dot(grad(p2*u3),grad(v)) + u3*v)*dx - f*v*dx
    problem = NonlinearVariationalProblem(F, u3)
    solver = NonlinearVariationalSolver(problem)
    solver.solve()

    a1 = assemble(u*dx)
    a2 = assemble(u2*dx)
    a3 = assemble(u3*dx)
    err = (a1-a2)**2 + (a2-a3)**2
    assert err < 1.0e-9



def test_compute_derivatives(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    v = Function(V).interpolate(sin(x))
    u = Function(V).interpolate(cos(x))

    m = u*v
    a1 = m*dx

    p = point_expr(lambda x, y: 0.5*x**2*y)
    p2 = p(u, v, function_space=P, derivatives=(1, 0))
    a2 = p2*dx

    assert p2.ufl_operands[0] == u
    assert p2.ufl_operands[1] == v
    assert p2._ufl_function_space == P
    assert p2.derivatives == (1, 0)
    assert p2.ufl_shape == ()
    assert p2.expr(u, v) == 0.5*u**2*v

    assemble_a1 = assemble(a1)
    assemble_a2 = assemble(a2)

    assert abs(assemble_a1 - assemble_a2) < 1.0e-3  # Not evaluate on the same space whence the lack of precision

    a = Function(V).assign(0)
    b = Function(V).assign(1)

    x0 = Function(P).assign(1.1)
    p = point_solve(lambda x, y, m1, m2: x - y**2 + m1*m2, solver_params={'x0': x0})
    p2 = p(b, a, a, function_space=P, derivatives=(1, 0, 0))
    a3 = p2*dx

    a4 = 2*b*dx  # dp2/db

    assemble_a3 = assemble(a3)
    assemble_a4 = assemble(a4)

    assert abs(assemble_a3 - assemble_a4) < 1.0e-7


def test_pointwise_neuralnet_PyTorch(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    nP = neuralnet('PyTorch')
    #nT = neuralnet('TensorFlow')
    #nK = neuralnet('Keras')

    x_target = float(0.5)*torch.ones(1)#(V.node_count)
    y_target = float(2.)*torch.ones(1)#(V.node_count)

    c1 = float(4.)
    u = Function(V).assign(0.5)
    #w = Function(V).interpolate(x)

    def get_batch(batch_size=32):
        #Builds a batch i.e. (x, f(x)) pair.
        random = torch.randn(batch_size)
        ftemp = float(random[0])*torch.ones(1)#(V.node_count)
        d = ftemp.type(torch.FloatTensor)
        d = torch.unsqueeze(d, 0)
        for i in range(1, batch_size):
            f1 = float(random[i])*torch.ones(1)#(V.node_count)
            tf = f1.type(torch.FloatTensor)
            f2 = torch.unsqueeze(tf, 0)
            d = torch.cat((d,f2), 0)
        x = d
        # Approximated function
        ftemp = (c1*float(random[0]))*torch.ones(1)#(V.node_count)
        y = ftemp.type(torch.FloatTensor)
        y = torch.unsqueeze(y, 0)
        for i in range(1, batch_size):
            f1 = (c1*float(random[i]))*torch.ones(1)#(V.node_count)
            tf = f1.type(torch.FloatTensor)
            f2 = torch.unsqueeze(tf, 0)
            y = torch.cat((y,f2), 0)
        return x, y


    # Define model
    fc = torch.nn.Linear(1, 1)#(V.node_count, V.node_count)
    nP2 = nP(u, function_space=V, model=fc)

    assert nP2.framework == 'PyTorch'
    assert fc == nP2.model

    for batch_idx in range(500):
        # Get data
        batch_x, batch_y = get_batch()

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

    nP = neuralnet('PyTorch')

    u = Function(V)
    g = Function(V, name='Control')

    # Define model
    fc = torch.nn.Linear(1, 1)
    nP2 = nP(u, g, function_space=V, model=fc)

    assert nP2.framework == 'PyTorch'
    assert fc == nP2.model

    from ufl.algorithms.apply_derivatives import apply_derivatives
    dnp2_du = apply_derivatives(diff(nP2,u))

    try:
        dnp2_dg = apply_derivatives(diff(nP2,g))
        test_diff_control = 0
    except:
        test_diff_control = 1
    assert test_diff_control

def test_derivation_wrt_pointwiseoperator(mesh):
    V = FunctionSpace(mesh, "CG", 1)
    P = FunctionSpace(mesh, "DG", 0)

    x, y = SpatialCoordinate(mesh)

    u = Function(V)
    g = Function(V)
    v = TestFunction(V)
    u_hat = Function(V)
    
    p = point_expr(lambda x, y: x*y)
    p2 = p(u, g, function_space=P)
    
    from ufl.algorithms.apply_derivatives import apply_derivatives

    l = sin(p2**2)*v
    dl_dp2 = p2*2.*cos(p2**2)*v
    dl = diff(l, p2)
    assert apply_derivatives(dl) == dl_dp2

    L = p2*u*dx
    dL_dp2 = u*u_hat*dx
    Gateaux_dL = derivative(L,p2, u_hat)
    assert apply_derivatives(Gateaux_dL) == dL_dp2

from firedrake import *
import numpy as np


def poisson_solver(ref_level, degree):
    mesh = UnitDiskMesh(refinement_level=ref_level, degree=degree)
    V = FunctionSpace(mesh, "CG", 2)

    x, y = SpatialCoordinate(mesh)
    u_ = (1 - x**2 - y**2)*(sin(x+y))
    f = -div(grad(u_))

    u = Function(V)
    v = TestFunction(V)
    F = (inner(grad(u), grad(v)) - f*v)*dx
    bcs = DirichletBC(V, 0, "on_boundary")
    solve(F == 0, u, bcs=bcs)
    err = errornorm(u_, u, norm_type='H1')

    # compute mesh width
    mesh0 = UnitDiskMesh(refinement_level=ref_level, degree=1)
    H = FunctionSpace(mesh0, "DG", 0)
    h = Function(H)
    h.interpolate(CellSize(mesh0))
    return [max(h.dat.data_ro), err]


def conv_rate(degree):
    err = []
    h = []
    for ref_level in range(0, 4):
        h_, err_ = poisson_solver(ref_level, degree)
        h.append(h_)
        err.append(err_)
    h_q = [np.log(h[ii]/h[ii-1]) for ii in range(1, len(h))]
    err_q = [np.log(err[ii]/err[ii-1]) for ii in range(1, len(err))]
    return np.asarray([err_q[ii]/h_q[ii] for ii in range(len(err_q))])


def test_poisson():
    # if degree = 1, rate of convergence ~= 1.5
    q1 = conv_rate(1)
    assert (q1 < 1.7).all()
    # if degree = 2, rate of convergence ~= 2.
    q2 = conv_rate(2)
    assert (q2 > 1.9).all()

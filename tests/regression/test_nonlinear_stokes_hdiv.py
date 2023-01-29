import pytest
from firedrake import *


@pytest.mark.skipcomplex
def test_nonlinear_stokes_hdiv():
    mesh = UnitSquareMesh(3, 3)

    Vc = mesh.coordinates.function_space()

    x, y = SpatialCoordinate(mesh)
    f = Function(Vc).interpolate(as_vector([x,
                                            y + (1 - y)*Constant(0.1)*sin(2*pi*x)]))
    mesh.coordinates.assign(f)

    V1 = FunctionSpace(mesh, "BDM", 1)
    V2 = FunctionSpace(mesh, "DG", 0)
    W = V1*V2
    W = V1*V2

    w = Function(W)
    dw = TestFunction(W)

    def g(u):
        beta = Constant(1.0)
        return 0.5*(beta + 10*inner(grad(u), grad(u)))**(1./6)

    def epsilon(u):
        return sym(grad(u))

    u, p = split(w)
    v, q = split(dw)

    n = FacetNormal(mesh)

    h = CellSize(mesh)

    eta = Constant(30.)

    F = (inner(g(u)*epsilon(u), grad(v))*dx
         - p*div(v)*dx + q*div(u)*dx)

    def T(u):
        return u - inner(u, n)*n

    def N(u):
        return inner(u, n)*n

    F += (- inner(2*avg(outer(v, n)), avg(g(u)*epsilon(u)))*dS  # Consistency
          - inner(2*avg(outer(u, n)), avg(g(u)*epsilon(v)))*dS  # Symmetry
          + eta/avg(h)*inner(jump(u), jump(v))*dS)             # Penalty

    # left = 1 - we enforce u.n strongly and u.t = 0 via penalty method
    # top = 4
    # bottom = 3
    # right = 2 (natural)
    # Penalty method for tangent component on left (delete these terms for slip)
    F += (- inner(g(u)*T(dot(epsilon(u), n)), T(v))*ds(1)
          - inner(g(u)*T(dot(epsilon(v), n)), T(u))*ds(1)
          + eta/h*inner(T(u), T(v))*ds(1))

    # nothing on right-hand side (edge 2)

    # edge 3 (bottom) is now slip -- u.n=0 strongly and
    # Weertman-style sliding law through boundary integral

    mWeert = Constant(2.0)
    CWeert = Constant(1.0)

    # This is a hack to regularize the sliding law, which is nondifferentiable at 0
    epsWeert = Constant(1.e-4)

    F += inner(CWeert**(-1.0/mWeert)*(inner(T(u), T(u))+epsWeert)**(0.5/mWeert-0.5)*T(u), T(v))*ds(3)

    # edge 4 also no slip (delete these terms for slip)
    F += (- inner(g(u)*T(dot(epsilon(u), n)), T(v))*ds(4)
          - inner(g(u)*T(dot(epsilon(v), n)), T(u))*ds(4)
          + eta/h*inner(T(u), T(v))*ds(4))

    params = {'ksp_type': 'preonly',
              'mat_type': 'aij',
              'pc_type': 'lu',
              'pc_factor_shift_type': 'inblocks',
              'snes_linesearch_type': 'basic',
              'snes_atol': 1.0e-8}

    # parabolic profile on edge 1
    bcfunc = Function(V1).project(as_vector([y*(1 - y), 0]))
    bcs = [DirichletBC(W.sub(0), bcfunc, 1),
           DirichletBC(W.sub(0), zero(), 3),
           DirichletBC(W.sub(0), zero(), 4)]

    solve(F == 0, w, bcs=bcs, solver_parameters=params)

    u, p = w.subfunctions

    # test for penetration on bottom
    assert sqrt(assemble(dot(u, n)**2*ds(3))) < 1e-14

    # test for incompressibility
    assert sqrt(assemble(div(u)**2*dx)) < 1e-14

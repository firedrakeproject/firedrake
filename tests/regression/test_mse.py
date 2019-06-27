import pytest
import numpy as np
from firedrake import *


def get_complex(meshtype, hdegree, vdegree=None):

    if meshtype == 'extruded':
        vdegree = vdegree or hdegree
        h1_horiz = FiniteElement("GLL", "interval", hdegree)
        l2_horiz = FiniteElement("GLL-Edge L2", "interval", hdegree-1)
        h1_vert = FiniteElement("GLL", "interval", vdegree)
        l2_vert = FiniteElement("GLL-Edge L2", "interval", vdegree-1)
        hcurlelem = HCurl(TensorProductElement(h1_horiz, l2_vert)) + HCurl(TensorProductElement(l2_horiz, h1_vert))
        hdivelem = HDiv(TensorProductElement(h1_horiz, l2_vert)) + HDiv(TensorProductElement(l2_horiz, h1_vert))
        l2elem = TensorProductElement(l2_horiz, l2_vert)
        h1elem = TensorProductElement(h1_horiz, h1_vert)
    else:
        l2elem = FiniteElement('DQ L2', quadrilateral, hdegree-1, variant='mse')
        hdivelem = FiniteElement('RTCF', quadrilateral, hdegree, variant='mse')
        hcurlelem = FiniteElement('RTCE', quadrilateral, hdegree, variant='mse')
        h1elem = FiniteElement('Q', quadrilateral, hdegree, variant='mse')

    return h1elem, hcurlelem, hdivelem, l2elem


def get_mesh(r, meshtype, meshd=None):

    meshd = meshd or 1
    if meshtype == 'spherical':
        mesh = UnitCubedSphereMesh(refinement_level=r, degree=meshd)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    elif meshtype == 'planar':
        mesh = UnitSquareMesh(2**r, 2**r, quadrilateral=True)
    elif meshtype == 'extruded':
        basemesh = UnitIntervalMesh(2**r)
        mesh = ExtrudedMesh(basemesh, 2**r)
    return mesh


def helmholtz_mixed(r, meshtype, hdegree, vdegree=None, meshd=None, useaction=False):

    mesh = get_mesh(r, meshtype, meshd=meshd)
    _, _, hdivelem, l2elem = get_complex(meshtype, hdegree, vdegree=vdegree)

    V1 = FunctionSpace(mesh, hdivelem, name="V")
    V2 = FunctionSpace(mesh, l2elem, name="P")
    W = V1 * V2

    # Define variational problem
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(V2)

    x = SpatialCoordinate(mesh)
    if meshtype in ['planar', 'extruded']:
        f.project((1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
    elif meshtype == 'spherical':
        f.project(x[0]*x[1]*x[2])
    a = (p*q - q*div(u) + lmbda*inner(v, u) + div(v)*p) * dx
    L = f*q*dx

    # Compute solution
    sol = Function(W)

    if useaction:
        system = action(a, sol) - L == 0
    else:
        system = a == L

    # Block system is:
    # V Ct
    # Ch P
    # Eliminate V by forming a schur complement
    solve(system, sol, solver_parameters={'pc_type': 'fieldsplit',
                                          'pc_fieldsplit_type': 'schur',
                                          'ksp_type': 'cg',
                                          'pc_fieldsplit_schur_fact_type': 'FULL',
                                          'fieldsplit_V_ksp_type': 'cg',
                                          'fieldsplit_P_ksp_type': 'cg'})

    if meshtype in ['planar', 'extruded']:
        f.project(sin(x[0]*pi*2)*sin(x[1]*pi*2))
        return sqrt(assemble(dot(sol[2] - f, sol[2] - f) * dx))
    elif meshtype == 'spherical':
        _, u = sol.split()
        f.project(x[0]*x[1]*x[2]/13.0)
        return errornorm(f, u, degree_rise=0)


# helmholtz mixed on plane

@pytest.mark.parametrize(('degree', 'action', 'threshold'),
                         [(1, False, 1.9),
                          (2, False, 2.9),
                          (3, False, 3.9),
                          (2, True, 2.9)])
def test_firedrake_helmholtz_mixed_mse(degree, action, threshold):
    diff = np.array([helmholtz_mixed(r, 'planar', degree, useaction=action) for r in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()


# helmholtz mixed on sphere

@pytest.mark.parametrize(('degree', 'md', 'action', 'threshold'),
                         [(1, 1, False, 1.67),
                          (2, 1, False, 1.9),
                          (2, 2, False, 2.9),
                          (3, 1, False, 1.87),
                          (3, 2, False, 2.9),
                          (3, 3, False, 3.9),
                          (2, 2, True, 2.9)])
def test_firedrake_helmholtz_mixed_mse_sphere(degree, md, action, threshold):
    diff = np.array([helmholtz_mixed(r, 'spherical', degree, meshd=md, useaction=action) for r in range(2, 5)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()


# helmholtz mixed on extruded

@pytest.mark.parametrize(('hdegree', 'vdegree', 'threshold', 'action'),
                         [(1, 1, 1.9, False),
                          (2, 1, 1.9, False),
                          (1, 2, 1.9, False),
                          (2, 2, 2.9, False),
                          (1, 1, 1.9, True)])
def test_firedrake_helmholtz_mixed_mse_extruded(hdegree, vdegree, threshold, action):
    diff = np.array([helmholtz_mixed(i, 'extruded', hdegree, vdegree=vdegree, useaction=action) for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()


# facet integrals- interior and exterior
# tested using an interior penalty formulation for the Helmholtz using DG/DQ L2 elements


def laplacian_IP(r, degree, meshd, meshtype):

    dIF = dS
    dEF = ds
    if meshtype == 'extruded':
        dIF = dS_v + dS_h
        dEF = ds_t + ds_b + ds_v

    mesh = get_mesh(r, meshtype, meshd=meshd)
    _, _, _, l2elem = get_complex(meshtype, degree+1, vdegree=degree+1)

    V = FunctionSpace(mesh, l2elem)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)

    x = SpatialCoordinate(mesh)
    if meshtype in ['planar', 'extruded']:
        f.project((1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
    elif meshtype == 'spherical':
        f.project(x[0]*x[1]*x[2])

    FA = FacetArea(mesh)
    CV = CellVolume(mesh)
    ddx = CV/FA
    ddx_avg = (ddx('+') + ddx('-'))/2.
    alpha = Constant(4. * degree * (degree + 1.))
    gamma = Constant(8. * degree * (degree + 1.))
    penalty_int = alpha / ddx_avg
    penalty_ext = gamma / ddx

    n = FacetNormal(mesh)
    aV = (inner(grad(u), grad(v)) + inner(u, v)) * dx  # volume term
    aIF = (inner(jump(u, n), jump(v, n)) * penalty_int - inner(avg(grad(u)), jump(v, n)) - inner(avg(grad(v)), jump(u, n))) * dIF  # interior facet term
    aEF = (inner(u, v) * penalty_ext - inner(grad(u), v*n) - inner(grad(v), u*n)) * dEF  # exterior facet term
    a = aV + aEF + aIF
    L = f*v*dx

    # Compute solution
    sol = Function(V)

    solve(a == L, sol, solver_parameters={'pc_type': 'ilu',
                                          'ksp_type': 'lgmres'})

    # Analytical solution
    if meshtype in ['planar', 'extruded']:
        f.project(sin(x[0]*pi*2)*sin(x[1]*pi*2))
    elif meshtype == 'spherical':
        f.project(x[0]*x[1]*x[2]/13.0)
    return sqrt(assemble(dot(sol - f, sol - f) * dx))


@pytest.mark.parametrize(('degree', 'meshd', 'meshtype', 'threshold'),
                         [(1, 1, 'planar', 1.8),
                          (2, 1, 'planar', 2.9),
                          (1, 1, 'extruded', 1.8),
                          (2, 1, 'extruded', 2.9),
                          (2, 2, 'spherical', 2.9),
                          (3, 3, 'spherical', 3.9)])
def test_firedrake_laplacian_IP_mse(degree, meshd, meshtype, threshold):
    diff = np.array([laplacian_IP(i, degree, meshd, meshtype) for i in range(2, 5)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()


def vector_laplace(r, meshtype, hdegree, vdegree=None):
    vdegree = vdegree or hdegree

    mesh = get_mesh(r, meshtype)
    h1elem, hcurlelem, _, _ = get_complex(meshtype, hdegree, vdegree=vdegree)
    h1elem_high, _, _, _ = get_complex(meshtype, hdegree+1, vdegree=vdegree+1)

    # spaces for calculation
    V0 = FunctionSpace(mesh, h1elem)
    V1 = FunctionSpace(mesh, hcurlelem)
    V = V0*V1

    # spaces to store 'analytic' functions
    W0 = FunctionSpace(mesh, h1elem_high)
    W1 = VectorFunctionSpace(mesh, h1elem_high)

    # constants
    k = 1.0
    l = 2.0

    xs = SpatialCoordinate(mesh)
    f_expr = as_vector([pi*pi*(k*k + l*l)*sin(k*pi*xs[0])*cos(l*pi*xs[1]), pi*pi*(k*k + l*l)*cos(k*pi*xs[0])*sin(l*pi*xs[1])])
    exact_s_expr = -(k+l)*pi*cos(k*pi*xs[0])*cos(l*pi*xs[1])
    exact_u_expr = as_vector([sin(k*pi*xs[0])*cos(l*pi*xs[1]), cos(k*pi*xs[0])*sin(l*pi*xs[1])])

    f = Function(W1).project(f_expr)
    exact_s = Function(W0).project(exact_s_expr)
    exact_u = Function(W1).project(exact_u_expr)

    sigma, u = TrialFunctions(V)
    tau, v = TestFunctions(V)
    a = (sigma*tau - dot(u, grad(tau)) + dot(grad(sigma), v) + dot(curl(u), curl(v)))*dx
    L = dot(f, v)*dx

    out = Function(V)

    # preconditioner for H1 x H(curl)
    aP = (dot(grad(sigma), grad(tau)) + sigma*tau + dot(curl(u), curl(v)) + dot(u, v))*dx

    solve(a == L, out, Jp=aP,
          solver_parameters={'pc_type': 'fieldsplit',
                             'pc_fieldsplit_type': 'additive',
                             'fieldsplit_0_pc_type': 'lu',
                             'fieldsplit_1_pc_type': 'lu',
                             'ksp_monitor': None})

    out_s, out_u = out.split()

    return (sqrt(assemble(dot(out_u - exact_u, out_u - exact_u)*dx)),
            sqrt(assemble((out_s - exact_s)*(out_s - exact_s)*dx)))


@pytest.mark.parametrize(('degree', 'threshold'),
                         [(1, 0.9),
                          (2, 1.9),
                          (3, 2.9),
                          (2, 0.9)])
def test_firedrake_helmholtz_vector_mse(degree, threshold):
    diff = np.array([vector_laplace(r, 'planar', degree) for r in range(2, 5)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()


@pytest.mark.parametrize(('hdegree', 'vdegree', 'threshold'),
                         [(1, 1, 0.9),
                          (2, 1, 0.9),
                          (1, 2, 0.9),
                          (2, 2, 1.9),
                          (1, 1, 0.9)])
def test_firedrake_helmholtz_vector_mse_extruded(hdegree, vdegree, threshold):
    diff = np.array([vector_laplace(i, 'extruded', hdegree, vdegree=vdegree) for i in range(2, 5)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1, :] / diff[1:, :])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()

import pytest
import numpy as np
from firedrake import *


def get_complex(family, hdegree, vdegree=None):

    if family == 'RT':
        l2elem = FiniteElement('DG L2', triangle, hdegree-1)
        hdivelem = FiniteElement('RT', triangle, hdegree)
    elif family == 'BDM':
        l2elem = FiniteElement('DG L2', triangle, hdegree-1)
        hdivelem = FiniteElement('BDM', triangle, hdegree)
    elif family == 'BDFM':
        l2elem = FiniteElement('DG L2', triangle, hdegree-1)
        hdivelem = FiniteElement('BDFM', triangle, hdegree)
    elif family == 'RTCF':
        l2elem = FiniteElement('DQ L2', quadrilateral, hdegree-1)
        hdivelem = FiniteElement('RTCF', quadrilateral, hdegree)
    elif family == 'ext':
        vdegree = vdegree or hdegree
        h1_horiz = FiniteElement("CG", "interval", hdegree)
        l2_horiz = FiniteElement("DG L2", "interval", hdegree-1)
        h1_vert = FiniteElement("CG", "interval", vdegree)
        l2_vert = FiniteElement("DG L2", "interval", vdegree-1)
        hdivelem = HDiv(TensorProductElement(h1_horiz, l2_vert)) + HDiv(TensorProductElement(l2_horiz, h1_vert))
        l2elem = TensorProductElement(l2_horiz, l2_vert)
    return hdivelem, l2elem


def get_mesh(r, meshtype, meshd=None):

    meshd = meshd or 1
    if meshtype == 'spherical-quad':
        mesh = UnitCubedSphereMesh(refinement_level=r, degree=meshd)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    elif meshtype == 'spherical-tri':
        mesh = UnitIcosahedralSphereMesh(refinement_level=r, degree=meshd)
        x = SpatialCoordinate(mesh)
        mesh.init_cell_orientations(x)
    elif meshtype == 'planar-quad':
        mesh = UnitSquareMesh(2**r, 2**r, quadrilateral=True)
    elif meshtype == 'planar-tri':
        mesh = UnitSquareMesh(2**r, 2**r, quadrilateral=False)
    elif meshtype == 'extruded':
        basemesh = UnitIntervalMesh(2**r)
        mesh = ExtrudedMesh(basemesh, 2**r)
    return mesh


def helmholtz_mixed(r, meshtype, family, hdegree, vdegree=None, meshd=None, useaction=False):

    mesh = get_mesh(r, meshtype, meshd=meshd)
    hdivelem, l2elem = get_complex(family, hdegree, vdegree=vdegree)

    V1 = FunctionSpace(mesh, hdivelem, name="V")
    V2 = FunctionSpace(mesh, l2elem, name="P")
    W = V1 * V2

    # Define variational problem
    lmbda = 1
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    f = Function(V2)

    x = SpatialCoordinate(mesh)
    if meshtype in ['planar-quad', 'planar-tri', 'extruded']:
        f.project((1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
    elif meshtype in ['spherical-quad', 'spherical-tri']:
        f.project(x[0]*x[1]*x[2])
    a = (inner(p, q) - inner(div(u), q) + lmbda*inner(u, v) + inner(p, div(v))) * dx
    L = inner(f, q)*dx

    # Compute solution
    sol = Function(W)

    if useaction:
        system = action(a, sol) - L == 0
    else:
        system = a == L

    solve(system, sol)

    # Analytical solution

    if meshtype in ['planar-quad', 'planar-tri', 'extruded']:
        f.project(sin(x[0]*pi*2)*sin(x[1]*pi*2))
        return sqrt(assemble(dot(sol[2] - f, sol[2] - f) * dx))
    elif meshtype in ['spherical-quad', 'spherical-tri']:
        _, u = sol.subfunctions
        f.project(x[0]*x[1]*x[2]/13.0)
        return errornorm(f, u, degree_rise=0)


# helmholtz mixed on plane

@pytest.mark.parametrize(('family', 'degree', 'celltype', 'action', 'threshold'),
                         [('RT', 1, 'tri', False, 1.9),
                          ('RT', 2, 'tri', False, 2.9),
                          ('BDM', 1, 'tri', False, 1.87),
                          ('BDM', 2, 'tri', False, 2.9),
                          ('BDFM', 2, 'tri', False, 2.9),
                          ('RTCF', 1, 'quad', False, 1.9),
                          ('RTCF', 2, 'quad', False, 2.9),
                          ('RT', 2, 'tri', True, 2.9),
                          ('BDM', 2, 'tri', True, 2.9)])
def test_firedrake_helmholtz_mixed_l2pullbacks(family, degree, celltype, action, threshold):
    diff = np.array([helmholtz_mixed(r, 'planar-' + celltype, family, degree, useaction=action) for r in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()


# helmholtz mixed on sphere

@pytest.mark.parametrize(('family', 'degree', 'celltype', 'md', 'action', 'threshold'),
                         [('RT', 1, 'tri', 1, False, 1.9),
                          ('RT', 2, 'tri', 1, False, 1.9),
                          ('RT', 2, 'tri', 2, False, 2.9),
                          ('BDM', 1, 'tri', 1, False, 1.88),
                          pytest.param('BDM', 2, 'tri', 1, False, 1.9, marks=pytest.mark.skipcomplex(
                              reason="See https://github.com/firedrakeproject/firedrake/issues/2125"
                          )),
                          ('BDM', 2, 'tri', 2, False, 2.9),
                          ('BDFM', 2, 'tri', 1, False, 1.9),
                          ('BDFM', 2, 'tri', 2, False, 2.9),
                          ('RTCF', 1, 'quad', 1, False, 1.67),
                          ('RTCF', 2, 'quad', 1, False, 1.9),
                          ('RTCF', 2, 'quad', 2, False, 2.9),
                          ('RT', 2, 'tri', 1, True, 1.9),
                          pytest.param('BDM', 2, 'tri', 1, True, 1.9, marks=pytest.mark.skipcomplex(
                              reason="See https://github.com/firedrakeproject/firedrake/issues/2125"
                          ))])
def test_firedrake_helmholtz_mixed_l2pullbacks_sphere(family, degree, celltype, md, action, threshold):
    diff = np.array([helmholtz_mixed(r, 'spherical-' + celltype, family, degree, meshd=md, useaction=action) for r in range(2, 5)])
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
def test_firedrake_helmholtz_mixed_l2pullbacks_extruded(hdegree, vdegree, threshold, action):
    diff = np.array([helmholtz_mixed(i, 'extruded', 'ext', hdegree, vdegree=vdegree, useaction=action) for i in range(3, 6)])
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
        hdivelem, l2elem = get_complex('ext', degree+1, vdegree=degree+1)
    elif meshtype in ['spherical-quad', 'planar-quad']:
        hdivelem, l2elem = get_complex('RTCF', degree+1)
    elif meshtype in ['spherical-tri', 'planar-tri']:
        hdivelem, l2elem = get_complex('RT', degree+1)

    mesh = get_mesh(r, meshtype, meshd=meshd)

    V = FunctionSpace(mesh, l2elem)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)

    x = SpatialCoordinate(mesh)
    if meshtype in ['planar-quad', 'planar-tri', 'extruded']:
        f.project((1+8*pi*pi)*sin(x[0]*pi*2)*sin(x[1]*pi*2))
    elif meshtype in ['spherical-quad', 'spherical-tri']:
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
    aIF = (inner(jump(u, n), jump(v, n)) * penalty_int - inner(avg(grad(u)), jump(v, n)) - inner(jump(u, n), avg(grad(v)))) * dIF  # interior facet term
    aEF = (inner(u, v) * penalty_ext - inner(grad(u), v*n) - inner(u*n, grad(v))) * dEF  # exterior facet term
    a = aV + aEF + aIF
    L = inner(f, v)*dx

    # Compute solution
    sol = Function(V)

    solve(a == L, sol, solver_parameters={
        'pc_type': 'ilu',
        'pc_factor_mat_solver_type': 'petsc',
        'ksp_type': 'lgmres'
    })

    # Analytical solution
    if meshtype in ['planar-quad', 'planar-tri', 'extruded']:
        f.project(sin(x[0]*pi*2)*sin(x[1]*pi*2))
    elif meshtype in ['spherical-quad', 'spherical-tri']:
        f.project(x[0]*x[1]*x[2]/13.0)
    return sqrt(assemble(dot(sol - f, sol - f) * dx))


@pytest.mark.parametrize(('degree', 'meshd', 'meshtype', 'threshold'),
                         [(1, 1, 'planar-quad', 1.8),
                          (2, 1, 'planar-quad', 2.9),
                          (1, 1, 'planar-tri', 1.5),
                          (2, 1, 'planar-tri', 2.9),
                          (1, 1, 'extruded', 1.8),
                          (2, 1, 'extruded', 2.9),
                          (2, 2, 'spherical-quad', 2.9),
                          (3, 3, 'spherical-quad', 3.9),
                          (2, 2, 'spherical-tri', 2.9),
                          (3, 3, 'spherical-tri', 3.9)])
def test_firedrake_laplacian_IP_l2pullbacks(degree, meshd, meshtype, threshold):
    diff = np.array([laplacian_IP(i, degree, meshd, meshtype) for i in range(2, 5)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > threshold).all()

from test_helmholtz import helmholtz
from test_poisson_strong_bcs import run_test
from test_steady_advection_3D import run_near_to_far
from fuse.cells import ufc_triangle
from firedrake import *
import pytest
import numpy as np
from fuse import *




@pytest.mark.parametrize(['conv_num', 'degree'],
                         [(p, d)
                          for p, d in zip([1.8, 2.8, 3.8], [1, 2, 3])])
def test_helmholtz(mocker, conv_num, degree):
    # mocker.patch('firedrake.mesh.as_cell', return_value=ufc_triangle().to_ufl("triangle"))
    diff = np.array([helmholtz(i, degree=degree)[0] for i in range(3, 6)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_num).all()


@pytest.mark.parametrize(['conv_num', 'degree'],
                         [(p, d)
                          for p, d in zip([ 2.8, 3.8], [2, 3])])
def test_helmholtz_3d(mocker, conv_num, degree):
    diff = np.array([helmholtz(i, degree=degree, mesh=UnitCubeMesh(2 ** i, 2 ** i, 2 ** i))[0] for i in range(2, 4)])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > conv_num).all()

# @pytest.mark.parametrize(['conv_num', 'degree'],
#                          [(p, d)
#                           for p,d in zip([1.8, 2.8, 3.8],[1, 2, 3])])
# def test_helmholtz(conv_num, degree):
#     diff = np.array([helmholtz(i, degree=degree)[0] for i in range(3, 6)])
#     print("l2 error norms:", diff)
#     conv = np.log2(diff[:-1] / diff[1:])
#     print("convergence order:", conv)
#     assert (np.array(conv) > conv_num).all()


def construct_cg3(tri=None):
    # [test_cg3 0]
    tri = polygon(3)
    edge = tri.edges()[0]
    vert = tri.vertices()[0]

    xs = [DOF(DeltaPairing(), PointKernel(()))]
    dg0 = ElementTriple(vert, (P0, CellL2, C0), DOFGenerator(xs, S1, S1))

    v_xs = [immerse(tri, dg0, TrH1)]
    v_dofs = DOFGenerator(v_xs, C3, S1)

    xs = [DOF(DeltaPairing(), PointKernel((-1/3)))]
    dg0_int = ElementTriple(edge, (P1, CellH1, C0), DOFGenerator(xs, S2, S1))

    e_xs = [immerse(tri, dg0_int, TrH1)]
    e_dofs = DOFGenerator(e_xs, C3, S1)

    i_xs = [DOF(DeltaPairing(), PointKernel((0, 0)))]
    i_dofs = DOFGenerator(i_xs, S1, S1)

    cg3 = ElementTriple(tri, (P3, CellH1, C0), [v_dofs, e_dofs, i_dofs])
    # [test_cg3 1]
    return cg3

def construct_dg1_tri():
    # [test_dg1_tri 0]
    tri = polygon(3)
    xs = [DOF(DeltaPairing(), PointKernel((-1, -np.sqrt(3)/3)))]
    dg1 = ElementTriple(tri, (P1, CellL2, C0), DOFGenerator(xs, S3/S2, S1))
    # [test_dg1_tri 1]
    return dg1

def create_dg1(cell):
    xs = [DOF(DeltaPairing(), PointKernel(cell.vertices(return_coords=True)[0]))]
    Pk = PolynomialSpace(1)
    dg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return dg


def create_cg1():
    cell = polygon(3)
    deg = 1
    vert_dg = create_dg1(cell.vertices()[0])
    xs = [immerse(cell, vert_dg, TrH1)]

    Pk = PolynomialSpace(deg)
    cg = ElementTriple(cell, (Pk, CellL2, C0), DOFGenerator(xs, get_cyc_group(len(cell.vertices())), S1))
    return cg

def test_orientation_string():
    elem = construct_cg3()
    mesh = UnitTriangleMesh()
    U = FunctionSpace(mesh, elem.to_ufl())

    v = TestFunction(U)
    L = v * dx
    l_a = assemble(L)

def test_minimal():
    # NB mesh size 3,3 fails - internal cell issue?
    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 1)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = inner(grad(v), grad(u)) * dx
    L = v * dx
    
    l_a = assemble(L)
    a_1 = assemble(a)
    x = Function(V)
    solve(a == L, x)
    print(x.dat.data)
    print("done with firedrake elem")

    cg1 = create_cg1()
    cg3 = construct_cg3()
    V = FunctionSpace(mesh, cg1.to_ufl())
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(v), grad(u)) * dx
    L = v * dx

    l_a = assemble(L)
    a_2 = assemble(a)

    x = Function(V)
    solve(a == L, x)

    print(x.dat.data)
    print("done with fuse elem")

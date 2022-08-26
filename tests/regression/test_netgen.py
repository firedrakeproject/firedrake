from firedrake import *
from firedrake import mesh
from netgen.geom2d import SplineGeometry
from netgen.occ import *
import numpy as np


def poisson(h, degree=2):
    # Setting up Netgen geometry and mesh
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (np.pi, np.pi), bc="rect")
    ngmesh = geo.GenerateMesh(maxh=h)
    msh = mesh.Mesh(ngmesh)

    # Setting up the problem
    V = FunctionSpace(msh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(msh)
    f.interpolate(2*sin(x)*sin(y))
    a = inner(grad(u), grad(v))*dx
    l = inner(f, v) * dx
    u = Function(V)
    bc = DirichletBC(V, 0.0, ngmesh.GetBCIDs("rect"))

    # Assembling matrix
    A = assemble(a, bcs=bc)
    b = assemble(l)
    bc.apply(b)

    # Solving the problem
    solve(A, u, b, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Computing the error
    f.interpolate(sin(x)*sin(y))
    return sqrt(assemble(inner(u - f, u - f) * dx)), u, f


def poisson3D(h, degree=2):
    # Setting up Netgen geometry and mesh
    box = Box(Pnt(0,0,0), Pnt(np.pi,np.pi,np.pi))
    box.bc("bcs")
    geo = OCCGeometry(box)
    ngmesh = geo.GenerateMesh(maxh=h);

    msh = mesh.Mesh(ngmesh)

    # Setting up the problem
    V = FunctionSpace(msh, "CG", degree)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y, z = SpatialCoordinate(msh)
    f.interpolate(3*sin(x)*sin(y)*sin(z))
    a = inner(grad(u), grad(v))*dx
    l = inner(f, v) * dx
    u = Function(V)
    bc = DirichletBC(V, 0.0, ngmesh.GetBCIDs("bcs"))

    # Assembling matrix
    A = assemble(a, bcs=bc)
    b = assemble(l)
    bc.apply(b)

    # Solving the problem
    solve(A, u, b, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    # Computing the error
    f.interpolate(sin(x)*sin(y)*sin(z))
    return sqrt(assemble(inner(u - f, u - f) * dx)), u, f

def test_firedrake_Poisson_netgen():
    diff = np.array([poisson(h)[0] for h in [1/2, 1/4, 1/8]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()
def test_firedrake_Poisson3D_netgen():
    diff = np.array([poisson3D(h)[0] for h in [2, 1, 1/2]])
    print("l2 error norms:", diff)
    conv = np.log2(diff[:-1] / diff[1:])
    print("convergence order:", conv)
    assert (np.array(conv) > 2.8).all()

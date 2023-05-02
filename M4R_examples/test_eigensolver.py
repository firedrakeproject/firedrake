from firedrake.eigensolver import *
from firedrake.utility_meshes import UnitSquareMesh, RectangleMesh
from firedrake.functionspace import FunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.constant import Constant
from firedrake import grad, inner, dx
from firedrake.bcs import DirichletBC

def helmholtz_test():
    mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    A = (inner(grad(u), grad(v)) + u*v )* dx
    bcs = DirichletBC(V, 0.0, "on_boundary")
    Lin_EP = LinearEigenproblem(A, bcs=bcs)
    Lin_ES = LinearEigensolver(Lin_EP, n_evals=1)
    nconv = Lin_ES.solve()
    print(nconv)

def tutorial():
    mesh = UnitSquareMesh(10, 10)
    Vcg  = FunctionSpace(mesh,'CG',3)
    Lx   = 1.
    Ly   = 1.
    n0   = 50
    mesh = RectangleMesh(n0, n0, Lx, Ly, reorder=None)
    bc = DirichletBC(Vcg, 0.0, "on_boundary")
    beta = Constant('1.0')
    F    = Constant('1.0')
    phi, psi = TestFunction(Vcg), TrialFunction(Vcg)
    a =  beta*phi*psi.dx(0)*dx
    m = -inner(grad(psi), grad(phi))*dx - F*psi*phi*dx
    eigenprob = LinearEigenproblem(a, m, bcs=bc) # try with no m
    eigensolver = LinearEigensolver(eigenprob, 1)
    eigensolver.solve()
    evals = eigensolver.eigenvalues()
    lam, eigenmodes_real, eigenmodes_imag = eigensolver.eigenfunctions(0)  # leading eval
    with eigenmodes_real.vector().dat.vec as evec_r:  # Firedrake vector
        print(evec_r[:])
    with eigenmodes_imag.vector().dat.vec as evec_i:  # Firedrake vector
        print(evec_i[:])



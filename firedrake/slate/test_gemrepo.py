
from firedrake import *

#FEM problem (Helmholtz equation)
mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

def test_assembletensorandsolve(a,L,V):
    _A = Tensor(a)
    _F = Tensor(L)
    A=assemble(_A)
    F=assemble(_F)
    A_comp=assemble(a)
    F_comp=assemble(F)

    #TEST1: compare slate and non-slate assembled tensors
    print((A.M.handle-A_comp.M.handle).norm())
    print((F.M.handle-F_comp.M.handle).norm())

    #TEST2: solve
    u=Function(V)
    u_comp=Function(V)
    solve(test1, u, test2, solver_parameters={'ksp_type': 'cg'})
    solve(a==L, u_comp, solver_parameters={'ksp_type': 'cg'})
    print((u.dat.data-u_comp.dat.data).norm())

def test_assembledvector(L):
    _coeff_F = AssembledVector(assemble(L))
    coeff_F=assemble(_coeff_F)
    coeff_F_comp = assemble(L)
    print((coeff_F.dat.data-coeff_F_comp.dat.data).norm())

def test_add(a):
    _A = Tensor(a)
    add_A=assemble(_A+_A)
    add_A_comp=assemble(a+a)
    print((add_A.M.handle-add_A_comp.M.handle).norm())


def test_negative(a):
    _A = Tensor(a)
    neg_A=assemble(-_A)
    neg_A_comp=assemble(-a)
    print((neg_A.M.handle-neg_A_comp.M.handle).norm())

#TODO we need advection problem rather than hemlholtz to have non symmetric tensors
def test_transpose(a):
    _A = Tensor(a)
    A = assemble(_A)
    trans_A=assemble(Transpose(_A))
    print((trans_A.M.handle-A_comp.M.handle).norm())#should be 0 here because of symmetry


#run test
test_assembletensorandsolve(a,L,V)




###############################################
#TODO: TEST: assemble contraction
#test=assemble(_A*_A)

#TODO: TEST: assemble contraction
#test=assemble(_A*_F)

#TODO: TEST: assemble blocks
#this is getting more interesting if mixed
#b=assemble(_A.blocks[0,0])


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

def test_assemble2form(a):
    _A = Tensor(a)
    A=assemble(_A)
    A_comp=assemble(a)
    print((A.M.handle-A_comp.M.handle).norm())

#def test_assemble1form(L):
#    _F = Tensor(L)
#    F=assemble(_F)
#    F_comp=assemble(F)
#    print((F.M.handle-F_comp.M.handle).norm())

#in order to be able to do solve I need to do mul first
def test_solve(a,L,V):
    #assemble
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))
    A=assemble(_A)
    F=assemble(_F)

    #solve
    u=Function(V)
    u_comp=Function(V)
    solve(assemble(_A), u, assemble(_F))
    solve(a==L, u_comp, solver_parameters={'ksp_type': 'cg'})
    print((u.dat.data-u_comp.dat.data).norm())

#TODO test fails!
def test_assembledvector(L):
    b=Function(assemble(L))
    _coeff_F = AssembledVector(b)

    print(_coeff_F._function.dat.data)
    coeff_F=assemble(_coeff_F)
    coeff_F_comp = assemble(L)
    print(coeff_F.dat.data)
    print(coeff_F_comp.dat.data)

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

#TODO 
def test_mul(a,L):
    pass

###########
#run tests
###########

#test_assemble2form(a)
test_assembledvector(L)
#TODO
#test_mul(L)
#test_solve(a,L,V)




###############################################
#TODO: TEST: assemble contraction
#test=assemble(_A*_A)

#TODO: TEST: assemble contraction
#test=assemble(_A*_F)

#TODO: TEST: assemble blocks
#this is getting more interesting if mixed
#b=assemble(_A.blocks[0,0])

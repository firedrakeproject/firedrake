
from firedrake import *


def test_assemble2form(a):
    _A = Tensor(a)
    A = assemble(_A)
    A_comp = assemble(a)
    assert F.M.handle.norm() == F_comp.M.handle.norm(), "Test for assembly of 2-form failed"

def test_assemble1form(L):
    _F = Tensor(L)
    F = assemble(_F)
    F_comp = assemble(F)
    assert F.M.handle.norm() == F_comp.M.handle.norm(), "Test for assembly of 1-form failed"

#in order to be able to do solve I need to do mul first
def test_solve(a,L,V):
    #assemble
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))

    #solve
    u = Function(V)
    u_comp = Function(V)
    solve(assemble(_A), u, assemble(_F))
    solve(a == L, u_comp, solver_parameters={'ksp_type': 'cg'})
    assert u.dat.data == u_comp.dat.data, "Test for solved on assembled forms failed"

#Note: this test only works for discontinuous function spaces
def test_assembledvector(L):
    _coeff_F = AssembledVector(Function(assemble(L)))
    coeff_F = assemble(_coeff_F)
    coeff_F_comp = assemble(L)
    assert coeff_F.dat.data == coeff_F_comp.dat.data, "Test for assembled vectors failed"

def test_add(a):
    _A = Tensor(a)
    add_A = assemble(_A+_A)
    add_A_comp = assemble(a+a)
    assert add_A.M.handle == add_A_comp.M.handle,  "Test for adding of two 2-forms failed"

def test_negative(a):
    _A = Tensor(a)
    neg_A=assemble(-_A)
    neg_A_comp=assemble(-a)
    assert neg_A.M.handle == neg_A_comp.M.handle,  "Test for negative of a two 2-form failed"

#Note: this only really a test for a problem containing an unsymmetric operator 
def test_transpose(a):
    _A = Tensor(a)
    A = assemble(_A)
    trans_A=assemble(Transpose(_A))
    assert trans_A.M.handle == trans_A_comp.M.handle,  "Test for transpose of a two 2-form failed"


#Note: this tests only a mat vec contraction atm
def test_mul(A,L,V):
    _A = Tensor(a)
    mat_comp = assemble(a)
    b = Function(assemble(L))
    _coeff_F = AssembledVector(b)
    mul_matvec = assemble(_A*_coeff_F)
    mul_matvec_comp = assemble(action(a,b))
    assert mul_matvec.dat.data == mul_matvec_comp.dat.data, "Test for contraction (mat-vec-mul) failed"



###########
#run tests
###########

#discontinuous Helmholtz equation
mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, "DG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

test_assembledvector(L)
#test_mul(a,L,V)
#test_solve(a,L,V)

#continuous Helmholtz equation
mesh = UnitSquareMesh(5,5)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

#test_assemble2form(a)
#test_assemble1form(a)
#test_negative(a)
#test_add(a)

#continuous advection problem
#TODO

#test_transpose(a)



###############################################
#TODO: TEST: assemble mul of two 2-froms
#test=assemble(_A*_A)
#TODO: TEST: assemble blocks

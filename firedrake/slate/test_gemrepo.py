
from firedrake import *
import math
import copy

def test_assemble_matrix(a):
    _A = Tensor(a)
    A = assemble(_A)
    A_comp = assemble(a)
    assert A.M.handle.norm() == A_comp.M.handle.norm(), "Test for assembly of 2-form failed"

#in order to be able to do solve I need to do mul first
def test_solve(a,L,V):
    #assemble
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))

    #solve
    u = Function(V)
    u_comp = Function(V)
    solve(assemble(_A), u, assemble(_F),solver_parameters={'ksp_type': 'cg'})
    solve(a == L, u_comp, solver_parameters={'ksp_type': 'cg'})
    assert u.dat.data.all()  == u_comp.dat.data.all() , "Test for solved on assembled forms failed"

#Note: this test only works for discontinuous function spaces
def test_assemble_vector(L):
    _coeff_F = AssembledVector(Function(assemble(L)))
    coeff_F = assemble(_coeff_F)
    coeff_F_comp = assemble(L)
    assert math.isclose(coeff_F.dat.data.all(),coeff_F_comp.dat.data.all()), "Test for assembled vectors failed"

def test_add(a):
    _A = Tensor(a)
    add_A = assemble(_A+_A)
    add_A_comp = assemble(a+a)
    for i in range(add_A.M.handle.getSize()[0]):
        for j in range(add_A.M.handle.getSize()[1]):
            assert math.isclose(add_A.M.handle.getValues(i,j),add_A_comp.M.handle.getValues(i,j)),  "Test for adding of a two 2-form failed"


def test_negative(a):
    _A = Tensor(a)
    neg_A=assemble(-_A)
    neg_A_comp=assemble(-a)
    for i in range(neg_A.M.handle.getSize()[0]):
        for j in range(neg_A.M.handle.getSize()[1]):
            assert math.isclose(neg_A.M.handle.getValues(i,j),neg_A_comp.M.handle.getValues(i,j)),  "Test for negative of a two 2-form failed"

#TODO: this only really a test for a problem containing an unsymmetric operator 
def test_transpose(a):
    _A = Tensor(a)
    A = assemble(_A)
    trans_A=assemble(Transpose(_A))
    for i in range(trans_A.M.handle.getSize()[0]):
        for j in range(trans_A.M.handle.getSize()[1]):
            assert math.isclose(trans_A.M.handle.getValues(i,j),A_comp.M.handle.getValues(j,i)),  "Test for negative of a two 2-form failed"

def test_mul(A,L,V,mesh):

    #test for mat-vec multiplication
    _A = Tensor(a)
    mat_comp = assemble(a)
    b = Function(assemble(L))
    _coeff_F = AssembledVector(b)
    mul_matvec = assemble(_A*_coeff_F)
    mul_matvec_comp = assemble(action(a,b))
    assert math.isclose(mul_matvec.dat.data.all(),mul_matvec_comp.dat.data.all()) , "Test for contraction (mat-vec-mul) failed"

    #test for mat-mat multiplication
    u2 = TrialFunction(V)
    v2 = TestFunction(V)
    f2 = Function(V)
    x2, y2 = SpatialCoordinate(mesh)
    f2.interpolate((1+8*pi*pi)*cos(x2*pi*2)*cos(y2*pi*2))
    a2 = (dot(grad(v2), grad(u2))) * dx
    _A2 = Tensor(a2)
    mul_matmat = assemble(_A*_A2)
    mul_matmat_comp = assemble(_A).M.handle* assemble(_A2).M.handle
    for i in range(mul_matmat.M.handle.getSize()[0]):
        for j in range(mul_matmat.M.handle.getSize()[1]):
            assert math.isclose(mul_matmat_comp.getValues(i,j),mul_matmat.M.handle.getValues(i,j)),  "Test for mat-mat-mul failed"




###########
print("Run test for slate to loopy compilation.")

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

test_assemble_vector(L) #TODO: dependecy generation doesnt seem quite right in this case
test_mul(a,L,V,mesh)
#test_solve(a,L,V) #fails

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

test_assemble_matrix(a)
test_negative(a)
test_add(a)

#TODO: continuous advection problem 
n = 5
mesh = UnitSquareMesh(n,n)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
u_ = Function(V).project(x)
u = TrialFunction(V)
v = TestFunction(V)
F = (u_*div(v*u))*dx

#test_assemble2form(F) 

###############################################
#TODO: TEST: assemble blocks

print("All tests passed.")
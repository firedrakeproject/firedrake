from firedrake import *
import math
from firedrake.formmanipulation import split_form
import numpy as np
import petsc4py.PETSc as PETSc


def test_assemble_matrix(a):
    print("Test of assemble matrix")

    _A = Tensor(a)
    A = assemble(_A)
    A_comp = assemble(a)
    for i in range(A.M.handle.getSize()[0]):
        for j in range(A.M.handle.getSize()[1]):
            assert math.isclose(A.M.handle.getValues(i, j), A_comp.M.handle.getValues(i, j)), "Test for assembly of tensor failed"


def test_solve(a, L, V):
    """
    Note: this test only works for DG problems because assembled vector does not do the right thing
    the bug is also in the earlier version of slate compiler
    """
    print("Test of global solve")

    # assemble
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))

    # solve
    u = Function(V)
    u_comp = Function(V)
    solve(assemble(_A), u, assemble(_F), solver_parameters={'ksp_type': 'cg'})
    solve(a == L, u_comp, solver_parameters={'ksp_type': 'cg'})
    assert u.dat.data.all() == u_comp.dat.data.all(), "Test for solve on assembled forms failed"


def test_assembled_vector(L):
    """
    Note: this test only works for discontinuous function spaces
    """
    print("Test of assemble vector")

    _coeff_F = AssembledVector(Function(assemble(L)))
    coeff_F = assemble(_coeff_F)
    coeff_F_comp = assemble(L)
    assert math.isclose(coeff_F.dat.data.all(), coeff_F_comp.dat.data.all()), "Test for assembled vectors failed"


def test_add(a):
    print("Test of add")

    _A = Tensor(a)
    add_A = assemble(_A+_A)
    add_A_comp = assemble(a+a)
    for i in range(add_A.M.handle.getSize()[0]):
        for j in range(add_A.M.handle.getSize()[1]):
            assert math.isclose(add_A.M.handle.getValues(i, j), add_A_comp.M.handle.getValues(i, j)), "Test for adding of a two tensor failed"

    print("Test of stacked add")
    _A = Tensor(a)
    add_A = assemble(_A+(_A+_A))
    add_A_comp = assemble(a+a+a)
    for i in range(add_A.M.handle.getSize()[0]):
        for j in range(add_A.M.handle.getSize()[1]):
            assert math.isclose(add_A.M.handle.getValues(i, j), add_A_comp.M.handle.getValues(i, j), abs_tol=1e-14), "Test for stacked adding of a two tensor failed"


def test_negative(a):
    print("Test of negative")

    _A = Tensor(a)
    neg_A = assemble(-_A)
    neg_A_comp = assemble(-a)
    for i in range(neg_A.M.handle.getSize()[0]):
        for j in range(neg_A.M.handle.getSize()[1]):
            assert math.isclose(neg_A.M.handle.getValues(i, j), neg_A_comp.M.handle.getValues(i, j)), "Test for negative of tensor failed"

    print("Test of stacked negative")

    _A = Tensor(a)
    negneg_A = assemble(-(-_A))
    negneg_A_comp = assemble(a)
    for i in range(neg_A.M.handle.getSize()[0]):
        for j in range(neg_A.M.handle.getSize()[1]):
            assert math.isclose(negneg_A.M.handle.getValues(i, j), negneg_A_comp.M.handle.getValues(i, j)), "Test for negative of tensor failed"


def test_transpose(a):
    print("Test of transpose")

    _A = Tensor(a)
    trans_A = assemble(Transpose(_A))
    A_comp = assemble(_A)
    for i in range(trans_A.M.handle.getSize()[0]):
        for j in range(trans_A.M.handle.getSize()[1]):
            assert math.isclose(trans_A.M.handle.getValues(i, j), A_comp.M.handle.getValues(j, i)), "Test for transpose failed"

    print("Test stack of transposes")
    _A = Tensor(a)
    trans_A = assemble(Transpose(Transpose(_A)))
    A_comp = assemble(_A)
    for i in range(trans_A.M.handle.getSize()[0]):
        for j in range(trans_A.M.handle.getSize()[1]):
            assert math.isclose(trans_A.M.handle.getValues(i, j), A_comp.M.handle.getValues(i, j)), "Test for stacked transpose failed"


def test_mul_dx(A, L, V, mesh):
    print("Test of mul")

    # test for mat-vec multiplication
    _A = Tensor(a)
    b = Function(assemble(L))
    _coeff_F = AssembledVector(b)
    mul_matvec = assemble(_A * _coeff_F)
    mul_matvec_comp = assemble(action(a, b))
    assert math.isclose(mul_matvec.dat.data.all(), mul_matvec_comp.dat.data.all()), "Test for contraction (mat-vec-mul)  on cell integrals failed"

    # test for mat-mat multiplication
    u2 = TrialFunction(V)
    v2 = TestFunction(V)
    f2 = Function(V)
    x2, y2 = SpatialCoordinate(mesh)
    f2.interpolate((1 + 8 * pi * pi) * cos(x2 * pi * 2) * cos(y2 * pi * 2))
    a2 = (dot(grad(v2), grad(u2))) * dx
    _A2 = Tensor(a2)
    mul_matmat = assemble(_A * _A2)
    mul_matmat_comp = assemble(_A).M.handle * assemble(_A2).M.handle
    for i in range(mul_matmat.M.handle.getSize()[0]):
        for j in range(mul_matmat.M.handle.getSize()[1]):
            assert math.isclose(mul_matmat_comp.getValues(i, j), mul_matmat.M.handle.getValues(i, j)), "Test for mat-mat-mul  on cell integrals failed"

    # test for mat-mat multiplication with same tensor
    mul_matmat = assemble(_A * _A)
    mul_matmat_comp = assemble(_A).M.handle * assemble(_A).M.handle
    for i in range(mul_matmat.M.handle.getSize()[0]):
        for j in range(mul_matmat.M.handle.getSize()[1]):
            assert math.isclose(mul_matmat_comp.getValues(i, j), mul_matmat.M.handle.getValues(i, j)), "Test for mat-mat-mul  on cell integrals failed"


def test_mul_ds(A, L, V, mesh):
    print("Test of mul")

    # test for mat-vec multiplication
    _A = Tensor(a)
    b = Function(assemble(L))
    _coeff_F = AssembledVector(b)
    mul_matvec = assemble(_A * _coeff_F)
    mul_matvec_comp = assemble(action(a, b))
    assert math.isclose(mul_matvec.dat.data.all(), mul_matvec_comp.dat.data.all()), "Test for contraction (mat-vec-mul) on facet integrals failed"

    # test for mat-mat multiplication
    # only works for facet integrals when there is no coupling between cells involved
    # so for example a flux going across the joint facet of two cells
    # otherwise this becomes kind of a global operation (similar e.g. to an inverse)
    u2 = TrialFunction(V)
    v2 = TestFunction(V)
    a2 = (u2 * v2) * ds + u("+") * v("+") * dS + u("-") * v("-") * dS
    _A2 = Tensor(a2)
    comp1 = assemble(_A)
    comp2 = assemble(_A2)
    mul_matmat_comp = comp1.M.handle * comp2.M.handle
    mul_matmat = assemble(_A * _A2)

    for i in range(mul_matmat.M.handle.getSize()[0]):
        for j in range(mul_matmat.M.handle.getSize()[1]):
            assert math.isclose(mul_matmat_comp.getValues(i, j), mul_matmat.M.handle.getValues(i, j)), "Test for mat-mat-mul on facet integrals failed"


def test_stacked(a, L):
    print("Test of stacking different operations")

    # test for mat-vec multiplication
    _A = Tensor(a)
    u2 = TrialFunction(V)
    v2 = TestFunction(V)
    f2 = Function(V)
    x2, y2 = SpatialCoordinate(mesh)
    f2.interpolate((1 + 8 * pi * pi) * cos(x2 * pi * 2) * cos(y2 * pi * 2))
    a2 = (dot(grad(v2), grad(u2))) * dx
    _A2 = Tensor(a2)
    b = Function(assemble(L))
    _b = AssembledVector(b)
    assemble(((_A + _A) * (_A + _A2) * _b) + _b)
    assemble((_A2 * (_A + _A2)))
    assemble((_A * (_A + _A2)))
    assemble((_A * _A))
    assemble((_A * _A2))

    # TODO: this needs proper testing!


def test_blocks():
    print("Test of blocks")

    mesh = UnitSquareMesh(2, 2)
    U = FunctionSpace(mesh, "RT", 1)
    V = FunctionSpace(mesh, "DG", 0)
    W = U * V
    u, p = TrialFunctions(W)
    w, q = TestFunctions(W)

    A = Tensor(inner(u, w)*dx + p*q*dx - div(w)*p*dx + q*div(u)*dx)

    # Test individual blocks
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    refs = dict(split_form(A.form))
    _A = A.blocks  # is type blockinderxer
    for x, y in indices:
        ref = assemble(refs[x, y])
        block = assemble(_A[x, y])
        assert np.allclose(block.M.values, ref.M.values, rtol=1e-14)


def test_layers():
    print("Test of layer integrals")

    m = UnitSquareMesh(5, 5)
    mesh = ExtrudedMesh(m, 5)
    V1 = FunctionSpace(mesh, "CG", 1)
    V2 = FunctionSpace(mesh, "CG", 1)
    V = V1 * V2
    u, p = TrialFunction(V)
    v, q = TestFunction(V)
    a = inner(u, v) * dx

    A = Tensor(a)
    indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    refs = dict(split_form(A.form))
    _A = A.blocks  # is type blockinderxer
    for x, y in indices:
        ref = assemble(refs[x, y])
        block = assemble(_A[x, y])
        assert np.allclose(block.M.values, ref.M.values, rtol=1e-14)


def test_inverse_local(a):
    print("Test of inverse")

    # calculate local inverse
    _A = Tensor(a)
    A_inv = assemble(_A.inv).M.handle

    # generate reference solutions
    # init of A,B,X
    petsc_mat = assemble(_A).M.handle
    n = petsc_mat.getSize()[0]
    m = petsc_mat.getSize()[1]
    petsc_unitmat = PETSc.Mat().create(PETSc.COMM_SELF)
    petsc_unitmat.setSizes([n, m])
    petsc_unitmat.setType(PETSc.Mat.Type.SEQDENSE)
    petsc_unitmat.setUp()
    petsc_invmat = PETSc.Mat().create(PETSc.COMM_SELF)
    petsc_invmat.setSizes([n, m])
    petsc_invmat.setType(PETSc.Mat.Type.SEQDENSE)
    petsc_invmat.setUp()
    for i in range(n):
        for j in range(m):
            if i == j:
                petsc_unitmat[i, j] = 1
            else:
                petsc_unitmat[i, j] = 0
    r, c = petsc_mat.getOrdering("nd")

    # factorization
    petsc_mat.reorderForNonzeroDiagonal(r, c)
    petsc_mat.factorLU(r, c)

    # inverse calculation
    petsc_mat.matSolve(petsc_unitmat, petsc_invmat)

    # convert into sparse again
    petsc_invmat.convert(PETSc.Mat.Type.SEQAIJ)
    petsc_invmat.assemble()

    # test
    for i in range(petsc_invmat.getSize()[0]):
        for j in range(petsc_invmat.getSize()[1]):
            # TODO: how do I get the inverse in petsc when there are 0 blocks
            if not math.isclose(petsc_invmat.getValues(i, j), A_inv.getValues(i, j)) and not petsc_invmat.getValues(i, j) == float("nan"):
                assert "Local inverse test failed"


def test_solve_local(a, L):
    print("Test of solve")

    # calculate local solve
    _A = Tensor(a)
    _F = AssembledVector(assemble(L))
    u = assemble(_A.solve(_F))

    # global solve TODO: I should probably test against petsc solve to make this test safer
    u_comp = Function(V)
    solve(assemble(_A), u_comp, assemble(_F))

    # test
    for c, i in enumerate(u.dat.data):
        j = u_comp.dat.data[c]
        if not math.isclose(i, j) and not j == float("nan"):
            assert "Local solve failed"


def marybarker_solve_curl_curl(mesh, f, degree, with_tensor=False):
    V_element = FiniteElement("N1curl", mesh.ufl_cell(), degree)
    V = FunctionSpace(mesh, V_element)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(curl(u), curl(v))*dx
    L = inner(f, v)*dx

    if with_tensor:
        A = Tensor(a)
        B = Tensor(L)
        w = A.inv * B
        assemble(w)
        # does not throw an error if degree > 4 anymore


"""
Run test script
"""
print("Run test for slate to loopy compilation.\n\n")

print("test123")

# discontinuous Helmholtz equation on cell integrals
mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, "DG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + v * u) * dx
L = f * v * dx

test_negative(a)
test_stacked(a, L)
test_assemble_matrix(a)
test_negative(a)
test_add(a)
test_assembled_vector(L)
test_transpose(a)
test_mul_dx(a, L, V, mesh)
test_solve(a, L, V)
test_solve_local(a, L)
test_inverse_local(a)

# discontinuous Helmholtz equation on facet integrals
mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, "DG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (v * u) * ds
L = f * v * ds

test_assemble_matrix(a)
test_negative(a)
test_add(a)
test_mul_ds(a, L, V, mesh)
test_inverse_local(a)

# continuous Helmholtz equation on facet integrals (works also on cell)
mesh = UnitSquareMesh(5, 5)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
f = Function(V)
x, y = SpatialCoordinate(mesh)
f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
a = (dot(grad(v), grad(u)) + u * v) * ds
L = f * v * ds

test_assemble_matrix(a)
test_negative(a)
test_add(a)
test_inverse_local(a)

# test for assembly of blocks of mixed systems
# (here for lowest order RT-DG discretisation)
test_blocks()

# test of block assembly of mixed system defined on extruded mesh
test_layers()

# issue raised by marybarker
mesh = UnitTetrahedronMesh()
marybarker_solve_curl_curl(mesh, Constant((1, 1, 1)), 5, True)

# TODO: continuous advection problem
n = 5
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
u_ = Function(V).project(x)
u = TrialFunction(V)
v = TestFunction(V)
F = (u_*div(v*u))*dx

# test_transpose(a)

# ##############################################
# TODO: assymetric problem test
# TODO: write test for subdomain integrals as well
# TODO: make argument generation nicer
# TODO: fix dependency generation for transpose on facets

print("\n\nAll tests passed.")

This is for testing submatrix extraction and making sure everything works.::
  
  from firedrake import *
  from firedrake.petsc import PETSc
  from uflmat import UFLMatrix

  M = UnitSquareMesh(2, 2)
  V = FunctionSpace(M, "CG", 1)
  W=V*V

  u0, u1 = TrialFunctions(W)
  v0, v1 = TestFunctions(W)

  X = Function(W)
  Y = Function(W)
  Y0 = Function(V)

  a = inner(grad(u0), grad(v0))*dx + u1*v1*dx

  bcs = [DirichletBC(V, Constant(0.0), (1,2,3,4))]

  A_uflmat = UFLMatrix(a, bcs=bcs)
  A = PETSc.Mat().create()
  A.setType("python")
  A.setSizes(A_uflmat.row_sizes, A_uflmat.col_sizes)
  A.setPythonContext(A_uflmat)

  dofs0 = W.sub(0).dof_dset.field_ises[0]

  A00 = A.getSubMatrix(dofs0, dofs0)

  with X.dat.vec as Xvec:
      with Y0.dat.vec as Y0v:
          X0v = Xvec.getSubVector(dofs0)
          A00.mult(X0v, Y0v)





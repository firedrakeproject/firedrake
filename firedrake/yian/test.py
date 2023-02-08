import numpy as np
from firedrake import *
from firedrake.petsc import PETSc
from math import pi
try:
    from slepc4py import SLEPc  
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)



# Define a mesh and function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, "CG", 1) # linear polynomials therefore 1 in third arg
bc = DirichletBC(V, 0.0, "on_boundary") # homogenous Dirichlet boundaries   

# Define the trial and test functions
u, v = TrialFunction(V), TestFunction(V)

lam = Constant(1)
# Define the weak form of the linear equation

a = (inner(grad(u), grad(v)) + u*v )* dx
#f = Function(V)
# L = inner(f, v) * dx FE PROBLEM
#L = inner(v, Constant(1)) * dxs
L = lam * inner(u,  v) * dx

#x, y = SpatialCoordinate(mesh)
#f.interpolate(x*y)

# Assemble the system matrix and right-hand side vector
A = assemble(a).M.handle
b = assemble(L, bcs=bc).M.handle # M.handle setting it to be PETSC objects

#options
opts = PETSc.Options()
opts.setValue("eps_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_largest_imaginary", None)
opts.setValue("eps_tol", 1e-10)

eigensolver = SLEPc.EPS().create(comm=COMM_WORLD)
num_evals = 5
eigensolver.setDimensions(num_evals)
eigensolver.setOperators(A, b)
#eigensolver.setFromOptions()
eigensolver.solve()


nconv = eigensolver.getConverged()
#print("Number of converged eigenpairs %d" % nconv)
eigenmodes_real = Function(V)
eigenmodes_imag = Function(V)
# vr, wr = A.getVecs()  # PetSC vector object
# vi, wi = A.getVecs()
# eigenmodes_real.vector()[:], eigenmodes_imag.vector()[:] = vr, vi # Firedrake Vector object
# print(eigenmodes_imag.vector())
# if nconv > 0 and None:

    # vr, wr = A.getVecs()
    # n = 0

    # vi, wi = A.getVecs()
    # evals = np.zeros(nconv, dtype=complex)
    # evec_arr = np.zeros(nconv, dtype=complex)

    # for i in range(nconv):
    #     eval = eigensolver.getEigenvalue(i)
    #     evals[i] = eval
    #     print(type(vr))
    #     k = eigensolver.getEigenpair(i, vr, vi) # same as eval

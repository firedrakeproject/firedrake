from firedrake import *
from firedrake.petsc import PETSc
from slepc4py import SLEPc

mesh = UnitSquareMesh(64, 64)
print("Meshed")
V = FunctionSpace(mesh, "CG", 2)
u = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(u), grad(v))*dx
m = inner(u,v)*dx

print("Problem Seted Up !")

sol = Function(V)
bc = DirichletBC(V, 0.0, [1,2,3,4], weight=1.)
print("Assembling ...")
A = assemble (a, bcs=bc)# There is an apperent difference in imposing the boudnary
bc = DirichletBC(V, 0.0, [1,2,3,4], weight=0.)
M = assemble (m, bcs=bc)# condition using DirichletBC or a penalty method.
Asc, Msc = A.M.handle, M.M.handle
print("Assembled !")
print("Solving ...");

def monitor(eps, its, nconv, eig, err):
        print("[It. {}] err: {}".format(its,err[-1]));
opts = PETSc.Options()
opts.setValue("monitor", None)
E = SLEPc.EPS().create()
E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
E.setProblemType(SLEPc.EPS.ProblemType.GHEP);
E.setDimensions(5,SLEPc.DECIDE);
E.setOperators(Asc,Msc)
E.setMonitor(monitor)
ST = E.getST();
ST.setType(SLEPc.ST.Type.SINVERT)
PC = ST.getKSP().getPC();
PC.setType("lu");
PC.setFactorSolverType("mumps");
E.setST(ST)
print("Vecotrized")
E.solve();
nconv = E.getConverged()
print("Number of converged eigenvalues is: {}".format(nconv))
for k in range(nconv):
    vr, vi = Asc.getVecs()
    lam = E.getEigenpair(k, vr, vi)
    print("[{}] Eigenvalue: {}".format(k,lam.real))
print("End")

from firedrake import *
import numpy as np

def projection(mesh):
    # Define the function spaces for testing (Pk CG, and Pk Trace)
    k = 1
    CG = FunctionSpace(mesh, "CG", k)
    TraceSpace = FunctionSpace(mesh, "HDiv Trace", k)

    # Define the space where we interpolate f
    hdeg = k+1
    HCG = FunctionSpace(mesh, "CG", hdeg)

    W = CG*TraceSpace

    # Define trial and test functions
    u, lambdar = TrialFunctions(W)
    v, gammar = TestFunctions(W)

    # Interpolate smooth function into the CG space
    f = Function(HCG)
    #x = SpatialCoordinate(mesh)
    f.interpolate(Expression("cos(x[0]*pi*2)*cos(x[1]*pi*2)"))

    # Construct the bilinear form
    a_dx = u*v*dx
    a_dS = lambdar*gammar*ds + avg(lambdar)*avg(gammar)*dS
    a = a_dx + a_dS

    # Construct the linear form
    L_dx = f*v*dx
    L_dS = f*gammar*ds + avg(f)*avg(gammar)*dS
    L = L_dx + L_dS

    # Solution
    w = Function(W)

    solve(a == L, w, solver_parameters={'ksp_rtol': 1e-14})
    u_h, tr_h = w.split()

    #uherr = sqrt(assemble((u_h-f)*(u_h-f)*dx))
    uherr = sqrt(assemble((u_h - f)*(u_h - f)*ds + (avg(u_h) - avg(f))*(avg(u_h) - avg(f))*dS))
    trerr = sqrt(assemble((tr_h - f)*(tr_h - f)*ds + (avg(tr_h) - avg(f))*(avg(tr_h) - avg(f))*dS))
    err = sqrt(assemble((u_h - tr_h)*(u_h - tr_h)*ds + (avg(u_h) - avg(tr_h))*(avg(u_h) - avg(tr_h))*dS))

    return uherr, trerr, err

uherr = []
trerr = []
err = []
# Create a mesh
for r in range(8):
    res = 2**r
    mesh = UnitSquareMesh(res, res)

    e = projection(mesh)
    uherr.append(e[0])
    trerr.append(e[1])
    err.append(e[2])

uherr = np.array(uherr)
trerr = np.array(trerr)
err = np.array(err)

print np.log(uherr[1:]/uherr[:-1])/np.log(0.5)
print np.log(trerr[1:]/trerr[:-1])/np.log(0.5)

#print uherr
#print trerr
print "The error in the trace-norm: ", err

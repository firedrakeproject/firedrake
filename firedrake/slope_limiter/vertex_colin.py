from __future__ import absolute_import
from firedrake import *
from math import ceil
import numpy as np

n = 100
dx0 = 1.0/n

mesh = PeriodicUnitIntervalMesh(30)

V0 = FunctionSpace(mesh,"DG",0) #used for storing alpha values
Vdg = FunctionSpace(mesh,"DG",2) #used for limiting
Vdg1 = FunctionSpace(mesh,"DG",1) #used for limiting
Vcg1 = FunctionSpace(mesh, "CG", 1) #used for building maxes and mins
Vcg = FunctionSpace(mesh, "CG", 2) #used for building maxes and mins


u = TrialFunction(Vdg)

#1D spaces in horizontal
# IUh0 = FiniteElement("CG", "interval", 2)
# IUh1 = FiniteElement("DG", "interval", 1)

#1D spaces in vertical
# IUv0d = FiniteElement("DG", "interval", 2)
# IUv0 = FiniteElement("CG", "interval", 2)
# IUv1 = FiniteElement("DG", "interval", 1)
# 
# IUhdg0 = FiniteElement("DG", "interval", 0)
# IUvcg1 = FiniteElement("CG", "interval", 1)

#space for derivative max and mins
# Vmin_elt = OuterProductElement(IUh1,IUvcg1)

#Taylor spaces
IT1 = FiniteElement("TDG", "interval", 1)
IT2 = FiniteElement("TDG", "interval", 2)

#Taylor spaces
# VTdg_elt = OuterProductElement(IUh1, IT2)
VTdg = FunctionSpace(mesh,"Discontinuous Taylor", 2)
# VTdg1_elt = OuterProductElement(IUh1, IT1)
VTdg1= FunctionSpace(mesh,"Discontinuous Taylor", 1)

#RT1 Space

V1f = VectorFunctionSpace(mesh,"CG",1)


umax_z = Function(Vcg1)
umin_z = Function(Vcg1)
umax = Function(Vcg1)
umin = Function(Vcg1)
ubar = Function(V0)
ubar2 = Function(V0)
# deltau = Function(Vdg)

#maps from P1dg x Tdg2 (2x3) to P1dg x Tdg1 (2x2)
#### NEEDS TO MAP FROM TDG2 TO TDG1 ####
remove_higher_kernel = """
qout[0][0] = qin[0][0];
qout[1][0] = qin[1][0];
qout[2][0] = qin[3][0];
qout[3][0] = qin[4][0];
"""

#maps from P1dg x Tdg2 (2x3) to P1dg x Tdg1 (2x2)
#### NEEDS TO get derivatives and averages FROM TDG2 TO TDG1 ####
get_derivative_kernel = """
double dz[2];
dz[0] = Z[1][0] - Z[0][0];
dz[1] = Z[3][0] - Z[2][0];
qder[1][0] = (q[1][0] + q[2][0]/2)/dz[0];
qder[0][0] = (q[1][0] - q[2][0]/2)/dz[0];
qder[3][0] = (q[4][0] + q[5][0]/2)/dz[1];
qder[2][0] = (q[4][0] - q[5][0]/2)/dz[1];
qderbar[1][0] = q[1][0];
qderbar[0][0] = q[1][0];
qderbar[3][0] = q[4][0];
qderbar[2][0] = q[4][0];
"""

#maps from P1dg x Tdg1 (2x2) to P1dg x Tdg1 (2x2)
taylor2lagrange_kernel = """
qL[0][0] = qT[0][0] - qT[1][0]/2;
qL[1][0] = qT[0][0] + qT[1][0]/2;
qL[2][0] = qT[2][0] - qT[3][0]/2;
qL[3][0] = qT[2][0] + qT[3][0]/2;
"""

max_kernel = """
            for(int i=0;i<maxq.dofs;i++){
            maxq[i][0] = fmax(maxq[i][0],qin[0][0]);
            }"""

min_kernel = """
            for(int i=0;i<minq.dofs;i++){
            minq[i][0] = fmin(minq[i][0],qin[0][0]);
            }"""

max_kernel2 = """
            for(int i=0;i<maxq.dofs;i++){
            maxq[i][0] = fmax(maxq[i][0],qin[i][0]);
            }"""

min_kernel2 = """
            for(int i=0;i<minq.dofs;i++){
            minq[i][0] = fmin(minq[i][0],qin[i][0]);
            }"""


#inputs qderbar,q,maxq,minq, alpha
alpha_limit_der_kernel = """
double alpha = 1.0;
double qavg;
for (int i=0; i<q.dofs; i++) {
    qavg = qderbar[i][0];
    if (q[i][0] > qavg)
        alpha = fmin(alpha, fmin(1, (qmax[i][0] - qavg)/(q[i][0] - qavg)));
    else if (q[i][0] < qavg)
        alpha = fmin(alpha, fmin(1, (qavg - qmin[i][0])/(qavg - q[i][0])));
}
alpha_out[0][0] = alpha;
"""

#inputs qbar,q,maxq,minq, alpha
alpha_limit_kernel = """
    double alpha = 1.0;
    double qavg = qbar[0][0];
    for (int i=0; i<q.dofs; i++) {
    if (q[i][0] > qavg)
        alpha = fmin(alpha, fmin(1, (qmax[i][0] - qavg)/(q[i][0] - qavg)));
    else if (q[i][0] < qavg)
        alpha = fmin(alpha, fmin(1, (qavg - qmin[i][0])/(qavg - q[i][0])));
    }
    alpha_out[0][0] = alpha;
    """


max2_kernel = """
alpha[0][0] = fmax(alpha[0][0],alpha1[0][0]);
"""

fcr_kernel = """
double alpha=1.0;
double P,Q;

//Compute element-wise max and mins
double umax0 = -1e10, umin0 = 1e10;
for(int i=0; i<umax.dofs; i++){
   umax0 = fmax(umax0,umax[i][0]);
   umin0 = fmin(umin0,umin[i][0]);
}

for(int i=0; i<f.dofs; i++){
   if(f[i][0]>0){
      P = fmax(0.,f[i][0]);
      Q = m[i][0]*(umax0 - uL[i][0]);
      alpha = fmin(alpha,fmin(1.0,Q/P));
   } else if(f[i][0]<0) {
      P = fmin(0.,f[i][0]);
      Q = m[i][0]*(umin0 - uL[i][0]);
      alpha = fmin(alpha,fmin(1.0,Q/P));
   }
}
for(int i=0; i<f.dofs; i++){
   mu[i][0] += m[i][0]*uL[i][0] + alpha*f[i][0];
}
"""

remap_kernel ="""
u_out[0][0] = u_in[0][0];
u_out[1][0] = u_in[1][0];
u_out[2][0] = u_in[3][0];
u_out[3][0] = u_in[4][0];
"""

#Function to compute Taylor basis in
u_Taylor = Function(VTdg)
u_Taylor1 = Function(VTdg1)

#function to project solution down to P1dg
u_P1dg = Function(Vdg1)
ubar_P1dg = Function(Vdg1)

# f_loc = Function(Vdg)
#function to store limiter values in
alpha0 = Function(V0)
alpha1 = Function(V0)

Z = Function(Vcg1).interpolate(Expression("x[1]"))

#LIMITERS
def limit_slope(u):
    #Transform to Taylor basis
    u_Taylor.project(u,solver_parameters={'ksp_type':'preonly',
                                          'pc_type':'lu'})

    #Extract the derivative
    par_loop(get_derivative_kernel, dx,
             {"Z":(Z,READ),
              "q":(u_Taylor,READ),
              "qder":(u_P1dg,WRITE),
              "qderbar":(ubar_P1dg,WRITE)})

    umax_z.assign(-1.0e10)
    umin_z.assign(1.0e10)

    par_loop(max_kernel2, dx,
             {"maxq":(umax_z,RW),
              "qin":(ubar_P1dg,READ)})    

    par_loop(min_kernel2, dx,
             {"minq":(umin_z,RW),
              "qin":(ubar_P1dg,READ)})

    par_loop(alpha_limit_der_kernel,dx,
             {"qderbar": (ubar_P1dg,READ),
              "q": (u_P1dg,READ),
              "alpha_out": (alpha1,WRITE),
              "qmax": (umax_z,READ), 
              "qmin": (umin_z,READ)})

    #isolate the linear part
    par_loop(remove_higher_kernel, dx,
             {"qin":(u_Taylor,READ),
              "qout":(u_Taylor1,WRITE)})

    #Remap to DG (don't want to project)
    #but do want to make conservative

    par_loop(taylor2lagrange_kernel, dx,
             {"qT":(u_Taylor1,READ),
              "qL":(u_P1dg,WRITE)})

    #conservative correction
    ubar.project(u)
    ubar2.project(u_P1dg)
    u_P1dg.project(u_P1dg-ubar2+ubar)

    #computing limiting factors alpha_0
    umax.assign(-1.0e10)
    umin.assign(1.0e10)

    #ubar.project(u_P1dg)

    par_loop(max_kernel, dx,
             {"maxq":(umax,RW),
              "qin":(ubar,READ)})

    par_loop(min_kernel, dx,
             {"minq":(umin,RW),
              "qin":(ubar,READ)})

    par_loop(alpha_limit_kernel,dx,
             {"qbar": (ubar,READ),
              "q": (u_P1dg,READ),
              "alpha_out": (alpha0,WRITE),
              "qmax": (umax,READ), 
              "qmin": (umin,READ)})    
    
    #Apply max 
    #par_loop(max2_kernel,dx,
    #         {"alpha":(alpha0,RW),
    #          "alpha1":(alpha1,READ)})

    #reconstruct
    # u -> ubar + max(alpha_0,alpha_1)*(u_1-ubar) + alpha_1*(u - u_1)
    u.project(ubar + alpha0*(u_P1dg-ubar) + alpha1*(u - u_P1dg),
              solver_parameters={'ksp_type':'preonly',
                                 'pc_type':'lu'})

#Test and Trial functions

phi = TestFunction(Vdg)
u = TrialFunction(Vdg)
phiCG = TestFunction(Vcg)
ML = assemble(phiCG*dx)
ML_dg = assemble(phi*dx)
f_loc2 = Function(Vdg)

VField = "rotation"

if(VField == "rotation"):
#(y,-x) - Solid rotation
    #v = Function(V1).project(Expression(("x[1]-0.5","-(x[0]-0.5)")))
    v = Function(V1f).interpolate(Expression(("x[1]-0.5")))
elif(VField == "rotate-translate"):
#Differential rotation plus translation
    a = 2.5
    Tr = 2*pi
    v = Function(V1).project(Expression(("T-a*2*(0.5-t)*sin(pi*(x[0]-T*t))*cos(pi*x[1])",
                                         "a*2*(0.5-t)*cos(pi*(x[0]-T*t))*sin(pi*x[1])"),a=a,t=0.,T=Tr))

magv = Function(Vcg).project(v[0]*v[0])
cval = (magv.dat.data.max())**0.5
c = Constant(cval)
print "C", cval

Courant = 0.30 #=u*dt/dx
Dt = dx0*Courant/cval
T = 2*pi
n = ceil(T/Dt)
Dt = T/n
dt = Constant(Dt)

# ( dot(v, n) + |dot(v, n)| )/2.0
n = FacetNormal(mesh)
vn = 0.5*(dot(v, n) + abs(dot(v, n)))

# advection equation (needs adapting to 2D)
# advection equation
a_mass = phi*u*dx
a_int = dot(div(phi*v), -u)*dx
a_flux = ( dot(jump(phi), vn('+')*u('+') - vn('-')*u('-')) )*(dS_h + dS_v)
arhs = a_mass-dt*(a_int + a_flux)


t = 0.0
u0 = Function(Vdg)
u1 = Function(Vdg)

# Define u to hold solution
u = Function(Vcg).interpolate(Expression("(x[0] < 0.5) ? 1 : 0"))

# solver
u1problem = LinearVariationalProblem(a_mass, action(arhs,u1), du1)
u1solver = LinearVariationalSolver(u1problem, parameters=
                                   {'ksp_type': 'preonly',
                                    'pc_type':'lu'})



while(t<T-0.5*Dt):
    t += Dt
    print t

    u0.project(u,solver_parameters={
        'ksp_type':'preonly',
        'pc_type':'lu'
    }) #a mathematical no-op
    limit_slope(u0)
    u1.assign(u0)
    limit_slope(u1)
    u1solver.solve()
    u1.assign(du1)
    limit_slope(u1)
    u1solver.solve()
    u1.assign(0.75*u0 + 0.25*du1)
    limit_slope(u1)
    u1solver.solve()
    u1.assign(u0/3.0 + 2.0*du1/3.0)
    limit_slope(u1)

    #High order solution
    u1.dat.data
    u.project(u1,solver_parameters={'ksp_type':'preonly',
                                               'pc_type':'lu'})
    u.dat.data
        
def test_step_function_loop(mesh, degree, iterations=100):
    # test function space
    v = FunctionSpace(mesh, "DG", degree)
    m = VectorFunctionSpace(mesh, "CG", 1)

    # advecting velocity
    if m.shape == (1, ):
        u0 = as_vector([1])
    else:
        u0 = as_vector([1, 0])
    u = Function(m).interpolate(u0)

    # advection problem
    dt = 1. / iterations
    phi = TestFunction(v)
    D = TrialFunction(v)
    n = FacetNormal(mesh)
    un = 0.5 * (dot(u, n) + abs(dot(u, n)))  # upwind value

    a_mass = phi * D * dx
    a_int = dot(grad(phi), -u * D) * dx
    a_flux = dot(jump(phi), un('+') * D('+') - un('-') * D('-')) * dS
    arhs = a_mass - dt * (a_int + a_flux)

    dD1 = Function(v)
    D1 = Function(v)
    x = SpatialCoordinate(mesh)

    # Initial Conditions
    D0 = conditional(x[0] < 0.5, 1., 0.)

    D = Function(v).interpolate(D0)
    D1.assign(D)
    D1_old = Function(D1)

    t = 0.0
    T = iterations * dt
    problem = LinearVariationalProblem(a_mass, action(arhs, D1), dD1)
    solver = LinearVariationalSolver(problem, solver_parameters={'ksp_type': 'cg'})

    # Make slope limiter
    limit_slope(D1)

    while t < (T - dt / 2):
        D1.assign(D)
        limit_slope(D1)
        solver.solve()
        D1.assign(dD1)
        limit_slope(D1)

        solver.solve()
        D1.assign(0.75 * D + 0.25 * dD1)
        limit_slope(D1)
        solver.solve()
        D.assign((1.0 / 3.0) * D + (2.0 / 3.0) * dD1)
        limit_slope(D1)

        t += dt

    diff = assemble((D1 - D1_old) ** 2 * dx) ** 0.5
    print "Error:", diff
    max = np.max(D1.dat.data_ro)
    min = np.min(D1.dat.data_ro)
    print "Max:", max, "Min:", min
    assert max <= 1.0 + 1e-2, "Failed by exceeding max values"
    assert min >= 0.0 - 1e-2, "Failed by exceeding min values"


# mesh = PeriodicUnitSquareMesh(30, 30)
# test_step_function_loop(mesh, 2)

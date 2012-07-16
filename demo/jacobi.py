from __future__ import print_function
from pyop2 import op2
import numpy as np
from math import sqrt

op2.init(backend='sequential')

NN = 6
NITER = 2

nnode = (NN-1)**2
nedge = nnode + 4*(NN-1)*(NN-2)

pp = np.zeros((2*nedge,),dtype=np.int)

A = np.zeros((nedge,), dtype=np.float64)
r = np.zeros((nnode,), dtype=np.float64)
u = np.zeros((nnode,), dtype=np.float64)
du = np.zeros((nnode,), dtype=np.float64)

e = 0

for i in xrange(1, NN):
    for j in xrange(1, NN):
        n = i-1 + (j-1)*(NN-1)
        pp[2*e] = n
        pp[2 * e + 1] = n
        A[e] = -1
        e += 1
        for p in xrange(0, 4):
            i2 = i
            j2 = j
            if p == 0:
                i2 += -1
            if p == 1:
                i2 += +1
            if p == 2:
                j2 += -1
            if p == 3:
                j2 += +1

            if i2 == 0 or i2 == NN or j2 == 0 or j2 == NN:
                r[n] += 0.25
            else:
                pp[2 * e] = n
                pp[2 * e + 1] = i2 - 1 + (j2 - 1)*(NN - 1)
                A[e] = 0.25
                e += 1


nodes = op2.Set(nnode, "nodes")
edges = op2.Set(nedge, "edges")
ppedge = op2.Map(edges, nodes, 2, pp, "ppedge")

p_A = op2.Dat(edges, 1, data=A, name="p_A")
p_r = op2.Dat(nodes, 1, data=r, name="p_r")
p_u = op2.Dat(nodes, 1, data=u, name="p_u")
p_du = op2.Dat(nodes, 1, data=du, name="p_du")

alpha = op2.Const(1, data=1.0, name="alpha")

beta = op2.Global(1, data=1.0, name="beta")
res = op2.Kernel("""void res(double *A, double *u, double *du, const double *beta){
  *du += (*beta)*(*A)*(*u);
}""", "res")

update = op2.Kernel("""void update(double *r, double *du, double *u, double *u_sum, double *u_max){
  *u += *du + alpha * (*r);
  *du = 0.0f;
  *u_sum += (*u)*(*u);
  *u_max = *u_max > *u ? *u_max : *u;
}""", "update")


for iter in xrange(0, NITER):
    op2.par_loop(res, edges,
                 p_A(op2.IdentityMap, op2.READ),
                 p_u(ppedge(1), op2.READ),
                 p_du(ppedge(0), op2.INC),
                 beta(op2.READ))
    u_sum = op2.Global(1, data=0.0, name="u_sum")
    u_max = op2.Global(1, data=0.0, name="u_max")

    op2.par_loop(update, nodes,
                 p_r(op2.IdentityMap, op2.READ),
                 p_du(op2.IdentityMap, op2.RW),
                 p_u(op2.IdentityMap, op2.INC),
                 u_sum(op2.INC),
                 u_max(op2.MAX))

    print( " u max/rms = %f %f \n" % (u_max.data[0], sqrt(u_sum.data/nnode)))



print("\nResults after %d iterations\n" % NITER)
for j in range(NN-1, 0, -1):
    for i in range(1, NN):
        print(" %7.4f" % p_u.data[i-1 + (j-1)*(NN-1)], end='')
    print("")
print("")


op2.exit()

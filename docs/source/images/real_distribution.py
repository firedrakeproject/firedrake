from firedrake import *
import pylab

m = UnitSquareMesh(5, 5)
V = FunctionSpace(m, 'CG', 1)
R = FunctionSpace(m, 'R', 0)
W = V * R
u, r = TrialFunctions(W)
v, s = TestFunctions(W)

a = inner(grad(u), grad(v))*dx + u*s*dx + v*r*dx
M = assemble(a)

A = M.M.blocks[0][0].values

L = A.shape[0]

for i_ in range(3):
    for i in range(i_*L/3, (i_+1)*L/3):
        for j in range(L):
            if A[i, j]:
                pylab.plot(j, L-i, '%ss' % "cmy"[i_], markersize=6)
            pylab.plot(L, L - i, '%ss' % "cmy"[i_], markersize=6)
            pylab.plot(i, 0, '%ss' % "cmy"[i_], markersize=6)
pylab.axis([-1, L+1, -1, L+1])
pylab.axis("off")
pylab.savefig("real_distribution.png")

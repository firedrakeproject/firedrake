from firedrake import *
from firedrake.petsc import PETSc
import ufl
mesh = UnitSquareMesh(20, 20)

mh = MeshHierarchy(mesh, 6)

fh = FunctionSpaceHierarchy(mh, 'CG', 1)


def restriction(level):
    "Create restriction from level to level - 1"
    mat = PETSc.Mat().create()
    mat.setType(mat.Type.PYTHON)
    nrow = fh[level-1].dof_dset.size * fh[level-1].dof_dset.cdim
    ncol = fh[level].dof_dset.size * fh[level].dof_dset.cdim
    mat.setSizes(((nrow, None), (ncol, None)))

    class Restrict(object):

        def __init__(self, fh):
            self.f = FunctionHierarchy(fh)

        def mult(self, mat, x, y, increment=False):
            with self.f[level].dat.vec as v:
                with v as v_, x as x_:
                    v_[:] = x_[:]
            self.f.restrict(level)
            with self.f[level-1].dat.vec_ro as v:
                with v as v_, y as y_:
                    if increment:
                        y_[:] += v_[:]
                    else:
                        y_[:] = v_[:]

        def multAdd(self, mat, x, y, w):
            if w.handle == y.handle:
                self.mult(mat, x, w, increment=True)
            else:
                self.mult(mat, x, w)
                w.array[:] += y.array

    mat.setPythonContext(Restrict(fh))
    mat.setUp()
    return mat


def interpolation(level):
    "Create interpolation from level - 1 to level"
    mat = PETSc.Mat().create()
    mat.setType(mat.Type.PYTHON)
    nrow = fh[level].dof_dset.size * fh[level].dof_dset.cdim
    ncol = fh[level-1].dof_dset.size * fh[level-1].dof_dset.cdim
    mat.setSizes(((nrow, None), (ncol, None)))

    class Interpolate(object):

        def __init__(self, fh):
            self.f = FunctionHierarchy(fh)

        def mult(self, mat, x, y, increment=False):
            with self.f[level-1].dat.vec as v:
                with v as v_, x as x_:
                    v_[:] = x_[:]
            self.f.prolong(level - 1)
            with self.f[level].dat.vec_ro as v:
                with v as v_, y as y_:
                    if increment:
                        y_[:] += v_[:]
                    else:
                        y_[:] = v_[:]

        def multAdd(self, mat, x, y, w):
            if w.handle == y.handle:
                self.mult(mat, x, w, increment=True)
            else:
                self.mult(mat, x, w)
                w.array[:] += y.array

    mat.setPythonContext(Interpolate(fh))
    mat.setUp()
    return mat


class LinearVariationalProblemHierarchy(object):
    def __init__(self, a, L, u, bcs=None, J=None):
        self.a = a
        self.L = L
        self.u = u
        if bcs:
            self.bcs = bcs
        else:
            self.bcs = [[] for _ in self.L]


class LinearVariationalSolverHierarchy(object):
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.bcs = self.problem.bcs
        self.u = self.problem.u
        self.x = FunctionHierarchy(u.function_space())
        self.a = self.problem.a
        self.L = self.problem.L
        self._a = [assemble(a, bcs=bcs) for a, bcs in zip(self.a, self.bcs)]
        self._aP = self._a

        ksp = PETSc.KSP().create()
        ksp.setOperators(self._a[-1].M.handle, self._a[-1].M.handle)
        ksp.setFromOptions()
        pc = ksp.pc

        self.ksp = ksp
        if pc.getType() == pc.Type.MG:
            nlevel = pc.getMGLevels()
            for i in range(1, nlevel):
                lvl = i + (len(self.x) - nlevel)
                pc.setMGRestriction(i, restriction(lvl))
                pc.setMGInterpolation(i, interpolation(lvl))

            for i in range(nlevel):
                lvl = i + (len(self.x) - nlevel)
                for ksp in [pc.getMGSmoother(i)]:
                    assemble(self.a[lvl], tensor=self._a[lvl],
                             bcs=self.bcs[lvl])
                    self._a[lvl] = self._a[lvl]
                    ksp.setOperators(self._a[lvl].M.handle, self._a[lvl].M.handle)

    def solve(self):
        rhs = assemble(self.L[-1])

        u_bc = Function(rhs.function_space())
        for bc in self.bcs[-1]:
            bc.apply(u_bc)

        u_bc.assign(rhs - assemble(action(self.a[-1], u_bc)))
        for bc in self.bcs[-1]:
            bc.apply(u_bc)

        with self.u[-1].dat.vec as u:
            with u_bc.dat.vec_ro as b:
                self.ksp.solve(b, u)

a = []
L = []
bcs = []
f = FunctionHierarchy(fh)
for f_ in f:
    V = f_.function_space()
    v = TestFunction(V)
    u = TrialFunction(V)
    dx = V.mesh()._dx
    a.append((dot(grad(u), grad(v)))*dx)
    f_.interpolate(Expression("32*pi*pi*sin(4*pi*x[0])*sin(4*pi*x[1])"))
    L.append(f_*v*dx)
    bcs.append([DirichletBC(V, 0.0, (1,2,3,4))])

u = FunctionHierarchy(fh)
problem = LinearVariationalProblemHierarchy(a, L, u, bcs=bcs)

solver = LinearVariationalSolverHierarchy(problem)

solver.solve()

exact = Function(fh[-1])
exact.interpolate(Expression("sin(4*pi*x[0])*sin(4*pi*x[1])"))

print norm(assemble(exact - u[-1]))
File('exact.pvd') << exact
File('u.pvd') << u[-1]

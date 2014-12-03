from firedrake import *
from firedrake.petsc import PETSc
import ufl


mesh = UnitSquareMesh(10, 10)

nlevel = 4

mh = MeshHierarchy(mesh, nlevel)

fh = FunctionSpaceHierarchy(mh, 'CG', 2)


def restriction(level, bcs):
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
            for bc in bcs:
                bc.zero(self.f[level-1])
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


def interpolation(level, bcs):
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
            for bc in bcs:
                bc.zero(self.f[level])
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


def injection(level, bcs):
    mat = PETSc.Mat().create()
    mat.setType(mat.Type.PYTHON)
    nrow = fh[level-1].dof_dset.size * fh[level-1].dof_dset.cdim
    ncol = fh[level].dof_dset.size * fh[level].dof_dset.cdim
    mat.setSizes(((nrow, None), (ncol, None)))

    class Inject(object):

        def __init__(self, fh):
            self.f = FunctionHierarchy(fh)

        def mult(self, mat, x, y, increment=False):
            with self.f[level].dat.vec as v:
                with v as v_, x as x_:
                    v_[:] = x_[:]
            self.f.inject(level)
            for bc in bcs:
                bc.apply(self.f[level-1])
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

    mat.setPythonContext(Inject(fh))
    mat.setUp()
    return mat


class NonlinearVariationalProblemHierarchy(object):
    def __init__(self, F, u, bcs=None, J=None):
        self.F = F
        self.J = J or [derivative(f, u_) for f, u_ in zip(F, u)]
        self.u = u
        self.bcs = bcs


class NonlinearVariationalSolverHierarchy(object):
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.bcs = self.problem.bcs
        self.u = self.problem.u
        self.x = FunctionHierarchy(u.function_space())
        self.F = [ufl.replace(f, {u_: x}) for f, u_, x in zip(self.problem.F,
                                                              self.problem.u,
                                                              self.x)]
        self.J = [ufl.replace(j, {u_: x}) for j, u_, x in zip(self.problem.J,
                                                              self.problem.u,
                                                              self.x)]
        self._F = FunctionHierarchy(u.function_space())
        self._J = [assemble(j, bcs=bcs) for j, bcs in zip(self.J, self.bcs)]
        self._P = self._J

        self.snes = PETSc.SNES().create()

        self.snes.setDM(u.function_space()[-1].mesh()._plex)
        self.snes.setFunction(self.form_function_level(len(self.u) - 1), None)
        self.snes.setJacobian(self.form_jacobian_level(len(self.u) - 1),
                              J=self._J[-1]._M.handle,
                              P=self._P[-1]._M.handle)

        self.snes.setUp()
        self.snes.setFromOptions()

        if self.snes.getType() == self.snes.Type.FAS:
            nlevel = self.snes.getFASLevels()
            for i in range(1, nlevel):
                lvl = i + (len(self.x) - nlevel)
                self.snes.setFASRestriction(i, restriction(lvl, self.bcs[lvl-1]))
                self.snes.setFASInterpolation(i, interpolation(lvl, self.bcs[lvl]))
                self.snes.setFASInjection(i, injection(lvl, self.bcs[lvl-1]))

                f = Function(self.u[lvl-1].function_space())
                f.assign(1)
                with f.dat.vec_ro as v:
                    self.snes.setFASRScale(i, v)

            for i in range(nlevel):
                lvl = i + (len(self.x) - nlevel)
                for snes in [self.snes.getFASCycleSNES(i),
                             self.snes.getFASSmootherDown(i),
                             self.snes.getFASSmootherUp(i)]:
                    snes.setDM(u.function_space()[lvl].mesh()._plex)
                    snes.setFunction(self.form_function_level(lvl), None)
                    snes.setJacobian(self.form_jacobian_level(lvl),
                                     J=self._J[lvl]._M.handle,
                                     P=self._P[lvl]._M.handle)
            coarse = self.snes.getFASCoarseSolve()
            lvl = len(self.x) - nlevel

            coarse.setDM(u.function_space()[lvl].mesh()._plex)
            coarse.setFunction(self.form_function_level(lvl), None)
            coarse.setJacobian(self.form_jacobian_level(lvl),
                               J=self._J[lvl]._M.handle,
                               P=self._P[lvl]._M.handle)

    def form_function_level(self, level):
        def form_function(snes, X, F):
            with self.x[level].dat.vec as x:
                if x != X:
                    with x as x_, X as X_:
                        x_[:] = X_[:]
            assemble(self.F[level], tensor=self._F[level])
            for bc in self.bcs[level]:
                bc.zero(self._F[level])
            with self._F[level].dat.vec_ro as f:
                if f != F:
                    with f as f_, F as F_:
                        F_[:] = f_[:]
        return form_function

    def form_jacobian_level(self, level):
        def form_jacobian(snes, X, J, P):
            with self.x[level].dat.vec as x:
                if x != X:
                    with x as x_, X as X_:
                        x_[:] = X_[:]
            assemble(self.J[level], tensor=self._P[level],
                     bcs=self.bcs[level])
            self._P[level].M._force_evaluation()
            if J != P:
                assemble(self.J[level], tensor=self._J[level],
                         bcs=self.bcs[level])
                self._J[level].M._force_evaluation()

            return PETSc.Mat.Structure.SAME_NONZERO_PATTERN
        return form_jacobian

    def solve(self):
        for i, bcs in enumerate(self.bcs):
            for bc in bcs:
                bc.apply(self.u[i])

        with self.u[-1].dat.vec as u:
            self.snes.solve(None, u)

F = []
bcs = []
u = FunctionHierarchy(fh)
for u_ in u:
    V = u_.function_space()
    v = TestFunction(V)
    dx = V.mesh()._dx
    rhs = Function(V)
    rhs.interpolate(Expression("32*pi*pi*sin(4*pi*x[0])*sin(4*pi*x[1])"))
    F.append(dot(grad(u_), grad(v))*dx - rhs*v*dx)
    bcs.append([DirichletBC(V, 0.0, (1, 2, 3, 4))])


problem = NonlinearVariationalProblemHierarchy(F, u, bcs=bcs)

opts = PETSc.Options()

opts["snes_type"] = "fas"
opts["snes_monitor"] = None
opts["fas_levels_snes_monitor"] = None
opts["fas_coarse_snes_monitor"] = None
opts["snes_fas_levels"] = nlevel+1
opts["snes_fas_type"] = "full"
opts["fas_coarse_ksp_rtol"] = 1.0e-10
opts["fas_coarse_snes_type"] = "ksponly"
opts["fas_coarse_snes_max_it"] = 1
opts["snes_converged_reason"] = None
opts["fas_levels_snes_type"] = "ksponly"
opts["fas_levels_snes_max_it"] = 1
opts["fas_levels_snes_norm_schedule"] = "always"
opts["fas_levels_ksp_type"] = "richardson"
opts["fas_levels_pc_type"] = "sor"
opts["fas_levels_ksp_max_it"] = 2
opts["fas_levels_ksp_convergence_test"] = "skip"
opts["snes_max_it"] = 1

solver = NonlinearVariationalSolverHierarchy(problem)

solver.solve()

exact = Function(fh[-1])
exact.interpolate(Expression("sin(4*pi*x[0])*sin(4*pi*x[1])"))

print norm(assemble(exact - u[-1]))

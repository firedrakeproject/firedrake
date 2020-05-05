import firedrake
import ufl
from functools import reduce
from enum import IntEnum
from operator import and_
from firedrake.petsc import PETSc


__all__ = ("TransferManager", )


native = frozenset(["Lagrange", "Discontinuous Lagrange", "Real", "Q", "DQ"])


class Op(IntEnum):
    PROLONG = 0
    RESTRICT = 1
    INJECT = 2


class TransferManager(object):
    class Cache(object):
        """A caching object for work vectors and matrices.

        :arg element: The element to use for the caching."""
        def __init__(self, element):
            cell = element.cell()
            degree = element.degree()
            family = lambda c: "DG" if c.is_simplex() else "DQ"
            if isinstance(cell, ufl.TensorProductCell):
                scalar_element = ufl.TensorProductElement(*(ufl.FiniteElement(family(c), cell=c, degree=d)
                                                            for (c, d) in zip(cell.sub_cells(), degree)))
            else:
                scalar_element = ufl.FiniteElement(family(cell), cell=cell, degree=degree)
            shape = element.value_shape()
            if len(shape) == 0:
                DG = scalar_element
            elif len(shape) == 1:
                shape, = shape
                DG = ufl.VectorElement(scalar_element, dim=shape)
            else:
                DG = ufl.TensorElement(scalar_element, shape=shape)
            self.embedding_element = DG
            self._V_DG_mass = {}
            self._DG_inv_mass = {}
            self._V_approx_inv_mass = {}
            self._V_inv_mass_ksp = {}
            self._DG_work = {}
            self._work_vec = {}
            self._V_dof_weights = {}

    def __init__(self, *, native_transfers=None, use_averaging=True):
        """
        An object for managing transfers between levels in a multigrid
        hierarchy (possibly via embedding in DG spaces).

        :arg native_transfers: dict mapping UFL element
           to "natively supported" transfer operators. This should be
           a three-tuple of (prolong, restrict, inject).
        :arg use_averaging: Use averaging to approximate the
           projection out of the embedded DG space? If False, a global
           L2 projection will be performed.
        """
        self.native_transfers = native_transfers or {}
        self.use_averaging = use_averaging
        self.caches = {}

    def is_native(self, element):
        if element in self.native_transfers.keys():
            return True
        if isinstance(element.cell(), ufl.TensorProductCell):
            return reduce(and_, map(self.is_native, element.sub_elements()))
        return element.family() in native

    def _native_transfer(self, element, op):
        try:
            return self.native_transfers[element][op]
        except KeyError:
            if self.is_native(element):
                ops = firedrake.prolong, firedrake.restrict, firedrake.inject
                return self.native_transfers.setdefault(element, ops)[op]
        return None

    def cache(self, element):
        try:
            return self.caches[element]
        except KeyError:
            return self.caches.setdefault(element, TransferManager.Cache(element))

    def V_dof_weights(self, V):
        """Dof weights for averaging projection.

        :arg V: function space to compute weights for.
        :returns: A PETSc Vec.
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._V_dof_weights[key]
        except KeyError:
            # Compute dof multiplicity for V
            # Spin over all (owned) cells incrementing visible dofs by 1.
            # After halo exchange, the Vec representation is the
            # global Vector counting the number of cells that see each
            # dof.
            f = firedrake.Function(V)
            firedrake.par_loop(("{[i, j]: 0 <= i < A.dofs and 0 <= j < %d}" % V.value_size,
                               "A[i, j] = A[i, j] + 1"),
                               firedrake.dx,
                               {"A": (f, firedrake.INC)},
                               is_loopy_kernel=True)
            with f.dat.vec_ro as fv:
                return cache._V_dof_weights.setdefault(key, fv.copy())

    def V_DG_mass(self, V, DG):
        """
        Mass matrix from between V and DG spaces.
        :arg V: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from V -> DG
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._V_DG_mass[key]
        except KeyError:
            M = firedrake.assemble(firedrake.inner(firedrake.TestFunction(DG),
                                                   firedrake.TrialFunction(V))*firedrake.dx)
            return cache._V_DG_mass.setdefault(key, M.petscmat)

    def DG_inv_mass(self, DG):
        """
        Inverse DG mass matrix
        :arg DG: the DG space
        :returns: A PETSc Mat.
        """
        cache = self.caches[DG.ufl_element()]
        key = DG.dim()
        try:
            return cache._DG_inv_mass[key]
        except KeyError:
            M = firedrake.assemble(firedrake.Tensor(firedrake.inner(firedrake.TestFunction(DG),
                                                                    firedrake.TrialFunction(DG))*firedrake.dx).inv)
            return cache._DG_inv_mass.setdefault(key, M.petscmat)

    def V_approx_inv_mass(self, V, DG):
        """
        Approximate inverse mass.  Computes (cellwise) (V, V)^{-1} (V, DG).
        :arg V: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from V -> DG.
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._V_approx_inv_mass[key]
        except KeyError:
            a = firedrake.Tensor(firedrake.inner(firedrake.TestFunction(V),
                                                 firedrake.TrialFunction(V))*firedrake.dx)
            b = firedrake.Tensor(firedrake.inner(firedrake.TestFunction(V),
                                                 firedrake.TrialFunction(DG))*firedrake.dx)
            M = firedrake.assemble(a.inv * b)
            return cache._V_approx_inv_mass.setdefault(key, M.petscmat)

    def V_inv_mass_ksp(self, V):
        """
        A KSP inverting a mass matrix
        :arg V: a function space.
        :returns: A PETSc KSP for inverting (V, V).
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._V_inv_mass_ksp[key]
        except KeyError:
            M = firedrake.assemble(firedrake.inner(firedrake.TestFunction(V),
                                                   firedrake.TrialFunction(V))*firedrake.dx)
            ksp = PETSc.KSP().create(comm=V.comm)
            ksp.setOperators(M.petscmat)
            ksp.setOptionsPrefix("{}_prolongation_mass_".format(V.ufl_element()._short_name))
            ksp.setType("preonly")
            ksp.pc.setType("cholesky")
            ksp.setFromOptions()
            ksp.setUp()
            return cache._V_inv_mass_ksp.setdefault(key, ksp)

    def DG_work(self, V):
        """A DG work Function matching V
        :arg V: a function space.
        :returns: A Function in the embedding DG space.
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._DG_work[key]
        except KeyError:
            DG = firedrake.FunctionSpace(V.mesh(), cache.embedding_element)
            return cache._DG_work.setdefault(key, firedrake.Function(DG))

    def work_vec(self, V):
        """A work Vec for V
        :arg V: a function space.
        :returns: A PETSc Vec for V.
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._work_vec[key]
        except KeyError:
            return cache._work_vec.setdefault(key, V.dof_dset.layout_vec.duplicate())

    def op(self, source, target, transfer_op):
        """Primal transfer (either prolongation or injection).

        :arg source: The source function.
        :arg target: The target function.
        :arg transfer_op: The transfer operation for the DG space.
        """
        Vs = source.function_space()
        Vt = target.function_space()

        source_element = Vs.ufl_element()
        target_element = Vt.ufl_element()
        if self.is_native(source_element) and self.is_native(target_element):
            return self._native_transfer(source_element, transfer_op)(source, target)
        if type(source_element) is ufl.MixedElement:
            assert type(target_element) is ufl.MixedElement
            for source_, target_ in zip(source.split(), target.split()):
                self.op(source_, target_, transfer_op=transfer_op)
            return target
        # Get some work vectors
        dgsource = self.DG_work(Vs)
        dgtarget = self.DG_work(Vt)
        VDGs = dgsource.function_space()
        VDGt = dgtarget.function_space()
        dgwork = self.work_vec(VDGs)

        # Project into DG space
        # u \in Vs -> u \in VDGs
        with source.dat.vec_ro as sv, dgsource.dat.vec_wo as dgv:
            self.V_DG_mass(Vs, VDGs).mult(sv, dgwork)
            self.DG_inv_mass(VDGs).mult(dgwork, dgv)

        # Transfer
        # u \in VDGs -> u \in VDGt
        self.op(dgsource, dgtarget, transfer_op)

        # Project back
        # u \in VDGt -> u \in Vt
        with dgtarget.dat.vec_ro as dgv, target.dat.vec_wo as t:
            if self.use_averaging:
                self.V_approx_inv_mass(Vt, VDGt).mult(dgv, t)
                t.pointwiseDivide(t, self.V_dof_weights(Vt))
            else:
                work = self.work_vec(Vt)
                self.V_DG_mass(Vt, VDGt).multTranspose(dgv, work)
                self.V_inv_mass_ksp(Vt).solve(work, t)

    def prolong(self, uc, uf):
        """Prolong a function.

        :arg uc: The source (coarse grid) function.
        :arg uf: The target (fine grid) function.
        """
        self.op(uc, uf, transfer_op=Op.PROLONG)

    def inject(self, uf, uc):
        """Inject a function (primal restriction)

        :arg uc: The source (fine grid) function.
        :arg uf: The target (coarse grid) function.
        """
        self.op(uf, uc, transfer_op=Op.INJECT)

    def restrict(self, gf, gc):
        """Restrict a dual function.

        :arg gf: The source (fine grid) dual function.
        :arg gc: The target (coarse grid) dual function.
        """
        Vc = gc.function_space()
        Vf = gf.function_space()

        source_element = Vf.ufl_element()
        target_element = Vc.ufl_element()
        if self.is_native(source_element) and self.is_native(target_element):
            return self._native_transfer(source_element, Op.RESTRICT)(gf, gc)
        if type(source_element) is ufl.MixedElement:
            assert type(target_element) is ufl.MixedElement
            for source_, target_ in zip(gf.split(), gc.split()):
                self.restrict(source_, target_)
            return gc
        dgf = self.DG_work(Vf)
        dgc = self.DG_work(Vc)
        VDGf = dgf.function_space()
        VDGc = dgc.function_space()
        work = self.work_vec(Vf)
        dgwork = self.work_vec(VDGc)

        # g \in Vf^* -> g \in VDGf^*
        with gf.dat.vec_ro as gfv, dgf.dat.vec_wo as dgscratch:
            if self.use_averaging:
                work.pointwiseDivide(gfv, self.V_dof_weights(Vf))
                self.V_approx_inv_mass(Vf, VDGf).multTranspose(work, dgscratch)
            else:
                self.V_inv_mass_ksp(Vf).solve(gfv, work)
                self.V_DG_mass(Vf, VDGf).mult(work, dgscratch)

        # g \in VDGf^* -> g \in VDGc^*
        self.restrict(dgf, dgc)

        # g \in VDGc^* -> g \in Vc^*
        with dgc.dat.vec_ro as dgscratch, gc.dat.vec_wo as gcv:
            self.DG_inv_mass(VDGc).mult(dgscratch, dgwork)
            self.V_DG_mass(Vc, VDGc).multTranspose(dgwork, gcv)
        return gc

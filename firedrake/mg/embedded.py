import firedrake
import ufl
import finat.ufl
import weakref
from enum import IntEnum
from firedrake.petsc import PETSc
from firedrake.embedding import get_embedding_dg_element


__all__ = ("TransferManager", )


native_families = frozenset(["Lagrange", "Discontinuous Lagrange", "Real", "Q", "DQ", "BrokenElement"])
alfeld_families = frozenset(["Hsieh-Clough-Tocher", "Reduced-Hsieh-Clough-Tocher", "Johnson-Mercier",
                             "Alfeld-Sorokina", "Arnold-Qin", "Reduced-Arnold-Qin", "Christiansen-Hu",
                             "Guzman-Neilan", "Guzman-Neilan Bubble"])
non_native_variants = frozenset(["integral", "fdm", "alfeld"])


def get_embedding_element(element, value_shape):
    broken_cg = element.sobolev_space in {ufl.H1, ufl.H2}
    dg_element = get_embedding_dg_element(element, value_shape, broken_cg=broken_cg)
    variant = element.variant() or "default"
    family = element.family()
    # Elements on Alfeld splits are embedded onto DG Powell-Sabin.
    # This yields supermesh projection
    if (family in alfeld_families) or ("alfeld" in variant.lower() and family != "Discontinuous Lagrange"):
        dg_element = dg_element.reconstruct(variant="powell-sabin")
    return dg_element


class Op(IntEnum):
    PROLONG = 0
    RESTRICT = 1
    INJECT = 2


class TransferManager(object):
    class Cache(object):
        """A caching object for work vectors and matrices.

        :arg element: The element to use for the caching."""
        def __init__(self, ufl_element, value_shape):
            self.embedding_element = get_embedding_dg_element(ufl_element, value_shape)
            self._dat_versions = {}
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

    def is_native(self, element, op):
        if element in self.native_transfers.keys():
            return self.native_transfers[element][op] is not None
        if isinstance(element.cell, ufl.TensorProductCell) and len(element.sub_elements) > 0:
            return all(self.is_native(e, op) for e in element.sub_elements)
        return (element.family() in native_families) and not (element.variant() in non_native_variants)

    def _native_transfer(self, element, op):
        try:
            return self.native_transfers[element][op]
        except KeyError:
            if self.is_native(element, op):
                ops = firedrake.prolong, firedrake.restrict, firedrake.inject
                return self.native_transfers.setdefault(element, ops)[op]
        return None

    def cache(self, V):
        key = (V.ufl_element(), V.value_shape)
        try:
            return self.caches[key]
        except KeyError:
            return self.caches.setdefault(key, TransferManager.Cache(*key))

    def V_dof_weights(self, V):
        """Dof weights for averaging projection.

        :arg V: function space to compute weights for.
        :returns: A PETSc Vec.
        """
        cache = self.cache(V)
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
            firedrake.par_loop(("{[i, j]: 0 <= i < A.dofs and 0 <= j < %d}" % V.block_size,
                               "A[i, j] = A[i, j] + 1"),
                               firedrake.dx,
                               {"A": (f, firedrake.INC)})
            with f.dat.vec_ro as fv:
                return cache._V_dof_weights.setdefault(key, fv.copy())

    def V_DG_mass(self, V, DG):
        """
        Mass matrix from between V and DG spaces.
        :arg V: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from V -> DG
        """
        cache = self.cache(V)
        key = V.dim()
        try:
            return cache._V_DG_mass[key]
        except KeyError:
            M = firedrake.assemble(firedrake.inner(firedrake.TrialFunction(V),
                                                   firedrake.TestFunction(DG))*firedrake.dx)
            return cache._V_DG_mass.setdefault(key, M.petscmat)

    def DG_inv_mass(self, DG):
        """
        Inverse DG mass matrix
        :arg DG: the DG space
        :returns: A PETSc Mat.
        """
        cache = self.cache(DG)
        key = DG.dim()
        try:
            return cache._DG_inv_mass[key]
        except KeyError:
            M = firedrake.assemble(firedrake.Tensor(firedrake.inner(firedrake.TrialFunction(DG),
                                                                    firedrake.TestFunction(DG))*firedrake.dx).inv)
            return cache._DG_inv_mass.setdefault(key, M.petscmat)

    def V_approx_inv_mass(self, V, DG):
        """
        Approximate inverse mass.  Computes (cellwise) (V, V)^{-1} (V, DG).
        :arg V: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from V -> DG.
        """
        cache = self.cache(V)
        key = V.dim()
        try:
            return cache._V_approx_inv_mass[key]
        except KeyError:
            a = firedrake.Tensor(firedrake.inner(firedrake.TrialFunction(V),
                                                 firedrake.TestFunction(V))*firedrake.dx)
            b = firedrake.Tensor(firedrake.inner(firedrake.TrialFunction(DG),
                                                 firedrake.TestFunction(V))*firedrake.dx)
            M = firedrake.assemble(a.inv * b)
            return cache._V_approx_inv_mass.setdefault(key, M.petscmat)

    def V_inv_mass_ksp(self, V):
        """
        A KSP inverting a mass matrix
        :arg V: a function space.
        :returns: A PETSc KSP for inverting (V, V).
        """
        cache = self.cache(V)
        key = V.dim()
        try:
            return cache._V_inv_mass_ksp[key]
        except KeyError:
            M = firedrake.assemble(firedrake.inner(firedrake.TrialFunction(V),
                                                   firedrake.TestFunction(V))*firedrake.dx)
            ksp = PETSc.KSP().create(comm=V._comm)
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
        needs_dual = ufl.duals.is_dual(V)
        cache = self.cache(V)
        key = (V.dim(), needs_dual)
        try:
            return cache._DG_work[key]
        except KeyError:
            if needs_dual:
                primal = self.DG_work(V.dual())
                dual = primal.riesz_representation(riesz_map="l2")
                return cache._DG_work.setdefault(key, dual)
            DG = firedrake.FunctionSpace(V.mesh(), cache.embedding_element)
            return cache._DG_work.setdefault(key, firedrake.Function(DG))

    def work_vec(self, V):
        """A work Vec for V
        :arg V: a function space.
        :returns: A PETSc Vec for V.
        """
        cache = self.cache(V)
        key = V.dim()
        try:
            return cache._work_vec[key]
        except KeyError:
            return cache._work_vec.setdefault(key, V.dof_dset.layout_vec.duplicate())

    def requires_transfer(self, V, transfer_op, source, target):
        """Determine whether either the source or target have been modified since
        the last time a grid transfer was executed with them."""
        key = (transfer_op, weakref.ref(source.dat), weakref.ref(target.dat))
        dat_versions = (source.dat.dat_version, target.dat.dat_version)
        try:
            return self.cache(V)._dat_versions[key] != dat_versions
        except KeyError:
            return True

    def cache_dat_versions(self, V, transfer_op, source, target):
        """Record the returned dat_versions of the source and target."""
        key = (transfer_op, weakref.ref(source.dat), weakref.ref(target.dat))
        dat_versions = (source.dat.dat_version, target.dat.dat_version)
        self.cache(V)._dat_versions[key] = dat_versions

    @PETSc.Log.EventDecorator()
    def op(self, source, target, transfer_op):
        """Primal transfer (either prolongation or injection).

        :arg source: The source :class:`.Function`.
        :arg target: The target :class:`.Function`.
        :arg transfer_op: The transfer operation for the DG space.
        """
        Vs = source.function_space()
        Vt = target.function_space()
        source_element = Vs.ufl_element()
        target_element = Vt.ufl_element()
        if not self.requires_transfer(Vs, transfer_op, source, target):
            return

        if all(self.is_native(e, transfer_op) for e in (source_element, target_element)):
            self._native_transfer(source_element, transfer_op)(source, target)
        elif type(source_element) is finat.ufl.MixedElement:
            assert type(target_element) is finat.ufl.MixedElement
            for source_, target_ in zip(source.subfunctions, target.subfunctions):
                self.op(source_, target_, transfer_op=transfer_op)
        else:
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
        self.cache_dat_versions(Vs, transfer_op, source, target)

    def prolong(self, uc, uf):
        """Prolong a function.

        :arg uc: The source (coarse grid) function.
        :arg uf: The target (fine grid) function.
        """
        self.op(uc, uf, transfer_op=Op.PROLONG)

    def inject(self, uf, uc):
        """Inject a function (primal restriction)

        :arg uf: The source (fine grid) function.
        :arg uc: The target (coarse grid) function.
        """
        self.op(uf, uc, transfer_op=Op.INJECT)

    def restrict(self, source, target):
        """Restrict a dual function.

        :arg source: The source (fine grid) :class:`.Cofunction`.
        :arg target: The target (coarse grid) :class:`.Cofunction`.
        """
        Vs_star = source.function_space()
        Vt_star = target.function_space()
        source_element = Vs_star.ufl_element()
        target_element = Vt_star.ufl_element()
        if not self.requires_transfer(Vs_star, Op.RESTRICT, source, target):
            return

        if all(self.is_native(e, Op.RESTRICT) for e in (source_element, target_element)):
            self._native_transfer(source_element, Op.RESTRICT)(source, target)
        elif type(source_element) is finat.ufl.MixedElement:
            assert type(target_element) is finat.ufl.MixedElement
            for source_, target_ in zip(source.subfunctions, target.subfunctions):
                self.restrict(source_, target_)
        else:
            Vs = Vs_star.dual()
            Vt = Vt_star.dual()
            # Get some work vectors
            dgsource = self.DG_work(Vs_star)
            dgtarget = self.DG_work(Vt_star)
            VDGs = dgsource.function_space().dual()
            VDGt = dgtarget.function_space().dual()
            work = self.work_vec(Vs)
            dgwork = self.work_vec(VDGt)

            # g \in Vs^* -> g \in VDGs^*
            with source.dat.vec_ro as sv, dgsource.dat.vec_wo as dgv:
                if self.use_averaging:
                    work.pointwiseDivide(sv, self.V_dof_weights(Vs))
                    self.V_approx_inv_mass(Vs, VDGs).multTranspose(work, dgv)
                else:
                    self.V_inv_mass_ksp(Vs).solve(sv, work)
                    self.V_DG_mass(Vs, VDGs).mult(work, dgv)

            # g \in VDGs^* -> g \in VDGt^*
            self.restrict(dgsource, dgtarget)

            # g \in VDGt^* -> g \in Vt^*
            with dgtarget.dat.vec_ro as dgv, target.dat.vec_wo as t:
                self.DG_inv_mass(VDGt).mult(dgv, dgwork)
                self.V_DG_mass(Vt, VDGt).multTranspose(dgwork, t)
        self.cache_dat_versions(Vs_star, Op.RESTRICT, source, target)

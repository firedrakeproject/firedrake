import firedrake
import ufl
import finat.ufl
import weakref
from functools import reduce
from enum import IntEnum
from operator import and_
from firedrake.petsc import PETSc
from firedrake.embedding import get_embedding_dg_element

__all__ = ("TransferManager", )


native = frozenset(["Lagrange", "Discontinuous Lagrange", "Real", "Q", "DQ", "Crouzeix-Raviart"])


class Op(IntEnum):
    PROLONG = 0
    RESTRICT = 1
    INJECT = 2


class TransferManager(object):
    class Cache(object):
        """A caching object for work vectors and matrices.

        :arg element: The element to use for the caching."""
        def __init__(self, element):
            self.embedding_element = get_embedding_dg_element(element)
            self._dat_versions = {}
            self._V_DG_mass = {}
            self._DG_inv_mass = {}
            self._V_approx_inv_mass = {}
            self._DG_work = {}
            self._work_vec = {}
            self._V_dof_weights = {}
            self._V_coarse_jacobian = {}
            self._V_approx_inv_mass_piola = {}
            self._V_DG_project_reference_values = {}
            self._V_inv_mass_ksp = {}
            self._V_DG_mass_piola = {}
            self._interpolators = {}

    def __init__(self, *, native_transfers=None, use_averaging=True, mat_type=None):
        """
        An object for managing transfers between levels in a multigrid
        hierarchy (possibly via embedding in DG spaces).

        :arg native_transfers: dict mapping UFL element
           to "natively supported" transfer operators. This should be
           a three-tuple of (prolong, restrict, inject).
        :arg use_averaging: Use averaging to approximate the
           projection out of the embedded DG space? If False, a global
           L2 projection will be performed.
        :arg mat_type: matrix type for the projection operators.
        """
        self.native_transfers = native_transfers or {}
        self.use_averaging = use_averaging
        self.caches = {}
        self.mat_type = mat_type or firedrake.parameters["default_matrix_type"]

    def is_native(self, element):
        if element in self.native_transfers.keys():
            return True
        if isinstance(element.cell, ufl.TensorProductCell) and len(element.sub_elements) > 0:
            return reduce(and_, map(self.is_native, element.sub_elements))
        return element.family() in native

    def _interpolator(self, uc, uf):
        Vc = uc.function_space()
        Vf = uf.function_space()
        key = (Vc, Vf)
        cache = self.cache(Vc.ufl_element())._interpolators
        try:
            return cache[key]
        except KeyError:
            return cache.setdefault(key, firedrake.Interpolator(firedrake.TestFunction(Vc), Vf))

    def _prolong(self, x, y):
        self._interpolator(x, y).interpolate(x, output=y, transpose=False)

    def _restrict(self, x, y):
        self._interpolator(y, x).interpolate(x, output=y, transpose=True)

    def _native_transfer(self, element, op):
        try:
            return self.native_transfers[element][op]
        except KeyError:
            if self.is_native(element):
                # ops = self._prolong, self._restrict, self._prolong
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
                               {"A": (f, firedrake.INC)})
            with f.dat.vec_ro as fv:
                return cache._V_dof_weights.setdefault(key, fv.copy())

    def pullback(self, mapping, F):
        if mapping.lower() == "identity":
            return 1
        elif mapping.lower() == "covariant piola":
            return ufl.inv(F).T
        elif mapping.lower() == "contravariant piola":
            S = ufl.diag(ufl.as_tensor([-1]+[1]*(F.ufl_shape[-1]-1)))
            return (1/ufl.det(F)) * F * S
        elif mapping.lower() == "l2 piola":
            return 1/abs(ufl.det(F))
        else:
            raise ValueError("Unrecognized mapping", mapping)

    def pushforward(self, mapping, F):
        if mapping.lower() == "identity":
            return 1
        elif mapping.lower() == "covariant piola":
            return F.T
        elif mapping.lower() == "contravariant piola":
            S = ufl.diag(ufl.as_tensor([-1]+[1]*(F.ufl_shape[-1]-1)))
            return ufl.det(F) * S * ufl.inv(F)
        elif mapping.lower() == "l2 piola":
            return abs(ufl.det(F))
        else:
            raise ValueError("Unrecognized mapping", mapping)

    def affine_function_space(self, V):
        if V.ufl_element().mapping() == "identity":
            return V
        if V.mesh().ufl_cell().is_simplex() and V.mesh().coordinates.function_space().ufl_element().degree() == 1:
            return V
        return firedrake.FunctionSpace(V.mesh(), finat.ufl.WithMapping(V.ufl_element(), mapping="identity"))

    def dx(self, V):
        degree = V.ufl_element().degree()
        try:
            degree = max(degree)
        except TypeError:
            pass
        return firedrake.dx(degree=2*degree, domain=V.mesh())

    def V_coarse_jacobian(self, Vc, Vf):
        cmesh = Vc.mesh()
        Fc = firedrake.Jacobian(cmesh)
        vector_element = get_embedding_dg_element(cmesh.coordinates.function_space().ufl_element())
        element = finat.ufl.TensorElement(vector_element.sub_elements[0], shape=Fc.ufl_shape)

        cache = self.cache(element)
        key = (Vc.mesh(), Vf.mesh())
        try:
            return cache._V_coarse_jacobian[key]
        except KeyError:
            Qc = firedrake.FunctionSpace(Vc.mesh(), element)
            Qf = firedrake.FunctionSpace(Vf.mesh(), element)
            qc = firedrake.Function(Qc)
            qf = firedrake.Function(Qf)
            qc.interpolate(Fc)
            if Qc.dim() < Qf.dim():
                self.prolong(qc, qf)
            else:
                self.inject(qc, qf)
            return cache._V_coarse_jacobian.setdefault(key, qf)

    def V_DG_project_reference_values(self, Vc, DG):
        """
        Project reference values in Vc onto DG.
        Computes (cellwise) (DG, DG)^{-1} (DG, hat{Vc}).
        :arg Vc: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from Vc -> DG.
        """
        cache = self.cache(Vc.ufl_element())
        key = Vc.dim()
        try:
            return cache._V_DG_project_reference_values[key]
        except KeyError:
            V = self.affine_function_space(Vc)
            idet = 1/abs(ufl.JacobianDeterminant(V.mesh()))
            a = firedrake.Tensor(firedrake.inner(firedrake.TrialFunction(DG),
                                                 firedrake.TestFunction(DG))*idet*self.dx(DG))
            b = firedrake.Tensor(firedrake.inner(firedrake.TrialFunction(V),
                                                 firedrake.TestFunction(DG))*idet*self.dx(DG))
            M = firedrake.assemble(a.inv * b, mat_type=self.mat_type)
            return cache._V_DG_project_reference_values.setdefault(key, M.petscmat)

    def V_approx_inv_mass_piola(self, Vc, Vf, DG):
        """
        Approximate inverse mass.
        Computes (cellwise) (hat{Vf}, hat{Vf})^{-1} (hat{Vf}, pushforward(Vf) * pullback(Vc) * DG).
        :arg Vc: a function space to extract the piola transform
        :arg Vf: the target function space
        :arg DG: the source DG space
        :returns: A PETSc Mat mapping from DG -> Vf.
        """
        cache = self.cache(Vf.ufl_element())
        key = (Vf.dim(), Vc.dim())
        try:
            return cache._V_approx_inv_mass_piola[key]
        except KeyError:
            dg_trial = firedrake.TrialFunction(DG)
            V = self.affine_function_space(Vf)
            if V != Vf:
                mapping = Vc.ufl_element().mapping()
                push = self.pushforward(mapping, ufl.Jacobian(Vf.mesh()))
                pull = self.pullback(mapping, self.V_coarse_jacobian(Vc, Vf))
                scale = push * pull
                if scale.ufl_shape and len(dg_trial.ufl_shape) > 1:
                    dg_trial = dg_trial * scale.T
                else:
                    dg_trial = scale * dg_trial

            idet = 1/abs(ufl.JacobianDeterminant(V.mesh()))
            a = firedrake.Tensor(firedrake.inner(firedrake.TrialFunction(V),
                                                 firedrake.TestFunction(V))*idet*self.dx(V))
            b = firedrake.Tensor(firedrake.inner(dg_trial, firedrake.TestFunction(V))*idet*self.dx(V))
            M = firedrake.assemble(a.inv * b, mat_type=self.mat_type)
            return cache._V_approx_inv_mass_piola.setdefault(key, M.petscmat)

    def V_DG_mass_piola(self, Vc, Vf, DG):
        """
        Computes (hat{Vf}, pushforward(Vf) * pullback(Vc) * DG).
        :arg Vc: a function space to extract the piola transform
        :arg Vf: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from Vf -> DG.
        """
        cache = self.cache(Vf.ufl_element())
        key = (Vf.dim(), Vc.dim())
        try:
            return cache._V_DG_mass_piola[key]
        except KeyError:
            dg_trial = firedrake.TrialFunction(DG)
            V = self.affine_function_space(Vf)
            if V != Vf:
                mapping = Vc.ufl_element().mapping()
                push = self.pushforward(mapping, ufl.Jacobian(Vf.mesh()))
                pull = self.pullback(mapping, self.V_coarse_jacobian(Vc, Vf))
                scale = push * pull
                if scale.ufl_shape and len(dg_trial.ufl_shape) > 1:
                    dg_trial = dg_trial * scale.T
                else:
                    dg_trial = scale * dg_trial

            idet = 1/abs(ufl.JacobianDeterminant(V.mesh()))
            b = firedrake.Tensor(firedrake.inner(dg_trial, firedrake.TestFunction(V))*idet*self.dx(V))
            M = firedrake.assemble(b, mat_type=self.mat_type)
            return cache._V_DG_mass_piola.setdefault(key, M.petscmat)

    def V_inv_mass_ksp(self, V):
        """
        A KSP inverting a reference mass matrix
        :arg V: a function space.
        :returns: A PETSc KSP for inverting (V, V)_hat{K}.
        """
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._V_inv_mass_ksp[key]
        except KeyError:
            V = self.affine_function_space(V)
            idet = 1/abs(ufl.JacobianDeterminant(V.mesh()))
            M = firedrake.assemble(firedrake.inner(firedrake.TrialFunction(V),
                                                   firedrake.TestFunction(V))*idet*self.dx(V),
                                   mat_type=self.mat_type)
            ksp = PETSc.KSP().create(comm=V._comm)
            ksp.setOperators(M.petscmat)
            try:
                short_name = V.ufl_element()._short_name
            except AttributeError:
                short_name = str(V.ufl_element())
            ksp.setOptionsPrefix("{}_prolongation_mass_".format(short_name))
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
        cache = self.cache(V.ufl_element())
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
        cache = self.cache(V.ufl_element())
        key = V.dim()
        try:
            return cache._work_vec[key]
        except KeyError:
            return cache._work_vec.setdefault(key, V.dof_dset.layout_vec.duplicate())

    def requires_transfer(self, element, transfer_op, source, target):
        """Determine whether either the source or target have been modified since
        the last time a grid transfer was executed with them."""
        key = (transfer_op, weakref.ref(source.dat), weakref.ref(target.dat))
        dat_versions = (source.dat.dat_version, target.dat.dat_version)
        try:
            return self.cache(element)._dat_versions[key] != dat_versions
        except KeyError:
            return True

    def cache_dat_versions(self, element, transfer_op, source, target):
        """Record the returned dat_versions of the source and target."""
        key = (transfer_op, weakref.ref(source.dat), weakref.ref(target.dat))
        dat_versions = (source.dat.dat_version, target.dat.dat_version)
        self.cache(element)._dat_versions[key] = dat_versions

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
        if not self.requires_transfer(source_element, transfer_op, source, target):
            return

        if self.is_native(source_element) and self.is_native(target_element):
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

            # Project into DG space
            # u \in Vs -> u \in VDGs
            with source.dat.vec_ro as sv, dgsource.dat.vec_wo as dgv:
                self.V_DG_project_reference_values(Vs, VDGs).mult(sv, dgv)

            # Transfer
            # u \in VDGs -> u \in VDGt
            self.op(dgsource, dgtarget, transfer_op)

            # Project back
            # u \in VDGt -> u \in Vt
            with dgtarget.dat.vec_ro as dgv, target.dat.vec_wo as t:
                if self.use_averaging:
                    self.V_approx_inv_mass_piola(Vs, Vt, VDGt).mult(dgv, t)
                    t.pointwiseDivide(t, self.V_dof_weights(Vt))
                else:
                    work = self.work_vec(Vt)
                    self.V_DG_mass_piola(Vs, Vt, VDGt).multTranspose(dgv, work)
                    self.V_inv_mass_ksp(Vt).solve(work, t)
        self.cache_dat_versions(source_element, transfer_op, source, target)

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
        if not self.requires_transfer(source_element, Op.RESTRICT, source, target):
            return

        if self.is_native(source_element) and self.is_native(target_element):
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

            # g \in Vs^* -> g \in VDGs^*
            with source.dat.vec_ro as sv, dgsource.dat.vec_wo as dgv:
                if self.use_averaging:
                    work.pointwiseDivide(sv, self.V_dof_weights(Vs))
                    self.V_approx_inv_mass_piola(Vt, Vs, VDGs).multTranspose(work, dgv)
                else:
                    self.V_inv_mass_ksp(Vs).solve(sv, work)
                    self.V_DG_mass_piola(Vt, Vs, VDGs).multTranspose(work, dgv)

            # g \in VDGs^* -> g \in VDGt^*
            self.restrict(dgsource, dgtarget)

            # g \in VDGt^* -> g \in Vt^*
            with dgtarget.dat.vec_ro as dgv, target.dat.vec_wo as t:
                self.V_DG_project_reference_values(Vt, VDGt).multTranspose(dgv, t)
        self.cache_dat_versions(source_element, Op.RESTRICT, source, target)

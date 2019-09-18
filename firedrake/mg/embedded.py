import firedrake
import ufl
from firedrake.petsc import PETSc


__all__ = ("EmbeddedDGTransfer", )


native = frozenset(["Lagrange", "Discontinuous Lagrange", "Real", "Q", "DQ"])


def EmbeddedDGTransfer(element, use_fortin_interpolation=True):
    """Return an object that provides grid transfers.

    This object provides transfers of functions between levels in a
    mesh hierarchy for element types that are not supported directly
    by Firedrake's multilevel implementation.  It computes transfers
    by embedding the element in an appropriate DG space, transferring
    the resulting DG function, and then mapping back.

    :arg element: The UFL element that will be transferred.
    :arg use_fortin_interpolation: If True, use Fortin operator
       as an approximate mass inverse, otherwise, use an actual mass
       solve.
    """
    if type(element) is firedrake.MixedElement:
        return MixedTransfer(element, use_fortin_interpolation=use_fortin_interpolation)
    else:
        return SingleTransfer(element, use_fortin_interpolation=use_fortin_interpolation)


class NativeTransfer(object):
    """A transfer object that just calls the native transfer
    routines."""
    @staticmethod
    def prolong(c, f):
        return firedrake.prolong(c, f)

    @staticmethod
    def inject(f, c):
        return firedrake.inject(f, c)

    @staticmethod
    def restrict(f, c):
        return firedrake.restrict(f, c)


class SingleTransfer(object):
    """Create a transfer object for a single (not mixed) element."""
    def __new__(cls, element, use_fortin_interpolation=True):
        if element.family() in native:
            return NativeTransfer
        else:
            return super().__new__(cls)

    def __init__(self, element, use_fortin_interpolation=True):
        degree = element.degree()
        cell = element.cell()
        shape = element.value_shape()
        if len(shape) == 0:
            DG = ufl.FiniteElement("DG", cell, degree)
        elif len(shape) == 1:
            shape, = shape
            DG = ufl.VectorElement("DG", cell, degree, dim=shape)
        else:
            DG = ufl.TensorElement("DG", cell, degree, shape=shape)

        self.embedding_element = DG
        self.use_fortin_interpolation = use_fortin_interpolation
        self._V_DG_mass = {}
        self._DG_inv_mass = {}
        self._V_approx_inv_mass = {}
        self._V_inv_mass_ksp = {}
        self._DG_work = {}
        self._work_vec = {}
        self._V_dof_weights = {}

    def V_dof_weights(self, V):
        """Dof weights for Fortin projection.

        :arg V: function space to compute weights for.
        :returns: A PETSc Vec.
        """
        key = V.dim()
        try:
            return self._V_dof_weights[key]
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
                return self._V_dof_weights.setdefault(key, fv.copy())

    def V_DG_mass(self, V, DG):
        """
        Mass matrix from between V and DG spaces.
        :arg V: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from V -> DG
        """
        key = V.dim()
        try:
            return self._V_DG_mass[key]
        except KeyError:
            M = firedrake.assemble(firedrake.inner(firedrake.TestFunction(DG),
                                                   firedrake.TrialFunction(V))*firedrake.dx)
            return self._V_DG_mass.setdefault(key, M.petscmat)

    def DG_inv_mass(self, DG):
        """
        Inverse DG mass matrix
        :arg DG: the DG space
        :returns: A PETSc Mat.
        """
        key = DG.dim()
        try:
            return self._DG_inv_mass[key]
        except KeyError:
            M = firedrake.assemble(firedrake.Tensor(firedrake.inner(firedrake.TestFunction(DG),
                                                                    firedrake.TrialFunction(DG))*firedrake.dx).inv)
            return self._DG_inv_mass.setdefault(key, M.petscmat)

    def V_approx_inv_mass(self, V, DG):
        """
        Approximate inverse mass.  Computes (cellwise) (V, V)^{-1} (V, DG).
        :arg V: a function space
        :arg DG: the DG space
        :returns: A PETSc Mat mapping from V -> DG.
        """
        key = V.dim()
        try:
            return self._V_approx_inv_mass[key]
        except KeyError:
            a = firedrake.Tensor(firedrake.inner(firedrake.TestFunction(V),
                                                 firedrake.TrialFunction(V))*firedrake.dx)
            b = firedrake.Tensor(firedrake.inner(firedrake.TestFunction(V),
                                                 firedrake.TrialFunction(DG))*firedrake.dx)
            M = firedrake.assemble(a.inv * b)
            return self._V_approx_inv_mass.setdefault(key, M.petscmat)

    def V_inv_mass_ksp(self, V):
        """
        A KSP inverting a mass matrix
        :arg V: a function space.
        :returns: A PETSc KSP for inverting (V, V).
        """
        key = V.dim()
        try:
            return self._V_inv_mass_ksp[key]
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
            return self._V_inv_mass_ksp.setdefault(key, ksp)

    def DG_work(self, V):
        """A DG work Function matching V
        :arg V: a function space.
        :returns: A Function in the embedding DG space.
        """
        key = V.dim()
        try:
            return self._DG_work[key]
        except KeyError:
            DG = firedrake.FunctionSpace(V.mesh(), self.embedding_element)
            return self._DG_work.setdefault(key, firedrake.Function(DG))

    def work_vec(self, V):
        """A work Vec for V
        :arg V: a function space.
        :returns: A PETSc Vec for V.
        """
        key = V.dim()
        try:
            return self._work_vec[key]
        except KeyError:
            return self._work_vec.setdefault(key, V.dof_dset.layout_vec.duplicate())

    def op(self, source, target, transfer_op):
        """Primal transfer (either prolongation or injection).

        :arg source: The source function.
        :arg target: The target function.
        :arg transfer_op: The transfer operation for the DG space.
        """
        Vs = source.function_space()
        Vt = target.function_space()

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
        transfer_op(dgsource, dgtarget)

        # Project back
        # u \in VDGt -> u \in Vt
        with dgtarget.dat.vec_ro as dgv, target.dat.vec_wo as t:
            if self.use_fortin_interpolation:
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
        self.op(uc, uf, transfer_op=firedrake.prolong)

    def inject(self, uf, uc):
        """Inject a function (primal restriction)

        :arg uc: The source (fine grid) function.
        :arg uf: The target (coarse grid) function.
        """
        self.op(uf, uc, transfer_op=firedrake.inject)

    def restrict(self, gf, gc):
        """Restrict a dual function.

        :arg gf: The source (fine grid) dual function.
        :arg gc: The target (coarse grid) dual function.
        """
        Vc = gc.function_space()
        Vf = gf.function_space()

        dgf = self.DG_work(Vf)
        dgc = self.DG_work(Vc)
        VDGf = dgf.function_space()
        VDGc = dgc.function_space()
        work = self.work_vec(Vf)
        dgwork = self.work_vec(VDGc)

        # g \in Vf^* -> g \in VDGf^*
        with gf.dat.vec_ro as gfv, dgf.dat.vec_wo as dgscratch:
            if self.use_fortin_interpolation:
                work.pointwiseDivide(gfv, self.V_dof_weights(Vf))
                self.V_approx_inv_mass(Vf, VDGf).multTranspose(work, dgscratch)
            else:
                self.V_inv_mass_ksp(Vf).solve(gfv, work)
                self.V_DG_mass(Vf, VDGf).mult(work, dgscratch)

        # g \in VDGf^* -> g \in VDGc^*
        firedrake.restrict(dgf, dgc)

        # g \in VDGc^* -> g \in Vc^*
        with dgc.dat.vec_ro as dgscratch, gc.dat.vec_wo as gcv:
            self.DG_inv_mass(VDGc).mult(dgscratch, dgwork)
            self.V_DG_mass(Vc, VDGc).multTranspose(dgwork, gcv)


class MixedTransfer(object):
    """Create a transfer object for a mixed element.

    This just makes :class:`SingleTransfer` objects for each sub element."""
    def __init__(self, element, use_fortin_interpolation=True):
        self._transfers = {}
        self._use_fortin_interpolation = use_fortin_interpolation

    def transfers(self, element):
        try:
            return self._transfers[element]
        except KeyError:
            transfers = tuple(SingleTransfer(e, use_fortin_interpolation=self._use_fortin_interpolation)
                              for e in element.sub_elements())
            return self._transfers.setdefault(element, transfers)

    def prolong(self, uc, uf):
        element = uc.function_space().ufl_element()
        for c, f, t in zip(uc.split(), uf.split(), self.transfers(element)):
            t.prolong(c, f)

    def inject(self, uf, uc):
        element = uf.function_space().ufl_element()
        for f, c, t in zip(uf.split(), uc.split(), self.transfers(element)):
            t.inject(f, c)

    def restrict(self, uf_dual, uc_dual):
        element = uf_dual.function_space().ufl_element()
        for f, c, t in zip(uf_dual.split(), uc_dual.split(), self.transfers(element)):
            t.restrict(f, c)

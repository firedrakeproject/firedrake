import petsctools
from firedrake.petsc import PETSc
from firedrake.function import Function


class CovariancePC(petsctools.PCBase):
    r"""
    A python PC context for a covariance operator.
    Will apply either the action or inverse of the covariance,
    whichever is the opposite of the Mat operator.

    .. math::

        B: V^{*} \to V

        B^{-1}: V \to V^{*}

    Available options:

    * ``-pc_use_amat`` - use Amat to apply the covariance operator.

    See Also
    --------
    ~firedrake.adjoint.covariance_operator.CovarianceOperatorBase
    ~firedrake.adjoint.covariance_operator.AutoregressiveCovariance
    ~firedrake.adjoint.covariance_operator.CovarianceMatCtx
    ~firedrake.adjoint.covariance_operator.CovarianceMat
    """
    needs_python_pmat = True
    prefix = "covariance"

    def initialize(self, pc):
        from firedrake.adjoint.covariance_operator import CovarianceMatCtx
        A, P = pc.getOperators()

        use_amat_prefix = self.parent_prefix + "pc_use_amat"
        self.use_amat = PETSc.Options().getBool(use_amat_prefix, False)
        mat = (A if self.use_amat else P).getPythonContext()

        if not isinstance(mat, CovarianceMatCtx):
            raise TypeError(
                "CovariancePC needs a CovarianceMatCtx")
        covariance = mat.covariance

        self.covariance = covariance
        self.mat = mat

        V = covariance.function_space()
        primal = Function(V)
        dual = Function(V.dual())

        # PC does the opposite of the Mat
        if mat.operation == CovarianceMatCtx.Operation.ACTION:
            self.operation = CovarianceMatCtx.Operation.INVERSE
            self.x = primal
            self.y = dual
            self._apply_op = covariance.apply_inverse
        elif mat.operation == CovarianceMatCtx.Operation.INVERSE:
            self.operation = CovarianceMatCtx.Operation.ACTION
            self.x = dual
            self.y = primal
            self._apply_op = covariance.apply_action

    def apply(self, pc, x, y):
        """Apply the action or inverse of the covariance operator
        to x, putting the result in y.

        y is not guaranteed to be zero on entry.

        Parameters
        ----------
        pc : PETSc.PC
            The PETSc preconditioner that self is the python context of.
        x : PETSc.Vec
            The vector acted on by the pc.
        y : PETSc.Vec
            The result of the pc application.
        """
        with self.x.dat.vec_wo as xvec:
            x.copy(result=xvec)

        self._apply_op(self.x, tensor=self.y)

        with self.y.dat.vec_ro as yvec:
            yvec.copy(result=y)

    def update(self, pc):
        pass

    def view(self, pc, viewer=None):
        """View object. Method usually called by PETSc with e.g. -ksp_view.
        """
        from firedrake.adjoint.covariance_operator import (
            CovarianceMatCtx, AutoregressiveCovariance)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return

        viewer.printfASCII(f"  firedrake covariance operator preconditioner: {type(self).__name__}\n")
        viewer.printfASCII(f"  Applying the {str(self.operation)} of the covariance operator {type(self.covariance).__name__}\n")

        if self.use_amat:
            viewer.printfASCII("  using Amat matrix\n")

        if (type(self.covariance) is AutoregressiveCovariance) and (self.covariance.iterations > 0):
            if viewer.getFormat() == PETSc.Viewer.Format.ASCII_INFO_DETAIL:
                if self.operation == CovarianceMatCtx.Operation.ACTION:
                    viewer.printfASCII("  Information for the diffusion solver for applying the action:\n")
                    ksp = self.covariance.solver.snes.ksp
                elif self.operation == CovarianceMatCtx.Operation.INVERSE:
                    viewer.printfASCII("  Information for the mass solver for applying the inverse:\n")
                    ksp = self.covariance.mass_solver.snes.ksp
                viewer.pushASCIITab()
                ksp.view(viewer)
                viewer.popASCIITab()
            else:
                prefix = pc.getOptionsPrefix() or ""
                viewer.printfASCII(f"  Use -{prefix}ksp_view ::ascii_info_detail to display information for diffusion or mass solver.\n")

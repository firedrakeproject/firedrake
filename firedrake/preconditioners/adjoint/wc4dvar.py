from firedrake.petsc import PETSc
import petsctools
from pyadjoint.optimization.tao_solver import (
    ReducedFunctionalMat, RFOperation)


class WC4DVarSchurPC(petsctools.PCBase):
    """
    Preconditioner to approximate the inverse of the Schur complement
    of the saddle point formulation of the weak constraint 4DVar, which is
    equivalent to the Gauss-Newton Hessian of the primal WC4DVar formulation.

    The exact Schur complement :math:`S` and the approximation
    :math:`\\tilde{S}` that this PC applies are:

    .. math::

      S & = L^{T}D^{-1}L + H^{T}R^{-1}H

      \\tilde{S}^{-1} & = \\tilde{L}^{-T}\\tilde{D}\\tilde{L}^{-1}

    where :math:`L` is the all-at-once system, and H, D, and R are the
    block-diagonal matrices with the observation operators, observation
    error covariances, and model error covariances at each observation
    time respectively.

    KSPs are created for; :math:`\\tilde{L}`, for :math:`\\tilde{L}^{-T}` using
    a :func:`~pyadjoint.optimization.tao_solver.ReducedFunctionalMat` for the
    :class:`~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional`;
    and for :math:`\\tilde{D}` using an :class:`~firedrake.ensemble.ensemble_mat.EnsembleBlockDiagonalMat`
    where each block is a :func:`~firedrake.adjoint.covariance_operator.CovarianceMat`.

    PETSc Options
    -------------
    * ``-wcschur_l`` - Options for solving the :math:`L` and :math:`L^{T}`.
    * ``-wcschur_ltlm`` - Options solely for :math:`L`, e.g. monitors.
    * ``-wcschur_ladj`` - Options solely for :math:`L^{T}`, e.g. monitors.
    * ``-wcschur_d`` - Options for solving the :math:`D^{-1}`

    Notes
    -----
    Identical solver options should be used for :math:`\\tilde{L}` and
    :math:`\\tilde{L}^{T}` to ensure symmetry of :math:`\\tilde{S}^{-1}``.

    References
    ----------
    Fisher M. and Gurol S., 2017: "Parallelization in the time dimension of
    four-dimensional variational data assimilation".
    Q.J.R. Meteorol. Soc. 142: 1136–1147, DOI:10.1002/qj.2997

    See Also
    --------
    ~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional
    ~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional
    ~firedrake.ensemble.ensemble_mat.EnsembleBlockDiagonalMat
    ~firedrake.adjoint.covariance_operator.CovarianceMat
    """

    prefix = "wcschur_"

    @PETSc.Log.EventDecorator()
    def initialize(self, pc):
        # TODO: petsctools.cite("Fisher2017")
        super().initialize(pc)

        A, P = pc.getOperators()

        Jhat = self._get_wc4dvar_rf(P)
        self.Jhat = Jhat

        self.ensemble = Jhat.ensemble
        global_comm = self.ensemble.global_comm

        self.col_space = Jhat.control_space
        self.row_space = Jhat.control_space.dual()

        # Create the Mats for each component

        LTmat_p, Dmat_p, Lmat_p = self._schur_comp_mats(Jhat)

        pc_amat_prefix = pc.getOptionsPrefix() + "pc_use_amat"
        self.use_amat = PETSc.Options().getBool(pc_amat_prefix, False)

        if self.use_amat:
            self.Ahat = self._get_wc4dvar_rf(A)
            LTmat, Dmat, Lmat = self._schur_comp_mats(self.Ahat)

        else:
            self.Ahat = Jhat
            (LTmat, Dmat, Lmat) = (LTmat_p, Dmat_p, Lmat_p)

        # Create the KSPs for each component

        self.Lksp = PETSc.KSP().create(comm=global_comm)
        self.LTksp = PETSc.KSP().create(comm=global_comm)
        self.Dksp = PETSc.KSP().create(comm=global_comm)

        self.Lksp.setOperators(Lmat, Lmat_p)
        self.LTksp.setOperators(LTmat, LTmat_p)
        self.Dksp.setOperators(Dmat, Dmat_p)

        # usually will set identical options for L and LT
        default_l_options = petsctools.DefaultOptionSet(
            base_prefix=self.full_prefix + "l_",
            custom_prefix_endings=("tlm", "adj"))

        petsctools.attach_options(
            self.Lksp,
            options_prefix=self.full_prefix+"l_tlm",
            default_options_set=default_l_options)

        petsctools.attach_options(
            self.LTksp,
            options_prefix=self.full_prefix+"l_adj",
            default_options_set=default_l_options)

        petsctools.attach_options(
            self.Dksp,
            options_prefix=self.full_prefix+"d")

        # default to behaving like a set of pcs
        petsctools.set_default_parameter(
            self.Lksp, "ksp_type", "preonly")
        petsctools.set_default_parameter(
            self.LTksp, "ksp_type", "preonly")
        petsctools.set_default_parameter(
            self.Dksp, "ksp_type", "preonly")

        petsctools.set_from_options(self.Lksp)
        petsctools.set_from_options(self.LTksp)
        petsctools.set_from_options(self.Dksp)

        # Make sure we print properly with view
        self.Lksp.incrementTabLevel(1, parent=pc)
        self.Lksp.pc.incrementTabLevel(1, parent=pc)

        self.LTksp.incrementTabLevel(1, parent=pc)
        self.LTksp.pc.incrementTabLevel(1, parent=pc)

        self.Dksp.incrementTabLevel(1, parent=pc)
        self.Dksp.pc.incrementTabLevel(1, parent=pc)

    def _schur_comp_mats(self, Jhat):
        from firedrake import EnsembleBlockDiagonalMat
        from firedrake.adjoint import CovarianceMat

        # L and LT: all-at-once system Mats
        Lmat = ReducedFunctionalMat(
            Jhat.JL, action=RFOperation.TLM,
            comm=self.ensemble.global_comm)

        LTmat = ReducedFunctionalMat(
            Jhat.JL, action=RFOperation.ADJOINT,
            comm=self.ensemble.global_comm)

        # D: background and model error covariances
        rank = self.ensemble.ensemble_rank
        BQ = [Jhat.background_covariance] if rank == 0 else []
        BQ.extend(Jhat.model_covariances)

        Dmat = EnsembleBlockDiagonalMat(
            [CovarianceMat(cov, operation='inverse') for cov in BQ],
            col_space=Jhat.control_space,
            row_space=Jhat.control_space.dual())

        return LTmat, Dmat, Lmat

    def _get_wc4dvar_rf(self, mat):
        from firedrake.adjoint import WC4DVarReducedFunctional
        # 1. If we are using the primal formulation then the mat is the
        #    WC4DVar Hessian and we can grab the RF off the context.
        # 2. If we are using the saddle point formulation then the mat
        #    is the zero (3,3) block of the saddle point MatNest that
        #    we previously stashed the RF on.
        if mat.getType() == "python":
            Jhat = mat.getPythonContext().rf
        else:
            Jhat = mat.getAttr("Jhat")

        if not isinstance(Jhat, WC4DVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jhat_name = petsctools.petscobj2str(Jhat)
            raise TypeError(
                f"{self_name} expects a WC4DVarReducedFunctional"
                f" not a {Jhat_name}")

        return Jhat

    @PETSc.Log.EventDecorator()
    def apply(self, pc, x, y):
        rhs = x.copy()
        sol = y
        sol.zeroEntries()

        # Just chain the solve for each KSP

        with petsctools.inserted_options(self.LTksp):
            self.LTksp.solve(rhs, sol)

        sol.copy(result=rhs)
        sol.zeroEntries()

        with petsctools.inserted_options(self.Dksp):
            self.Dksp.solve(rhs, sol)

        sol.copy(result=rhs)
        sol.zeroEntries()

        with petsctools.inserted_options(self.Lksp):
            self.Lksp.solve(rhs, sol)

        # y is already sol so no copy needed

    @PETSc.Log.EventDecorator()
    def update(self, pc):
        # The mat should have taken care of updating
        # the Amat but we should check if the Pmat
        # needs updating.
        if self.Jhat is not self.Ahat:
            Adata = self.Ahat.control.data()._ad_to_petsc()
            Jdata = self.Jhat.control.data()._ad_to_petsc()
            if (Adata - Jdata).norm() > 1e-10:
                self.Jhat(self.Ahat.control.data())
        self.LTksp.setUp()
        self.Dksp.setUp()
        self.Lksp.setUp()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        # Need to view each of the subsolvers as well as ourselves.
        viewer.printfASCII(
            "PC to apply the approximate Weak Constraint 4DVar Schur complement.\n")
        # L ksp
        viewer.printfASCII(
            "The KSP for the all-at-once tangent linear model L is:\n")
        viewer.pushASCIITab()
        self.Lksp.view(viewer)
        viewer.popASCIITab()
        # LT ksp
        viewer.printfASCII(
            "The KSP for the all-at-once adjoint model L^{T} is:\n")
        viewer.pushASCIITab()
        self.LTksp.view(viewer)
        viewer.popASCIITab()
        # D ksp
        viewer.printfASCII(
            "The KSP for the all-at-once model error covariances is:\n")
        viewer.pushASCIITab()
        self.Dksp.view(viewer)
        viewer.popASCIITab()


def getSubWC4DVarSaddleMat(mat, sub: str | None = None):
    """
    Return a sub matrix of the saddle point MatNest.
    Options are 'D', 'R', 'L', 'LT', 'H', 'HT',
    or ``None`` to return all sub matrices.

    Parameters
    ----------
    mat : petsc4py.PETSc.Mat
        The MatNest for the saddle point system returned by
        :func:`WC4DVarSaddleMat`.

    sub :
        Which sub matrix to return.

    Returns
    -------
    tuple[petsc4py.PETSc.Mat] | petsc4py.PETSc.Mat :
        The sub Mat requested or a tuple of all sub Mats.

    See Also
    --------
    WC4DVarSaddleMat
    """
    idx = {
        'D': (0, 0),
        'R': (1, 1),
        'L': (0, 2),
        'LT': (2, 0),
        'H': (1, 2),
        'HT': (2, 1),
    }
    return (
        mat.getNestSubMatrix(*idx[sub])
        if sub is not None else
        tuple(mat.getNestSubMatrix(*i) for i in idx.values())
    )


@PETSc.Log.EventDecorator()
def WC4DVarSaddleMat(Jhat):
    """
    PETSc MatNest for the saddle point formulation of Weak Constraint 4DVar.

    .. math::

      A =
      \\begin{pmatrix}
         D     &  0    &  L  \\\\
         0     &  R    &  H  \\\\
         L^{T} & H^{T} &  0
      \\end{pmatrix}

    where :math:`L` is the all-at-once system, and H, D, and R are the
    block-diagonal matrices with the observation operators, observation
    error covariances, and model error covariances at each observation
    time respectively.

    Parameters
    ----------
    Jhat : WC4DVarReducedFunctional
        :class:`~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional`
        to construct the saddle point matrix from.

    Returns
    -------
    petsc4py.PETSc.Mat :
        The 3x3 PETSc MatNest for the saddle point system.

    Raises
    ------
    TypeError :
        If ``Jhat`` is not a
        :class:`~.firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional`.

    See Also
    --------
    getSubWC4DVarSaddleMat
    WC4DVarSaddleKSP
    WC4DVarSaddlePC
    ~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional
    ~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional
    """
    from firedrake import EnsembleBlockDiagonalMat
    from firedrake.adjoint import WC4DVarReducedFunctional, CovarianceMat
    # TODO: petsctools.cite("Fisher2017")

    if not isinstance(Jhat, WC4DVarReducedFunctional):
        raise TypeError(
            "WC4DVarSaddleMat must be constructed from a"
            f" WC4DVarReducedFunctional, not a {type(Jhat).__name__}")

    ensemble = Jhat.ensemble
    rank = ensemble.ensemble_rank
    Wc = Jhat.control_space
    Wo = Jhat.observation_space

    # L: all-at-once blocks
    Lmat = ReducedFunctionalMat(
        Jhat.JL, action=RFOperation.TLM,
        comm=ensemble.global_comm)

    LTmat = ReducedFunctionalMat(
        Jhat.JL, action=RFOperation.ADJOINT,
        comm=ensemble.global_comm)

    # H: observation operator blocks
    hmats = [
        ReducedFunctionalMat(
            rf, action=RFOperation.TLM, comm=ensemble.comm)
        for rf in Jhat.observation_rfs]

    htmats = [
        ReducedFunctionalMat(
            rf, action=RFOperation.ADJOINT, comm=ensemble.comm)
        for rf in Jhat.observation_rfs]

    Hmat = EnsembleBlockDiagonalMat(
        hmats, row_space=Wc, col_space=Wo)

    HTmat = EnsembleBlockDiagonalMat(
        htmats, row_space=Wo.dual(), col_space=Wc.dual())

    # D: model covariance block
    BQ = [Jhat.background_covariance] if rank == 0 else []
    BQ.extend(Jhat.model_covariances)

    Dmat = EnsembleBlockDiagonalMat(
        [CovarianceMat(cov, operation='action') for cov in BQ],
        col_space=Wc, row_space=Wc.dual())

    # R: observation covariance block
    R = Jhat.observation_covariances

    Rmat = EnsembleBlockDiagonalMat(
        [CovarianceMat(cov, operation='action') for cov in R],
        col_space=Wo, row_space=Wo.dual())

    vec_dx = Wc.layout_vec.duplicate()

    # We need to create a WC4DVarSchurPC on (3,3) block, so
    # to make sure we have access to the WC4DVarReducedFunctional.
    # The default -pc_fieldsplit_schur_precondition
    # we manually create a zero Mat for the (3,3) block even though
    # PETSc would treat is as zero anyway.
    A22 = PETSc.Mat().createConstantDiagonal(
        (vec_dx.sizes, vec_dx.sizes), 0.,
        comm=ensemble.global_comm)
    A22.setUp()
    A22.assemble()
    A22.setAttr("Jhat", Jhat)

    saddle_mat = PETSc.Mat().createNest(
        mats=[[Dmat,  None,  Lmat],  # noqa: E241
              [None,  Rmat,  Hmat],  # noqa: E241
              [LTmat, HTmat, A22]],  # noqa: E241
        comm=ensemble.global_comm)
    saddle_mat.setUp()
    saddle_mat.assemble()

    return saddle_mat


@PETSc.Log.EventDecorator()
def WC4DVarSaddleKSP(Jhat, Jphat=None, *,
                     solver_parameters: dict | None = None,
                     options_prefix: str | None = None):
    """
    PETSc KSP for the saddle point formulation of Weak Constraint 4DVar.

    Parameters
    ----------
    Jhat : WC4DVarReducedFunctional
        :class:`~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional`
        to use to construct the :func:`.WC4DVarSaddleMat` for the Amat operator.
    Jphat : WC4DVarReducedFunctional | None
        :class:`~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional`
        to construct the :func:`.WC4DVarSaddleMat` for the Pmat operator.
        If not provided then ``Jhat`` is used for both Amat and Pmat.
    solver_parameters :
        PETSc options for the KSP.
    options_prefix :
        Options prefix for the KSP.

    Returns
    -------
    petsc4py.PETSc.KSP :
        The KSP for the saddle point system.

    Raises
    ------
    TypeError :
        If ``Jhat`` is not a
        :class:`~.firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional`.

    See Also
    --------
    WC4DVarSaddleMat
    WC4DVarSaddlePC
    ~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional
    """
    amat = WC4DVarSaddleMat(Jhat)

    if Jphat:
        pmat = WC4DVarSaddleMat(Jphat)
    else:
        pmat = amat

    ksp = PETSc.KSP().create(
        comm=Jhat.ensemble.global_comm)
    ksp.setOperators(amat, pmat)

    petsctools.set_from_options(
        ksp, parameters=solver_parameters,
        options_prefix=options_prefix)

    return ksp


class WC4DVarSaddlePC(petsctools.PCBase):
    """
    Preconditioner for Weak Constraint 4DVar using the saddle point formulation.

    .. math::

      \\begin{pmatrix}
         D     &  0    &  L  \\\\
         0     &  R    &  H  \\\\
         L^{T} & H^{T} &  0
      \\end{pmatrix}
      \\begin{pmatrix}
        \\eta \\\\ \\lambda \\\\ \\delta x
      \\end{pmatrix}
      =
      \\begin{pmatrix}
        b \\\\ d \\\\ 0
      \\end{pmatrix}

    This PC acts on a :class:`~pyadjoint.optimization.tao_solver.ReducedFunctionalHessianMat` for a
    :class:`~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional`.
    It solves the larger saddle point system and returns just the :math:`\\delta x` part of the solution.

    The (3, 3) Schur complement of the saddle point system is
    the WC4DVar Hessian :math:`L^{T}D^{-1}L +H^{T}R^{-1}H`
    and is often approximated with the :class:`WC4DVarSchurPC`.

    PETSc Options
    -------------
    * ``-wcsaddle`` - Options for the KSP for the saddle point system.

    References
    ----------
    Fisher M. and Gurol S., 2017: "Parallelization in the time dimension of
    four-dimensional variational data assimilation".
    Q.J.R. Meteorol. Soc. 142: 1136–1147, DOI:10.1002/qj.2997

    See Also
    --------
    ~firedrake.adjoint.fourdvar_reduced_functional.WC4DVarReducedFunctional
    WC4DVarSchurPC
    """
    needs_python_amat = True
    needs_python_pmat = True

    prefix = "wcsaddle_"

    @PETSc.Log.EventDecorator()
    def initialize(self, pc):
        from firedrake.adjoint import WC4DVarReducedFunctional
        # TODO: petsctools.cite("Fisher2017")
        super().initialize(pc)

        Jhat = self.amat.rf
        if not isinstance(Jhat, WC4DVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jhat_name = petsctools.petscobj2str(Jhat)
            raise TypeError(
                f"{self_name} expects a WC4DVarReducedFunctional"
                f" not a {Jhat_name}")

        Jphat = self.pmat.rf
        if not isinstance(Jphat, WC4DVarReducedFunctional):
            self_name = petsctools.petscobj2str(self)
            Jphat_name = petsctools.petscobj2str(Jphat)
            raise TypeError(
                f"{self_name} expects a WC4DVarReducedFunctional"
                f" not a {Jphat_name}")

        self.Jhat = Jhat
        self.Jphat = Jphat
        self.ensemble = Jphat.ensemble

        rhs_prefix = self.parent_prefix + "pc_" + self.prefix + "rhs_type"
        self.rhs_type = PETSc.Options().getString(rhs_prefix, "saddle")

        self.use_amat = PETSc.Options().getBool(
            self.full_prefix + "use_amat", False)

        if self.use_amat:
            Jhat_a = Jhat
        else:
            Jhat_a = Jphat
        Jhat_p = Jphat

        self.saddle_ksp = WC4DVarSaddleKSP(
            Jhat_a, Jhat_p, options_prefix=self.full_prefix)
        self.saddle_mat, _ = self.saddle_ksp.getOperators()

        self.saddle_ksp.incrementTabLevel(1, parent=pc)
        self.saddle_ksp.pc.incrementTabLevel(1, parent=pc)

        self.rhs = self._create_vec()
        self.sol = self._create_vec()

        self.rhs_subvecs = self.rhs.getNestSubVecs()
        self.sol_subvecs = self.sol.getNestSubVecs()
        self.rhs_dn, self.rhs_dl, self.rhs_dx = self.rhs_subvecs
        self.sol_dn, self.sol_dl, self.sol_dx = self.sol_subvecs

    def _create_vec(self):
        """Create a VecNest for the saddle point solution or right hand side.
        """
        Wc = self.Jphat.control_space
        Wo = self.Jphat.observation_space

        v_dn = Wc.layout_vec.duplicate()
        v_dl = Wo.layout_vec.duplicate()
        v_dx = Wc.layout_vec.duplicate()

        v = PETSc.Vec().createNest(
            vecs=(v_dn, v_dl, v_dx),
            isets=self.saddle_mat.getNestISs()[0],
            comm=self.Jphat.ensemble.global_comm)

        v.setUp()

        return v

    @PETSc.Log.EventDecorator()
    def apply(self, pc, x, y):
        self.sol.zeroEntries()
        self.rhs.zeroEntries()

        val = self.Jhat.control.data()

        if self.rhs_type == "saddle":

            # JL and JH actually return -b and -d, but TAO expects
            # the negative update so it all comes out in the wash.
            with self.Jphat.JL(val).vec_ro() as bvec:
                bvec.copy(result=self.rhs_dn)

            with self.Jphat.JH(val).vec_ro() as dvec:
                dvec.copy(result=self.rhs_dl)

            self.rhs_dx.zeroEntries()

        elif self.rhs_type == "primal":
            self.rhs_dn.zeroEntries()
            self.rhs_dl.zeroEntries()
            x.copy(result=self.rhs_dx)
            self.rhs_dx *= -1

        # Make sure that the monolithic vec are up to date
        self.rhs.setNestSubVecs(self.rhs_subvecs)

        with petsctools.inserted_options(self.saddle_ksp):
            self.saddle_ksp.solve(self.rhs, self.sol)

        self.sol.getNestSubVecs()[2].copy(result=y)

    @PETSc.Log.EventDecorator()
    def update(self, pc):
        # The mat should have taken care of updating
        # the Amat but we should check if the Pmat
        # needs updating.
        if self.Jphat is not self.Jhat:
            Jdata = self.Jhat.control.data()._ad_to_petsc()
            Jpdata = self.Jphat.control.data()._ad_to_petsc()
            if (Jdata - Jpdata).norm() > 1e-10:
                self.Jphat(self.Jhat.control.data())
        self.saddle_ksp.setUp()

    def view(self, pc, viewer=None):
        super().view(pc, viewer)
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(
            "PC to solve the saddle point formulation of Weak Constraint 4DVar\n")
        viewer.printfASCII(
            "The KSP for the 3x3 saddle point system is:\n")
        viewer.pushASCIITab()
        self.saddle_ksp.view(viewer)
        viewer.popASCIITab()

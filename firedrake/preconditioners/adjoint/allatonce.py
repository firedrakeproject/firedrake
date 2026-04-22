import petsctools
from pyadjoint.optimization.tao_solver import (
    RFOperation,
    ReducedFunctionalTLMMat,
    ReducedFunctionalAdjointMat
)
from firedrake.petsc import PETSc


class AllAtOnceRFGaussSeidelPC(petsctools.PCBase):
    """
    Python preconditioner to approximate the inverse
    of the tangent linear or adjoint model of an
    :class:`~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional`.

    The tangent linear :math:`L` (or adjoint :math:`L^{T}`) is block lower
    (upper) triangular so can be solved exactly with forward (backward)
    substitution, i.e. block Gauss-Seidel.

    .. math::

      L =
      \\begin{pmatrix}
         I &  0 &  0 &  0 \\\\
        -M &  I &  0 &  0 \\\\
         0 & -M &  I &  0 \\\\
         0 &  0 & -M &  I \\\\
      \\end{pmatrix}

    The default is to use :math:`M` in the Gauss-Seidel iteration, in which
    case this PC is an exact solve. Two approximations of :math:`M`
    commonly used in weak-constraint 4DVar are also available:

    1. :math:`M=I`. Approximating the propagator with identity is equivalent to
       computing an inclusive sum of the right hand side.
    2. :math:`M=0`. This "approximation" is equivalent to
       ``"pc_type": "none"``. This option is just to allow
       easily switching between the three common approximations with a single option.

    Use the ``-pc_aaogs_model`` option to select which approximation to use.

    PETSc Options
    -------------
    * ``-pc_aaogs_type ('model'|'identity'|'zero')``
      - Whether to use M or to approximate it with I or 0.

    Notes
    -----
    The ``Mat`` for this PC must be a
    :func:`~pyadjoint.optimization.tao_solver.ReducedFunctionalMat` for an
    :class:`~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional`.

    See Also
    --------
    ~firedrake.adjoint.allatonce_reduced_functional.AllAtOnceReducedFunctional
    ~firedrake.preconditioners.adjoint.wc4dvar.WC4DVarSchurPC
    """
    prefix = "aaogs_"
    needs_python_pmat = True

    @PETSc.Log.EventDecorator()
    def initialize(self, pc):
        from firedrake.adjoint import AllAtOnceReducedFunctional
        pcname = type(self).__name__

        if isinstance(self.pmat, ReducedFunctionalTLMMat):
            self.operation = RFOperation.TLM
        elif isinstance(self.pmat, ReducedFunctionalAdjointMat):
            self.operation = RFOperation.ADJOINT
        else:
            raise TypeError(
                f"{pcname} requires a ReducedFunctionalMat for the TLM"
                f" or Adjoint action, not a {type(self.pmat).__name__}")

        Jhat = self.pmat.rf

        if not isinstance(Jhat, AllAtOnceReducedFunctional):
            raise TypeError(
                f"{pcname} requires an AllAtOnceReducedFunctional,"
                f" not a {type(Jhat).__name__}")

        self.Jhat = Jhat
        self.ensemble = Jhat.ensemble

        # We need both the primal and dual no matter what
        # the pmat type is so that we can use applyTranspose

        self.x = Jhat.controls[0]._ad_init_zero()
        self.y = Jhat.functional._ad_init_zero()

        self.xstar = Jhat.functional._ad_init_zero(dual=True)
        self.ystar = Jhat.controls[0]._ad_init_zero(dual=True)

        ptype_prefix = f"{self.parent_prefix}pc_{self.prefix}type"
        self.propagator_type = PETSc.Options().getString(ptype_prefix, "model")

        # Are we going to use M=M, M=I, or M=0?
        if self.propagator_type == "model":
            self.M = [M.tlm for M in Jhat.propagator_rfs]
            self.MT = [M.derivative for M in Jhat.propagator_rfs]
        elif self.propagator_type == "identity":
            self.M = [lambda x: x for _ in Jhat.propagator_rfs]
            self.MT = [lambda x: x for _ in Jhat.propagator_rfs]
        elif self.propagator_type != "zero":
            raise ValueError(
                f"{pcname} propagator_type must be 'model', 'identity',"
                f" or 'zero', not {self.propagator_type}")

        self.nlocal_stages = Jhat.nlocal_stages

    @PETSc.Log.EventDecorator()
    def apply(self, pc, x, y):
        if self.propagator_type == "zero":
            x.copy(y)
            return

        if self.operation == RFOperation.TLM:
            self.apply_tlm(pc, x, y)
        elif self.operation == RFOperation.ADJOINT:
            self.apply_adjoint(pc, x, y)

    @PETSc.Log.EventDecorator()
    def applyTranspose(self, pc, x, y):
        if self.propagator_type == "zero":
            x.copy(y)
            return

        if self.operation == RFOperation.TLM:
            self.apply_adjoint(pc, x, y)
        elif self.operation == RFOperation.ADJOINT:
            self.apply_tlm(pc, x, y)

    @PETSc.Log.EventDecorator()
    def apply_tlm(self, pc, x, y):
        ensemble = self.ensemble
        rank = ensemble.ensemble_rank
        is_first = rank == 0

        # This will either be the real M or I
        M = self.M

        with self.x.vec_wo() as xvec:
            x.copy(xvec)
        self.y.zero()

        xs = self.x.subfunctions
        ys = self.y.subfunctions

        # buffer for y_{i-1} being halo or local
        yprev = ys[0]._ad_init_zero()

        # First row of L is just identity
        if is_first:
            ys[0].assign(xs[0])
            yprev.assign(ys[0])

        off = 1 if is_first else 0

        # Step forward through each row on each member
        # in turn acculumating the action of M
        with ensemble.sequential(yprev=yprev) as ctx:
            yprev.assign(ctx.yprev)
            for i in range(off, self.nlocal_stages+off):
                ys[i].assign(xs[i] + M[i](yprev))
                yprev.assign(ys[i])
            ctx.yprev.assign(yprev)

        with self.y.vec_ro() as yvec:
            yvec.copy(y)

    @PETSc.Log.EventDecorator()
    def apply_adjoint(self, pc, x, y):
        ensemble = self.ensemble
        rank = ensemble.ensemble_rank
        is_last = rank == ensemble.ensemble_size - 1
        is_first = rank == 0

        # This will either be the real M^{T} or I
        MT = self.MT

        with self.xstar.vec_wo() as xvec:
            x.copy(xvec)
        self.ystar.zero()

        xs = self.xstar.subfunctions
        ys = self.ystar.subfunctions

        # buffer for M^{T}(x_{i+1}) being halo or local
        Mtxi = ys[0]._ad_init_zero()

        # Last row of L^{T} is just identity
        if is_last:
            ys[-1].assign(xs[-1])

        off = 0 if is_first else 1

        # Step backward through each row on each member
        # in turn acculumating the action of M^{T}
        with ensemble.sequential(reverse=True, Mtxi=Mtxi) as ctx:
            Mtxi.assign(ctx.Mtxi)
            for i in range(self.nlocal_stages-off, -1, -1):
                ys[i].assign(xs[i] + Mtxi)
                Mtxi.assign(MT[i](ys[i]))
            ctx.Mtxi.assign(Mtxi)

        with self.ystar.vec_ro() as yvec:
            yvec.copy(y)

    @PETSc.Log.EventDecorator()
    def update(self, pc):
        # The Mat context is responsible care of making sure
        # that the ReducedFunctional for each M is up to date.
        pass

    def view(self, pc, viewer=None):
        if hasattr(self, "Jhat"):
            viewer.printfASCII(
                "PC to apply the inverse of an AllAtOnceReducedFunctional.\n")
            if self.operation == RFOperation.TLM:
                op = "tangent linear"
                direction = "forward"
            else:
                op = "adjoint"
                direction = "backward"
            viewer.printfASCII(
                f"  Solving the {op} model via {direction} substitution.\n")
            viewer.printfASCII(
                f"  Using propagator type: {self.propagator_type}\n")

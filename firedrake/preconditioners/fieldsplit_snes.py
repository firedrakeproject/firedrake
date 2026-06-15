from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx as get_snesctx

__all__ = ("FieldsplitSNES",)


class FieldsplitSNES(SNESBase):
    """
    Nonlinear solver created by combining separate nonlinear solvers for
    individual collections (called splits) of variables (called fields).

    This a nonlinear extension of PETSc's linear preconditioner
    `PCFIELDSPLIT <https://petsc.org/release/manualpages/PC/PCFIELDSPLIT/>`_

    **PETSc Options**

    * ``-snes_fieldsplit_type (additive|multiplicative)``
      - whether to apply the component updates additively or multiplicatively.
      Defaults to ``"additive"``.
    * ``-snes_fieldsplit_%d_fields a,b,...``
      - indicates the fields to be used in the ``%d``'th split.
      Defaults to one split per field.
    * ``-fieldsplit_%d_``
      - the options prefix for the ``%d``'th split.
      Defaults to the default options for :class:`~firedrake.variational_solver.NonlinearVariationalSolver`.

    FieldsplitSNES is used for solving nonlinear problems with multiple
    components. There are two types, which are each defined below for
    the following problem with two components:

    .. math ::

        \\textrm{Find: }
        w = (u, v) \\in W = U \\times V

        \\textrm{Subject to: }
        F(w; dw) = 0 \\quad \\forall \\; dw = (du, dv) \\in W

        \\textrm{Where: }F(w) =
        \\begin{pmatrix}
            f(u, v; du) \\\\
            g(u, v; dv) \\\\
        \\end{pmatrix}
        = 0

    **Additive fieldsplit**

    Additive fieldsplit completely decouples the components at each
    iteration. This can be interpreted as a nonlinear block Jacobi
    relaxation. At iteration ``k`` each row of the following nonlinear
    problem is solved independently for the components of :math:`w^{k+1}`.

    .. math ::

        F(w^{k+1}) =
        \\begin{pmatrix}
            f(u^{k+1}, v^{k}; du) \\\\
            g(u^{k}, v^{k+1}; dv) \\\\
        \\end{pmatrix}
        = 0

    **Multiplicative fieldsplit**

    Multiplicative fieldsplit always uses the latest value of each component
    when solving each split. This can be interpreted as a nonlinear block
    Gauss-Seidel relaxation. At iteration ``k`` each row of the following
    nonlinear problem is solved in turn for the components of :math:`w^{k+1}`,
    with the updated value of each component used in later rows.

    .. math ::

        F(w^{k+1}) =
        \\begin{pmatrix}
            f(u^{k+1}, v^{k}; du) \\\\
            g(u^{k+1}, v^{k+1}; dv) \\\\
        \\end{pmatrix}
        = 0

    Notes
    -----
    The order in which the fields are solved with multiplicative fieldsplit
    can be controlled via the ``-snes_fieldsplit_%d_fields`` options,
    identically to with PCFIELDSPLIT.

    If a component subspace of the mixed function space has been given
    a name ``"fsname"`` then the prefix for the corresponding split will
    be ``-fieldsplit_fsname_`` instead of ``-fieldsplit_%d_``.

    See Also
    --------
    ~firedrake.preconditioners.auxiliary_snes.AuxiliaryOperatorSNES
    """

    _prefix = "fieldsplit_"

    # TODO:
    #   -fieldsplit_default Allow setting default options for all splits

    @PETSc.Log.EventDecorator()
    def initialize(self, snes):
        from firedrake import (  # circular import if we do this at file level
            NonlinearVariationalSolver, Function, Cofunction, replace)
        from pyop2 import MixedDat

        ctx = get_snesctx(snes.dm)

        self.u = ctx._x
        W = self.u.function_space()

        # buffer to save solution to outer problem during solve
        self.u_outer = Function(W)

        # buffers for shuffling solutions during solve
        self.uk = Function(W)
        self.uk1 = Function(W)

        # options for setting up the fieldsplit are "snes_fieldsplit_option"
        outer_prefix = snes.getOptionsPrefix() or ""
        snes_prefix = outer_prefix + 'snes_' + self._prefix

        snes_options = PETSc.Options(snes_prefix)
        self.fieldsplit_type = snes_options.getString('type', 'additive')
        if self.fieldsplit_type not in ('additive', 'multiplicative'):
            raise ValueError(
                'FieldsplitSNES option snes_fieldsplit_type must be'
                ' "additive" or "multiplicative"')

        self.splits = self._get_splits(snes_options, len(W))

        self.Gk = replace(ctx.F, {ctx._x: self.uk})

        # Break the SNESContext apart into one per split
        split_ctxs = ctx.split(self.splits)

        self._set_nullspaces(ctx, split_ctxs, self.splits)

        # Each split_ctx holds the form for it's own part
        # of G^{k+1}, but we also need to apply the forcing
        # from (G^{k} - F^{k}).
        # We do this by creating a Cofunction on the full
        # space then breaking it apart and subtracting the
        # relevant piece from each split_ctx's form.
        # Doing it this way means we can update the full
        # Cofunction and the data in the split pieces will
        # automatically be updated.
        self.b = Cofunction(W.dual())
        for split_ctx, fields in zip(split_ctxs, self.splits):
            V = split_ctx._x.function_space()
            if len(V) == 1:
                val = self.b.dat[fields[0]]
            else:
                val = MixedDat(self.b.dat[i] for i in fields)
            split_ctx.F -= Cofunction(V.dual(), val=val)

        # The solution for each split context views the data
        # of the relevant components of the solution of the
        # original context, so field_solver.solve() will also
        # update ctx._x of the outer snes.
        self.split_solvers = tuple(
            NonlinearVariationalSolver(
                split_ctx._problem,
                appctx=split_ctx.appctx,
                nullspace=split_ctx._nullspace,
                options_prefix=split_ctx.options_prefix)
            for i, split_ctx in enumerate(split_ctxs)
        )

        outer_snes = snes
        for solver in self.split_solvers:
            split_snes = solver.snes
            split_snes.incrementTabLevel(1, parent=outer_snes)
            split_snes.ksp.incrementTabLevel(1, parent=split_snes)
            split_snes.ksp.pc.incrementTabLevel(1, parent=split_snes.ksp)

    def _get_splits(self, snes_options, nfields):
        split_opts = {}
        # extract split specification options
        for k in range(nfields):
            if snes_options.hasName(f"{k}_fields"):
                split_opts[k] = snes_options.getIntArray(f"{k}_fields")

        # natural ordering with one field per split if no splits are specified
        if len(split_opts) == 0:
            return [[i] for i in range(nfields)]

        # split_list[k] is the list of fields in split k
        split_list = split_opts.values()

        # Are all fields specified exactly once?
        specified_fields = [k for fields in split_list for k in fields]
        specified_fields.sort()
        required_fields = [k for k in range(nfields)]

        if specified_fields != required_fields:
            raise ValueError(
                "Not all required fields were specified exactly once.\n"
                f"Required fields = {required_fields}\n"
                f"Specified fields = {split_opts}")

        return split_list

    def _set_nullspaces(self, ctx, split_ctxs, splits):
        from firedrake.nullspace import (
            VectorSpaceBasis, MixedVectorSpaceBasis)

        if ctx._nullspace is None:
            return

        for split_ctx, fields in zip(split_ctxs, splits):
            if len(fields) == 1:
                base = ctx._nullspace._bases[fields[0]]
                if not isinstance(base, VectorSpaceBasis):
                    continue
                split_space = base

            else:
                W = split_ctx._x.function_space()
                mono_bases = [ctx._nullspace._bases[i] for i in fields]

                split_bases = [
                    base if isinstance(base, VectorSpaceBasis) else W.sub(k)
                    for k, base in enumerate(mono_bases)
                ]

                split_space = MixedVectorSpaceBasis(W, split_bases)

            split_ctx._nullspace = split_space

    def update(self, snes):
        pass

    @PETSc.Log.EventDecorator()
    def step(self, snes, x, f, y):
        """Take one iteration of the nonlinear solver.
        """
        from firedrake.assemble import assemble

        # store current value of outer solution to restore
        # later in case it isn't the same as x.
        self.u_outer.assign(self.u)

        # make sure that self.u in the full form in
        # ctx has the most up to date solution u^{k}.
        with self.u.dat.vec_wo as vec:
            x.copy(vec)

        # save u^{k}
        self.uk.assign(self.u)

        assemble(self.Gk, tensor=self.b)

        # Grab F for solving Gk1 - (Gk - Fk)
        with self.b.dat.vec_wo as vec:
            vec -= f

        # The current snes solution x is held in uk, and we
        # will place the new solution in uk1.
        # The split_solvers evaluate forms containing u, so for each
        # splitting type u needs to hold:
        #   - additive: all fields need to hold uk values
        #   - multiplicative: fields need to hold uk before
        #       they are are solved for, and keep the updated uk1
        #       values afterwards.
        uks = self.uk.subfunctions
        uk1s = self.uk1.subfunctions
        us = self.u.subfunctions
        for solver, fields in zip(self.split_solvers, self.splits):

            solver.solve()

            # update the uk1 buffer
            for i in fields:
                uk1s[i].assign(us[i])

            # reset the values in this field
            if self.fieldsplit_type == 'additive':
                for i in fields:
                    us[i].assign(uks[i])

        with self.uk1.dat.vec_ro as vec:
            vec.copy(y)
            y.aypx(-1, x)

        # restore outer solution
        self.u.assign(self.u_outer)

    def view(self, snes, viewer=None):
        """View information about this object with ``-snes_view``.
        """
        super().view(snes, viewer)
        if hasattr(self, "split_solvers"):
            viewer.printfASCII(
                "SNES to solve for groups of variables separately.\n"
                f"  fieldsplit_type: {self.fieldsplit_type}\n"
                f"  total splits = {len(self.splits)}\n")
            for i, fields in enumerate(self.splits):
                viewer.printfASCII(f"  split {i} has fields {fields}\n")
            viewer.printfASCII(
                "Solver info for each split in the following SNES objects:\n")
            viewer.pushASCIITab()
            for i, (fields, solver) in enumerate(zip(self.splits,
                                                     self.split_solvers)):
                viewer.printfASCII(f"Split number {i} with fields {fields}:\n")
                solver.snes.view(viewer)
            viewer.popASCIITab()

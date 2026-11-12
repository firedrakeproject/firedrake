from __future__ import annotations

import re
import typing
from collections.abc import Sequence
from typing import Any

from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx as get_snesctx

if typing.TYPE_CHECKING:
    from firedrake.nullspace import MixedVectorSpaceBasis, VectorSpaceBasis
    from firedrake.solving_utils import _SNESContext

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
      Splits must be numbered consecutively from 0.
      Defaults to one split per field.
    * ``-snes_fieldsplit_default``
      - put any fields that are not in a ``-snes_fieldsplit_%d_fields``
      split into their own split, appended in field order after the
      specified splits. Without this option every field must be in
      exactly one split.
    * ``-fieldsplit_%d_``
      - the options prefix for the ``%d``'th split.
    * ``-fieldsplit_``
      - default options for all splits. Options set for an individual
      split with ``-fieldsplit_%d_`` take priority over these. Any options
      not set by either are the default options for
      :class:`~firedrake.variational_solver.NonlinearVariationalSolver`.
      For example, to use the same nonlinear solver for every split
      but a tighter tolerance for split 1:

      .. code-block:: python3

          solver_parameters = {
              "snes_type": "python",
              "snes_python_type": "firedrake.FieldsplitSNES",
              "snes_fieldsplit_type": "multiplicative",
              "fieldsplit": {
                  "snes_rtol": 1e-4,
                  "ksp_type": "preonly",
                  "pc_type": "lu",
              },
              "fieldsplit_1": {
                  "snes_rtol": 1e-8,
              },
          }

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

    A split that contains more than one field can itself be solved with
    a FieldsplitSNES. The options for this inner FieldsplitSNES have the
    prefix of the split, for example ``-fieldsplit_0_snes_fieldsplit_type``,
    and its splits have the prefix ``-fieldsplit_0_fieldsplit_%d_``. The
    default options that are given with the ``-fieldsplit_`` prefix apply
    only to the splits of the outer FieldsplitSNES. They are not applied
    to the splits of the inner FieldsplitSNES, which take their default
    options from the ``-fieldsplit_0_fieldsplit_`` prefix.

    See Also
    --------
    ~firedrake.preconditioners.auxiliary_snes.AuxiliaryOperatorSNES
    """

    _prefix = "fieldsplit_"

    @PETSc.Log.EventDecorator()
    def initialize(self, snes: PETSc.SNES) -> None:
        from firedrake import (  # circular import if we do this at file level
            NonlinearVariationalSolver, NonlinearVariationalProblem,
            Function, Cofunction, replace)

        ctx = get_snesctx(snes.dm)

        # Remember which context we were built from so that
        # update can tell if it has been replaced.
        self._ctx = ctx

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

        nullspaces = self._get_nullspaces(ctx, split_ctxs, self.splits)

        default_parameters = self._get_default_parameters(ctx, split_ctxs)

        # Each split problem holds the form for its own part
        # of G^{k+1}, but we also need to apply the forcing
        # from (G^{k} - F^{k}).
        # We do this by creating a Cofunction on the full
        # space then breaking it apart and subtracting the
        # relevant piece from each split problem's residual.
        # Doing it this way means we can update the full
        # Cofunction and the data in the split pieces will
        # automatically be updated.
        # The forcing must be added to the problem (not just the
        # split_ctx) because the NonlinearVariationalSolvers below
        # build their own SNESContexts from the problem.
        # ctx.split caches its results, so the split problems may be
        # shared with other users of ctx (e.g. another FieldsplitSNES
        # or a fieldsplit PC). We must not modify them, so we build
        # a new problem with the forcing for each split.
        # The forcing is independent of u so the Jacobian is unchanged.
        self.b = Cofunction(W.dual())
        split_problems = []
        for split_ctx, fields in zip(split_ctxs, self.splits):
            split_problem = split_ctx._problem
            V = split_problem.u_restrict.function_space()
            if len(fields) == 1:
                slice_ = W._labels[fields[0]]
            else:
                slice_ = [W._labels[i] for i in fields]
            forcing = Cofunction(V.dual(), val=self.b.dat[slice_])
            forced_problem = NonlinearVariationalProblem(
                split_problem.F - forcing,
                split_problem.u_restrict,
                bcs=split_problem.bcs,
                J=split_problem.J,
                Jp=None if split_problem.Jp_eq_J else split_problem.Jp,
                is_linear=split_problem.is_linear,
                form_compiler_parameters=split_problem.form_compiler_parameters)
            forced_problem._constant_jacobian = split_problem._constant_jacobian
            split_problems.append(forced_problem)

        # The solution for each split context views the data
        # of the relevant components of the solution of the
        # original context, so field_solver.solve() will also
        # update ctx._x of the outer snes.
        self.split_solvers = tuple(
            NonlinearVariationalSolver(
                split_problem,
                solver_parameters=dict(default_parameters),
                appctx=split_ctx.appctx,
                nullspace=nullspace,
                options_prefix=split_ctx.options_prefix)
            for split_problem, split_ctx, nullspace in zip(
                split_problems, split_ctxs, nullspaces)
        )

        outer_snes = snes
        for solver in self.split_solvers:
            split_snes = solver.snes
            split_snes.incrementTabLevel(1, parent=outer_snes)
            split_snes.ksp.incrementTabLevel(1, parent=split_snes)
            split_snes.ksp.pc.incrementTabLevel(1, parent=split_snes.ksp)

    def _get_splits(self, snes_options: PETSc.Options,
                    nfields: int) -> list[list[int]]:
        """Return the list of fields in each split.

        Splits are specified with ``-snes_fieldsplit_%d_fields``,
        numbered consecutively from 0. If no splits are specified
        then there is one split per field. If
        ``-snes_fieldsplit_default`` is set then any fields not
        assigned to a split are each put in their own split, appended
        in field order after the specified splits. Otherwise every
        field must be in exactly one split.
        """
        # Find every split that was specified, so that gaps in the
        # numbering are reported rather than silently ignored.
        split_numbers = sorted(
            int(match.group(1))
            for name in snes_options.getAll()
            if (match := re.fullmatch(r"(\d+)_fields", name))
        )
        if split_numbers != list(range(len(split_numbers))):
            raise ValueError(
                "FieldsplitSNES splits must be numbered consecutively from 0,"
                f" but snes_fieldsplit_%d_fields was given for splits {split_numbers}")

        split_list = [
            [int(i) for i in snes_options.getIntArray(f"{k}_fields")]
            for k in split_numbers
        ]

        specified_fields = [i for fields in split_list for i in fields]
        invalid_fields = sorted(set(i for i in specified_fields
                                    if not 0 <= i < nfields))
        if invalid_fields:
            raise ValueError(
                f"FieldsplitSNES splits contain fields {invalid_fields}"
                f" but there are only {nfields} fields")

        repeated_fields = sorted(set(i for i in specified_fields
                                     if specified_fields.count(i) > 1))
        if repeated_fields:
            raise ValueError(
                f"FieldsplitSNES fields {repeated_fields} appear in more"
                f" than one split: {split_list}")

        missing_fields = [i for i in range(nfields)
                          if i not in specified_fields]

        # natural ordering with one field per split if no splits are specified
        if len(split_list) == 0 or snes_options.getBool("default", False):
            split_list.extend([i] for i in missing_fields)

        elif missing_fields:
            raise ValueError(
                f"FieldsplitSNES fields {missing_fields} are not in any split:"
                f" {split_list}. Set -snes_fieldsplit_default to put each"
                " of them in its own split.")

        return split_list

    def _get_default_parameters(
            self, ctx: _SNESContext,
            split_ctxs: Sequence[_SNESContext]) -> dict[str, Any]:
        """Return the default solver options for all splits.

        These are the options with the ``-fieldsplit_`` prefix that are
        not for a specific split, i.e. do not have a ``-fieldsplit_%d_``
        or ``-fieldsplit_fsname_`` prefix.

        They are passed to each split's
        :class:`~firedrake.variational_solver.NonlinearVariationalSolver`
        as ``solver_parameters``, so they take priority over Firedrake's
        default solver parameters. This must be called while the outer
        solver's options are in the global options database (i.e. during
        the outer solve), in which case the options for each specific split
        are also in the database so will take priority over these defaults.
        """
        from petsctools.options import DefaultOptionSet, get_default_options

        # The split prefixes are built from the options prefix of ctx
        # (see _SNESContext.split), so the defaults must be too.
        base_prefix = (ctx.options_prefix or "") + self._prefix
        split_prefixes = [split_ctx.options_prefix for split_ctx in split_ctxs]
        assert all(prefix.startswith(base_prefix) for prefix in split_prefixes)
        default_options_set = DefaultOptionSet(
            base_prefix=base_prefix,
            custom_prefix_endings=[prefix[len(base_prefix):]
                                   for prefix in split_prefixes])

        # Options for a numbered split that does not exist (e.g.
        # -fieldsplit_7_ when there are only 3 splits) are not defaults.
        return {
            k: v for k, v in get_default_options(default_options_set).items()
            if not re.match(r"\d+_", k)
        }

    def _get_nullspaces(
            self, ctx: _SNESContext,
            split_ctxs: Sequence[_SNESContext],
            splits: Sequence[Sequence[int]],
    ) -> list[VectorSpaceBasis | MixedVectorSpaceBasis | None]:
        """Return the nullspace for each split, or None if it has none."""
        from firedrake.nullspace import (
            VectorSpaceBasis, MixedVectorSpaceBasis)

        if ctx._nullspace is None:
            return [None for _ in splits]

        nullspaces = []
        for split_ctx, fields in zip(split_ctxs, splits):
            if len(fields) == 1:
                base = ctx._nullspace._bases[fields[0]]
                if isinstance(base, VectorSpaceBasis):
                    nullspaces.append(base)
                else:
                    nullspaces.append(None)

            else:
                W = split_ctx._x.function_space()
                mono_bases = [ctx._nullspace._bases[i] for i in fields]

                split_bases = [
                    base if isinstance(base, VectorSpaceBasis) else W.sub(k)
                    for k, base in enumerate(mono_bases)
                ]

                nullspaces.append(MixedVectorSpaceBasis(W, split_bases))

        return nullspaces

    def update(self, snes: PETSc.SNES) -> None:
        """Rebuild the splits if the SNES context has been replaced.

        All of the state built in :meth:`initialize` refers to the
        solution, forms and function spaces of the context on the
        SNES's DM. Changes to coefficient values are picked up
        automatically, but if the DM's context is replaced (for
        example by PETSc-driven mesh adaptivity) then the state is
        stale and must be rebuilt from scratch.
        """
        ctx = get_snesctx(snes.dm)
        # ctx is None if we are called outside of a solve.
        if ctx is not None and ctx is not self._ctx:
            self.initialize(snes)

    @PETSc.Log.EventDecorator()
    def step(self, snes: PETSc.SNES, x: PETSc.Vec,
             f: PETSc.Vec, y: PETSc.Vec) -> None:
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

    def view(self, snes: PETSc.SNES,
             viewer: PETSc.Viewer | None = None) -> None:
        """View information about this object with ``-snes_view``.
        """
        super().view(snes, viewer)
        if self.initialized:
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

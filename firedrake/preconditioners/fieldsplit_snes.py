from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx as get_snesctx

__all__ = ("FieldsplitSNES",)


class FieldsplitSNES(SNESBase):
    _prefix = "fieldsplit_"

    # TODO:
    #   -fieldsplit_ Allow setting default splits for unspecified fields
    #   -snes_fieldsplit_%d_fields Test setting field grouping/ordering like PCFieldsplit
    #   -snes_fieldsplit_default Allow setting default options for all splits

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
        # options for each field are "fieldsplit_%d"
        sub_prefix = outer_prefix + self._prefix

        snes_options = PETSc.Options(snes_prefix)
        self.fieldsplit_type = snes_options.getString('type', 'additive')
        if self.fieldsplit_type not in ('additive', 'multiplicative'):
            raise ValueError(
                'FieldsplitSNES option snes_fieldsplit_type must be'
                ' "additive" or "multiplicative"')

        self.fields = self._get_fields(snes_options, len(W))

        self.Gk = replace(ctx.F, {ctx._x: self.uk})
        self.b = Cofunction(W.dual())

        field_ctxs = ctx.split(self.fields)
        for field_ctx, fields in zip(field_ctxs, self.fields):
            V = field_ctx._x.function_space()
            if len(V) == 1:
                val = self.b.dat[fields[0]]
            else:
                val = MixedDat(self.b.dat[i] for i in fields)
            field_ctx.F -= Cofunction(V.dual(), val=val)

        # The solution for each split context views the data
        # of the relevant components of the solution of the
        # original context, so field_solver.solve() will also
        # update ctx._x of the outer snes.
        self.field_solvers = tuple(
            NonlinearVariationalSolver(
                field_ctx._problem, appctx=field_ctx.appctx,
                options_prefix=sub_prefix+str(i))
            for i, field_ctx in enumerate(field_ctxs)
        )

        outer_snes = snes
        for solver in self.field_solvers:
            field_snes = solver.snes
            field_snes.incrementTabLevel(1, parent=outer_snes)
            field_snes.ksp.incrementTabLevel(1, parent=outer_snes)
            field_snes.ksp.pc.incrementTabLevel(1, parent=outer_snes)

    def _get_fields(self, snes_options, nfields):
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

    def update(self, snes):
        pass

    @PETSc.Log.EventDecorator()
    def step(self, snes, x, f, y):
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
        # The field_solvers evaluate forms containing u, so for each
        # splitting type u needs to hold:
        #   - additive: all fields need to hold uk values
        #   - multiplicative: fields need to hold uk before
        #       they are are solved for, and keep the updated uk1
        #       values afterwards.
        uks = self.uk.subfunctions
        uk1s = self.uk1.subfunctions
        us = self.u.subfunctions
        for solver, fields in zip(self.field_solvers, self.fields):

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
        super().view(snes, viewer)
        if hasattr(self, "field_solvers"):
            viewer.printfASCII("SNES to solve for groups of fields separately.\n")
            viewer.printfASCII(f"  fieldsplit_type: {self.fieldsplit_type}\n")
            viewer.printfASCII(f"  total fields = {len(self.fields)}\n")
            for i, fields in enumerate(self.fields):
                viewer.printfASCII(f"  split {i} has fields {fields}\n")
            viewer.printfASCII("Solver info for each split is in the following SNES objects:\n")
            viewer.pushASCIITab()
            for i, (fields, field_solver) in enumerate(zip(self.fields, self.field_solvers)):
                viewer.printfASCII(f"Split number {i} with fields {fields}:\n")
                field_solver.snes.view(viewer)
            viewer.popASCIITab()

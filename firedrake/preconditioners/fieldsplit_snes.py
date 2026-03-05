from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx

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
            NonlinearVariationalSolver, Function)

        ctx = get_appctx(snes.dm)

        self.sol = ctx._x
        W = self.sol.function_space()

        # buffer to save solution to outer problem during solve
        self.sol_outer = Function(W)

        # buffers for shuffling solutions during solve
        self.sol_current = Function(W)
        self.sol_new = Function(W)

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
        field_ctxs = ctx.split(self.fields)

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

        # store current value of outer solution to restore
        # later in case it isn't the same as x.
        self.sol_outer.assign(self.sol)

        # make sure that self.sol in the full form in
        # ctx has the most up to date solution u^{k}.
        with self.sol.dat.vec_wo as vec:
            x.copy(vec)

        # save u^{k}
        self.sol_current.assign(self.sol)

        # The current snes solution x is held in sol_current, and we
        # will place the new solution in sol_new.
        # The field_solvers evaluate forms containing sol, so for each
        # splitting type sol needs to hold:
        #   - additive: all fields need to hold sol_current values
        #   - multiplicative: fields need to hold sol_current before
        #       they are are solved for, and keep the updated sol_new
        #       values afterwards.
        uk = self.sol_current.subfunctions
        uk1 = self.sol_new.subfunctions
        usol = self.sol.subfunctions
        for solver, fields, in zip(self.field_solvers, self.fields):
            solver.solve()

            # update the uk1 buffer
            for i in fields:
                uk1[i].assign(usol[i])

            # reset the values in this field
            if self.fieldsplit_type == 'additive':
                for i in fields:
                    usol[i].assign(uk[i])

        with self.sol_new.dat.vec_ro as vec:
            vec.copy(y)
            y.aypx(-1, x)

        # restore outer solution
        self.sol.assign(self.sol_outer)

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

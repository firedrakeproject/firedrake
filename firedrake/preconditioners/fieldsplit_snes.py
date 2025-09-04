from firedrake.preconditioners.base import SNESBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_appctx, get_function_space
from firedrake.function import Function
from firedrake import TestFunctions, inner, dx

__all__ = ("FieldsplitSNES",)


class FieldsplitSNES(SNESBase):
    prefix = "fieldsplit_"

    # TODO:
    #   - Allow setting field grouping/ordering like fieldsplit

    @PETSc.Log.EventDecorator()
    def initialize(self, snes):
        from firedrake.variational_solver import NonlinearVariationalSolver  # ImportError if we do this at file level
        ctx = get_appctx(snes.dm)
        W = get_function_space(snes.dm)
        self.sol = ctx._problem.u_restrict

        # buffer to save solution to outer problem during solve
        self.sol_outer = Function(self.sol.function_space())

        # buffers for shuffling solutions during solve
        self.sol_current = Function(self.sol.function_space())
        self.sol_new = Function(self.sol.function_space())

        # options for setting up the fieldsplit
        outer_prefix = snes.getOptionsPrefix() or ""
        snes_prefix = outer_prefix + 'snes_' + self.prefix
        # options for each field
        sub_prefix = outer_prefix + self.prefix

        snes_options = PETSc.Options(snes_prefix)
        self.fieldsplit_type = snes_options.getString('type', 'additive')
        if self.fieldsplit_type not in ('additive', 'multiplicative'):
            raise ValueError(
                'FieldsplitSNES option snes_fieldsplit_type must be'
                ' "additive" or "multiplicative"')

        self.fields = self._get_fields(snes_options, len(W))
        split_ctxs = ctx.split(self.fields)

        self.b = Function(W.dual())
        self.bu = Function(W)
        busubs = self.bu.subfunctions
        tests = TestFunctions(W)

        for fields, ctx in zip(self.fields, split_ctxs):
            for k in fields:
                ctx.F += inner(busubs[k], tests[k])*dx

        self.solvers = tuple(
            NonlinearVariationalSolver(
                ctx._problem, appctx=ctx.appctx,
                options_prefix=sub_prefix+str(i))
            for i, ctx in enumerate(split_ctxs)
        )

        outer_snes = snes
        for solver in self.solvers:
            inner_snes = solver.snes
            inner_snes.incrementTabLevel(1, parent=outer_snes)
            inner_snes.ksp.incrementTabLevel(1, parent=outer_snes)
            inner_snes.ksp.pc.incrementTabLevel(1, parent=outer_snes)

        print("End setup")

    def _get_fields(self, snes_options, nfields):
        split_opts = {}
        # extract split specification options
        for k in range(nfields):
            if snes_options.hasName(f"{k}_fields"):
                split_opts[k] = snes_options.getIntArray(f"{k}_fields")

        # natural ordering with one field per split if no splits are specified
        if len(split_opts) == 0:
            return [[k] for k in range(nfields)]

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
        print("Start step")

        # store current value of outer solution
        self.sol_outer.assign(self.sol)

        # the full form in ctx now has the most up to date solution
        with self.sol_current.dat.vec_wo as vec:
            x.copy(vec)
        self.sol.assign(self.sol_current)

        # forcing from outer residual
        if f is not None:
            with self.b.dat.vec as bvec:
                f.copy(bvec)
            self.bu.assign(self.b.riesz_representation())

        # The current snes solution x is held in sol_current, and we
        # will place the new solution in sol_new.
        # The solvers evaluate forms containing sol, so for each
        # splitting type sol needs to hold:
        #   - additive: all fields need to hold sol_current values
        #   - multiplicative: fields need to hold sol_current before
        #       they are are solved for, and keep the updated sol_new
        #       values afterwards.
        for solver, u, ucurr, unew in zip(self.solvers,
                                          self.sol.subfunctions,
                                          self.sol_current.subfunctions,
                                          self.sol_new.subfunctions):
            solver.solve()
            unew.assign(u)
            if self.fieldsplit_type == 'additive':
                u.assign(ucurr)

        with self.sol_new.dat.vec_ro as vec:
            vec.copy(y)
            y.aypx(-1, x)

        # restore outer solution
        self.sol.assign(self.sol_outer)
        print("End step")

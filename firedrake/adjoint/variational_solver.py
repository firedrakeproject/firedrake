from functools import wraps
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from firedrake.adjoint.blocks import NonlinearVariationalSolveBlock


class NonlinearVariationalProblemMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._ad_F = self.F
            self._ad_u = self.u
            self._ad_bcs = self.bcs
            self._ad_J = self.J
            self._ad_kwargs = {'Jp': self.Jp, 'form_compiler_parameters': self.form_compiler_parameters, 'is_linear': self.is_linear}
        return wrapper


class NonlinearVariationalSolverMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, problem, *args, **kwargs):
            init(self, problem, *args, **kwargs)
            self._ad_problem = problem
            self._ad_args = args
            self._ad_kwargs = kwargs

        return wrapper

    @staticmethod
    def _ad_annotate_solve(solve):
        @wraps(solve)
        def wrapper(self, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""

            annotate = annotate_tape(kwargs)
            if annotate:
                tape = get_working_tape()
                problem = self._ad_problem
                sb_kwargs = NonlinearVariationalSolveBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)
                block = NonlinearVariationalSolveBlock(problem._ad_F == 0,
                                                       problem._ad_u,
                                                       problem._ad_bcs,
                                                       problem_J=problem._ad_J,
                                                       solver_params=self.parameters,
                                                       solver_kwargs=self._ad_kwargs,
                                                       **sb_kwargs)
                tape.add_block(block)

            with stop_annotating():
                out = solve(self, **kwargs)

            if annotate:
                block.add_output(self._ad_problem._ad_u.create_block_variable())

            return out

        return wrapper

from functools import wraps
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from firedrake.adjoint.blocks import SolveLinearSystemBlock


class LinearSolverMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, A, *args, **kwargs):
            from firedrake import LinearSolver
            self.ad_block_tag = kwargs.pop("ad_block_tag", None)
            init(self, A, *args, **kwargs)
            self._ad_A = A
            self._ad_args = args
            self._ad_kwargs = kwargs
            kwargs["annotate"] = False
            self._ad_ls = LinearSolver(A, **kwargs)
            self._ad_kwargs.pop('P')

        return wrapper

    @staticmethod
    def _ad_annotate_solve(solve):
        @wraps(solve)
        def wrapper(self, x, b, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""

            annotate = annotate_tape(kwargs)
            if annotate:
                tape = get_working_tape()
                A = self._ad_A
                sb_kwargs = SolveLinearSystemBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)

                block = SolveLinearSystemBlock(A, x, b,
                                               forward_args=self._ad_args,
                                               forward_kwargs=self._ad_kwargs,
                                               ad_block_tag=self.ad_block_tag,
                                               **sb_kwargs)
                tape.add_block(block)
                block._ad_ls = self._ad_ls

            with stop_annotating():
                out = solve(self, x, b, **kwargs)

            if annotate:
                block.add_output(x.create_block_variable())

            return out

        return wrapper

from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape, no_annotations
from firedrake.adjoint.blocks import InterpolateBlock

def add_annotate_kwarg(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        annotate = annotate_tape(kwargs)
        if annotate:
            return func(*args, **kwargs)
        else:
            with stop_annotating():
                out = func(*args, **kwargs)
            return out

    return wrapper


class InterpolatorMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, expr, V, **kwargs):
            init(self, expr, V, **kwargs)
            self._ad_expr = expr

        return wrapper

    @staticmethod
    def _ad_annotate_interpolate(interpolate):
        @wraps(interpolate)
        def wrapper(self, *function, output=None, transpose=False, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False`
            this routine.  This is useful in cases where the interpolation is
            known to be irrelevant or diagnostic for the purposes of the
            adjoint computation."""
            annotate = annotate_tape(kwargs)
            if annotate:
                assert len(function) == 0
                tape = get_working_tape()
                block = InterpolateBlock(self._ad_expr)
                tape.add_block(block)

            with stop_annotating():
                result = interpolate(self, *function, output=output,
                                     transpose=transpose)

            if annotate:
                block.add_output(result.create_block_variable())

            return result

        return wrapper

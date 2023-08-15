from firedrake.adjoint_utils.blocks import InterpolateBlock
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape
from functools import wraps


def annotate_interpolate(interpolate):
    @wraps(interpolate)
    def wrapper(interpolator, *function, **kwargs):
        """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
        Firedrake interpolate call."""
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)

        if annotate:
            sb_kwargs = InterpolateBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = InterpolateBlock(interpolator, *function, ad_block_tag=ad_block_tag, **sb_kwargs)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            output = interpolate(interpolator, *function, **kwargs)

        if annotate:
            from firedrake import Function
            if isinstance(interpolator.V, Function):
                block.add_output(output.create_block_variable())
            else:
                block.add_output(output.block_variable)

        return output

    return wrapper

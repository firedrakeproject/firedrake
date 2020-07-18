from firedrake.adjoint.blocks import InterpolateBlock
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape
from functools import wraps


def annotate_interpolate(interpolate):
    @wraps(interpolate)
    def wrapper(interpolator, *function, **kwargs):
        annotate = annotate_tape(kwargs)

        if annotate:
            sb_kwargs = InterpolateBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = InterpolateBlock(interpolator, *function, **sb_kwargs)
            tape = get_working_tape()
            tape.add_block(block)

        with stop_annotating():
            output = interpolate(interpolator, *function, **kwargs)

        if annotate:
            block.add_output(output.create_block_variable())

        return output

    return wrapper

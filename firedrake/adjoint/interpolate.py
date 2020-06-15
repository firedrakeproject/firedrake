from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape

from firedrake.adjoint.blocks import InterpolateBlock

def annotate_interpolate(interpolate):
    def wrapper(*args, **kwargs):
        annotate = annotate_tape(kwargs)
        with stop_annotating():
            output = interpolate(*args, **kwargs)

        if annotate:
            sb_kwargs = InterpolateBlock.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = InterpolateBlock(*args, **sb_kwargs)

            tape = get_working_tape()
            tape.add_block(block)
            block.add_output(output.block_variable)

        return output

    return wrapper
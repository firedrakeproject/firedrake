from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from firedrake.adjoint.blocks import ProjectBlock, SupermeshProjectBlock
from firedrake import function


def annotate_project(project):
    @wraps(project)
    def wrapper(*args, **kwargs):
        """The project call performs an equation solve, and so it too must be annotated so that the
        adjoint and tangent linear models may be constructed automatically by pyadjoint.

        To disable the annotation of this function, just pass :py:data:`annotate=False`. This is useful in
        cases where the solve is known to be irrelevant or diagnostic for the purposes of the adjoint
        computation (such as projecting fields to other function spaces for the purposes of
        visualisation)."""

        annotate = annotate_tape(kwargs)
        if annotate:
            bcs = kwargs.get("bcs", [])
            sb_kwargs = ProjectBlock.pop_kwargs(kwargs)
            if isinstance(args[1], function.Function):
                # block should be created before project because output might also be an input that needs checkpointing
                output = args[1]
                V = output.function_space()
                if args[0].function_space().mesh() != V.mesh():
                    block = SupermeshProjectBlock(args[0], V, output, bcs, **sb_kwargs)
                else:
                    block = ProjectBlock(args[0], V, output, bcs, **sb_kwargs)

        with stop_annotating():
            output = project(*args, **kwargs)

        if annotate:
            tape = get_working_tape()
            if not isinstance(args[1], function.Function):
                if args[0].function_space().mesh() != args[1].mesh():
                    block = SupermeshProjectBlock(args[0], args[1], output, bcs, **sb_kwargs)
                else:
                    block = ProjectBlock(args[0], args[1], output, bcs, **sb_kwargs)
            tape.add_block(block)
            block.add_output(output.create_block_variable())

        return output

    return wrapper

from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from pyadjoint.overloaded_type import create_overloaded_object
from firedrake.adjoint.blocks import AssembleBlock


def annotate_assemble(assemble):
    @wraps(assemble)
    def wrapper(*args, **kwargs):
        """When a form is assembled, the information about its nonlinear dependencies is lost,
        and it is no longer easy to manipulate. Therefore, we decorate :func:`.assemble`
        to *attach the form to the assembled object*. This lets the automatic annotation work,
        even when the user calls the lower-level :py:data:`solve(A, x, b)`.
        """
        annotate = annotate_tape(kwargs)
        with stop_annotating():
            output = assemble(*args, **kwargs)

        form = args[0]
        if isinstance(output, float):
            if not annotate:
                return output

            output = create_overloaded_object(output)
            block = AssembleBlock(form)

            tape = get_working_tape()
            tape.add_block(block)

            block.add_output(output.block_variable)
        else:
            # Assembled a vector or matrix
            output.form = form

        return output

    return wrapper

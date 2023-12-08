import numbers
from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from pyadjoint.overloaded_type import create_overloaded_object
from firedrake.adjoint_utils.blocks import AssembleBlock


def annotate_assemble(assemble):
    @wraps(assemble)
    def wrapper(*args, **kwargs):
        """When a form is assembled, the information about its nonlinear dependencies is lost,
        and it is no longer easy to manipulate. Therefore, we decorate :func:`.assemble`
        to *attach the form to the assembled object*. This lets the automatic annotation work,
        even when the user calls the lower-level :py:data:`solve(A, x, b)`.
        """
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        with stop_annotating():
            output = assemble(*args, **kwargs)

        from firedrake.function import Function
        from firedrake.cofunction import Cofunction
        form = args[0]
        if isinstance(output, (numbers.Complex, Function, Cofunction)):
            # Assembling a 0-form or 1-form (e.g. Form)
            if not annotate:
                return output

            if not isinstance(output, (float, Function, Cofunction)):
                raise NotImplementedError("Taping for complex-valued 0-forms not yet done!")
            output = create_overloaded_object(output)
            block = AssembleBlock(form, ad_block_tag=ad_block_tag)

            tape = get_working_tape()
            tape.add_block(block)

            block.add_output(output.block_variable)
        else:
            # Assembled a 2-form
            output.form = form

        return output

    return wrapper

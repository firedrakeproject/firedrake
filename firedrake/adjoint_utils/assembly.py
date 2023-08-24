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
            # Assembling a 0-form or 1-form (e.g. Form or BaseFormOperator)
            if not annotate:
                return output

            if not isinstance(output, (float, Function, Cofunction)):
                raise NotImplementedError("Taping for complex-valued 0-forms not yet done!")

            tape = get_working_tape()

            """
            extops_form = form.external_operators()
            for coeff in form.coefficients():
                extops_coeff_form = [e.result_coefficient() for e in extops_form]
                dict_extops = dict(zip(extops_coeff_form, extops_form))
                if coeff in extops_coeff_form:
                    block_extops = PointwiseOperatorBlock(dict_extops[coeff], *args, **kwargs)
                    tape.add_block(block_extops)

                    block_variable = coeff.block_variable
                    block_extops.add_output(block_variable)
            """

            output = create_overloaded_object(output)
            block = AssembleBlock(form, ad_block_tag=ad_block_tag)
            tape.add_block(block)

            block.add_output(output.block_variable)
        else:
            # Assembled a 2-form
            output.form = form

        return output

    return wrapper

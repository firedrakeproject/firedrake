import numbers
from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from pyadjoint.overloaded_type import create_overloaded_object
from firedrake.adjoint_utils.blocks import AssembleBlock


def annotate_assemble(assemble):
    @wraps(assemble)
    def wrapper(form, *args, **kwargs):
        """When a form is assembled, the information about its nonlinear dependencies is lost,
        and it is no longer easy to manipulate. Therefore, we decorate :func:`.assemble`
        to *attach the form to the assembled object*. This lets the automatic annotation work,
        even when the user calls the lower-level :py:data:`solve(A, x, b)`.
        """
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        with stop_annotating():
            from firedrake.assemble import BaseFormAssembler
            from firedrake.slate import slate
            if not isinstance(form, slate.TensorBase):
                # Preprocess the form at the annotation stage so that the `AssembleBlock`
                # records the preprocessed form. This facilitates derivation of the tangent linear/adjoint models.
                # For example,
                # -> `interp = Action(Interpolate(v1, v0), f)` with `v1` and `v0` being respectively `Argument`
                # and `Coargument`. Differentiating `interp` is not currently supported as the action's left slot
                # is a 2-form. However, after preprocessing, we obtain `Interpolate(f, v0)`, which can be differentiated.
                form = BaseFormAssembler.preprocess_base_form(form)
                kwargs['is_base_form_preprocessed'] = True
            output = assemble(form, *args, **kwargs)

        from firedrake.function import Function
        from firedrake.cofunction import Cofunction
        if isinstance(output, (numbers.Complex, Function, Cofunction)):
            # Assembling a 0-form or 1-form (e.g. Form or BaseFormOperator)
            if not annotate:
                return output

            if not isinstance(output, (float, Function, Cofunction)):
                raise NotImplementedError("Taping for complex-valued 0-forms not yet done!")
            output = create_overloaded_object(output)
            block = AssembleBlock(form, ad_block_tag=ad_block_tag)

            tape = get_working_tape()
            tape.add_block(block)

            if kwargs.get("tensor") is not None:
                # Create a new block variable when a tensor is provided to the assembly.
                # This is necessary as this tensor may belong to the block dependency as well,
                # which would result in a cyclic dependency.
                # Example (self-interpolation):
                #  -> u.interpolate(u + c), with `u` a Function and `c` a Constant.
                block.add_output(output.create_block_variable())
            else:
                block.add_output(output.block_variable)
        else:
            # Assembled a 2-form
            output.form = form

        return output

    return wrapper

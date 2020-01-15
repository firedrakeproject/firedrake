import ufl
from pyadjoint.overloaded_type import create_overloaded_object, FloatingType
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape, no_annotations
from firedrake.adjoint.function import FunctionMixin
from firedrake.adjoint.blocks import PointwiseOperatorBlock
import firedrake


class PointwiseOperatorsMixin(FunctionMixin):

    @staticmethod
    def _ad_annotate_init(init):
        def wrapper(self, *args, **kwargs):
            FloatingType.__init__(self, *args,
                                  block_class=kwargs.pop("block_class", None),
                                  _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                  _ad_args=kwargs.pop("_ad_args", None),
                                  output_block_class=kwargs.pop("output_block_class", None),
                                  _ad_output_args=kwargs.pop("_ad_output_args", None),
                                  _ad_outputs=kwargs.pop("_ad_outputs", None), **kwargs)
            init(self, *args, **kwargs)
        return wrapper

    """
    @staticmethod
    def _ad_annotate_copy(copy):

        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)
            func = copy(self, *args, **kwargs)

            if annotate:
                if kwargs.pop("deepcopy", False):
                    block = PointwiseOperatorAssignBlock(func, self)#FunctionAssignBlock(func, self)
                    tape = get_working_tape()
                    tape.add_block(block)
                    block.add_output(func.create_block_variable())
                else:
                    # TODO: Implement. Here we would need to use floating types.
                    pass

            return func

        return wrapper

    @staticmethod
    def _ad_annotate_assign(assign):
        RAISEERROR
        def wrapper(self, other, *args, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Dolfin assign call."""

            # do not annotate in case of self assignment
            annotate = annotate_tape(kwargs) and self != other

            if annotate:
                if not isinstance(other, ufl.core.operator.Operator):
                    other = create_overloaded_object(other)
                block = PointwiseOperatorAssignBlock(self, other)#FunctionAssignBlock(self, other)
                tape = get_working_tape()
                tape.add_block(block)

            with stop_annotating():
                ret = assign(self, other, *args, **kwargs)

            if annotate:
                block.add_output(self.create_block_variable())

            return ret

        return wrapper

    def _ad_create_checkpoint(self):
        return self.copy(deepcopy=True)

    @staticmethod
    def _ad_annotate_evaluate(evaluate):
        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)

            with stop_annotating():
                output = evaluate()
                output = create_overloaded_object(output)

            if annotate:
                tape = get_working_tape()
                block = PointwiseEvalBlock(self, *args, **kwargs)
                tape.add_block(block)
                print("selong arg qui changera selon methode et type de pointop executera soit block different soit dans block method differente...")
                block.add_output(output.create_block_variable())

            return output
        return wrapper

    @staticmethod
    def _ad_annotate_compute_derivatives(compute_der):
        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)

            with stop_annotating():
                output = compute_der()
                output = create_overloaded_object(output)

            if annotate:
                tape = get_working_tape()
                block = PointwiseDerivativesBlock(self, *args, **kwargs)
                tape.add_block(block)

                block.add_output(output.create_block_variable())

            return output
        return wrapper
    """
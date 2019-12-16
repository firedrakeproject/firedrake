from pyadjoint import OverloadedType
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape

class FunctionMixin(OverloadedType):

    @staticmethod
    def _ad_annotate_init(init):
        def wrapper(self, *args, **kwargs):
            OverloadedType.__init__(self, *args,
                                           block_class=kwargs.pop("block_class",
                                                                  None),
                                           _ad_floating_active=kwargs.pop(
                                               "_ad_floating_active", False),
                                           _ad_args=kwargs.pop("_ad_args", None),
                                           output_block_class=kwargs.pop(
                                               "output_block_class", None),
                                           _ad_output_args=kwargs.pop(
                                               "_ad_output_args", None),
                                           _ad_outputs=kwargs.pop("_ad_outputs",
                                                                  None),
                                           annotate=kwargs.pop("annotate", True),
                                           **kwargs)
            init(self, *args, **kwargs)
        return wrapper


    @staticmethod
    def _ad_annotate_project(project):

        def wrapper(self, b, *args, **kwargs):
            
            annotate = annotate_tape(kwargs)
            
            with stop_annotating():
                output = project(self, b, *args, **kwargs)
            #output = create_overloaded_object(output)

            if annotate:
                from fenics_adjoint.projection import ProjectBlock
                bcs = kwargs.pop("bcs", [])
                block = ProjectBlock(b, self.function_space(), output, bcs)

                tape = get_working_tape()
                tape.add_block(block)

                block.add_output(output.create_block_variable())

            return output
        return wrapper

from pyadjoint import OverloadedType
from fenics_adjoint.projection import ProjectBlock
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape

class FunctionMixin(OverloadedType):

    @staticmethod
    def _ad_annotate_project(project):

        def wrapper(self, b, *args, **kwargs):
            
            annotate = annotate_tape(kwargs)
            
            with stop_annotating():
                output = project(self, b, *args, **kwargs)
            #output = create_overloaded_object(output)

            if annotate:
                bcs = kwargs.pop("bcs", [])
                block = ProjectBlock(b, self.function_space(), output, bcs)

                tape = get_working_tape()
                tape.add_block(block)

                block.add_output(output.create_block_variable())

            return output
        return wrapper
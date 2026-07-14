from functools import wraps
from pyadjoint.tape import annotate_tape, stop_annotating, get_working_tape
from firedrake.adjoint_utils.blocks import ProjectBlock, SupermeshProjectBlock
from firedrake import function
from ufl.domain import extract_unique_domain


def annotate_super_project(project):
    """
    A wrapper for SupermeshProjector.project(), only handles
    the code path that leads to the creation of a
    SupermeshProjectBlock.
    """

    @wraps(project)
    def wrapper(self, **kwargs):
        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)
        V = self.target.function_space()
        if annotate:
            bcs = kwargs.get("bcs", [])
            sb_kwargs = ProjectBlock.pop_kwargs(kwargs)
            if self._target_is_function:
                # block should be created before project because output might also be an input that needs checkpointing
                block = SupermeshProjectBlock(self.source, V, self.target, bcs, ad_block_tag=ad_block_tag, **sb_kwargs)

        with stop_annotating():
            output = project(self, **kwargs)

        if annotate:
            tape = get_working_tape()
            if not self._target_is_function:
                block = SupermeshProjectBlock(self.source, V, output, bcs, ad_block_tag=ad_block_tag, **sb_kwargs)
            tape.add_block(block)
            block.add_output(output.create_block_variable())

        return output

    return wrapper

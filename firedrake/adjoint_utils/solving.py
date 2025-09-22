from functools import wraps
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape

from firedrake.adjoint_utils.blocks import SolveVarFormBlock, SolveLinearSystemBlock, GenericSolveBlock, ProjectBlock
import ufl


def annotate_solve(solve):
    """This solve routine wraps the Firedrake :func:`.solve` call. Its purpose is to annotate the model,
    recording what solves occur and what forms are involved, so that the adjoint and tangent linear models may be
    constructed automatically by pyadjoint.

    To disable the annotation, just pass ``annotate=False`` to this routine, and it acts exactly like the
    Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
    for the purposes of the adjoint computation (such as projecting fields to other function spaces
    for the purposes of visualisation).

    The overloaded solve takes optional callback functions to extract adjoint solutions.
    All of the callback functions follow the same signature, taking a single argument of type Function.

    Keyword Args:
        adj_cb (:obj:`firedrake.function`, optional):
            callback function supplying the adjoint solution in the interior. The boundary values are zero.
        adj_bdy_cb (:obj:`firedrake.function`, optional):
            callback function supplying the adjoint solution on the boundary.
            The interior values are not guaranteed to be zero.
        adj2_cb (:obj:`firedrake.function`, optional):
            callback function supplying the second-order adjoint solution in the interior.
            The boundary values are zero.
        adj2_bdy_cb (:obj:`firedrake.function`, optional):
            callback function supplying the second-order adjoint solution on
            the boundary. The interior values are not guaranteed to be zero.
        ad_block_tag (:obj:`string`, optional):
            tag used to label the resulting block on the Pyadjoint tape. This
            is useful for identifying which block is associated with which equation in the forward model.

    """

    @wraps(solve)
    def wrapper(*args, **kwargs):

        ad_block_tag = kwargs.pop("ad_block_tag", None)
        annotate = annotate_tape(kwargs)

        if annotate:
            tape = get_working_tape()
            solve_block_type = SolveVarFormBlock
            if not isinstance(args[0], ufl.equation.Equation):
                solve_block_type = SolveLinearSystemBlock

            sb_kwargs = solve_block_type.pop_kwargs(kwargs)
            sb_kwargs.update(kwargs)
            block = solve_block_type(*args, ad_block_tag=ad_block_tag, **sb_kwargs)
            tape.add_block(block)

        with stop_annotating():
            output = solve(*args, **kwargs)

        if annotate:
            if hasattr(args[1], "create_block_variable"):
                block_variable = args[1].create_block_variable()
            else:
                block_variable = args[1].function.create_block_variable()
            block.add_output(block_variable)

        return output

    return wrapper


def get_solve_blocks():
    """
    Extract all blocks of the tape which correspond
    to PDE solves, except for those which correspond
    to calls of the ``project`` operator.
    """
    return [
        block
        for block in get_working_tape().get_blocks()
        if issubclass(type(block), GenericSolveBlock)
        and not issubclass(type(block), ProjectBlock)
    ]

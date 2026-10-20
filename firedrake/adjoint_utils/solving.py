from pyadjoint.tape import get_working_tape
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock


def get_solve_blocks():
    """
    Extract all blocks of the tape which correspond
    to PDE solves, except for those which correspond
    to calls of the ``project`` operator.
    """
    return [
        block
        for block in get_working_tape().get_blocks()
        if issubclass(type(block), NonlinearVariationalSolveBlock)
        and not block._is_project
    ]

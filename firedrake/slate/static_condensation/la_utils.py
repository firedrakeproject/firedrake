from collections import namedtuple
from pyop2.utils import as_tuple

from firedrake.slate import slate


LAContext = namedtuple("LAContext",
                       ["lhs", "rhs"])
LAContext.__doc__ = """\
Context information for systems of equations after
applying algebraic transformation via Slate-supported
operations. This object provides the symbolic expressions
for the transformed linear system of equations.

:param lhs: The resulting expression for the transformed
            left-hand side matrix.
:param rhs: The resulting expression for the transformed
            right-hand side vector.
"""


def condense_and_forward_eliminate(A, b, elim_fields):
    """Returns Slate expressions for the operator and
    right-hand side vector after eliminating specified
    unknowns.

    :arg A: a `slate.Tensor` corresponding to the
            mixed UFL operator.
    :arg b: a `slate.AssembledVector` corresponding
            to the right-hand side.
    :arg elim_fields: a `tuple` of indices denoting
                      which fields to eliminate.
    """

    if not isinstance(A, slate.Tensor):
        raise ValueError("Left-hand operator must be a Slate Tensor")

    if not isinstance(b, slate.AssembledVector):
        raise ValueError("Right-hand side must be an AssembledVector")

    # Ensures field indices are in increasing order
    elim_fields = list(as_tuple(elim_fields))
    elim_fields.sort()

    all_fields = list(range(len(A.arg_function_spaces[0])))

    condensed_field = list(set(all_fields) - set(elim_fields))

    _A = A.blocks
    _b = b.blocks

    # NOTE: Does not support non-contiguous field elimination
    e_idx0 = elim_fields[0]
    e_idx1 = elim_fields[-1]
    f_idx0 = condensed_field[0]
    f_idx1 = condensed_field[-1]

    # Finite element systems where static condensation
    # is possible have the general form:
    #
    #  | A_ee A_ef || x_e |   | b_e |
    #  |           ||     | = |     |
    #  | A_fe A_ff || x_f |   | b_f |
    #
    # where subscript `e` denotes the coupling with fields
    # that will be eliminated, and `f` denotes the condensed
    # fields.
    Aff = _A[f_idx0:f_idx1 + 1, f_idx0:f_idx1 + 1]
    Aef = _A[e_idx0:e_idx1 + 1, f_idx0:f_idx1 + 1]
    Afe = _A[f_idx0:f_idx1 + 1, e_idx0:e_idx1 + 1]
    Aee = _A[e_idx0:e_idx1 + 1, e_idx0:e_idx1 + 1]

    bf = _b[f_idx0:f_idx1 + 1]
    be = _b[e_idx0:e_idx1 + 1]

    # The reduced operator and right-hand side are:
    #  S = A_ff - A_fe * A_ee.inv * A_ef
    #  r = b_f - A_fe * A_ee.inv * b_e
    # as show in Slate:
    S = Aff - Afe * Aee.inv * Aef
    r = bf - Afe * Aee.inv * be

    return LAContext(lhs=S, rhs=r)

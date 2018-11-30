from collections import namedtuple
from pyop2.utils import as_tuple

from firedrake.slate import slate


LAContext = namedtuple("LAContext",
                       ["lhs", "rhs", "field_idx"])
LAContext.__doc__ = """\
Context information for systems of equations after
applying algebraic transformation via Slate-supported
operations. This object provides the symbolic expressions
for the transformed linear system of equations.

:param lhs: The resulting expression for the transformed
            left-hand side matrix.
:param rhs: The resulting expression for the transformed
            right-hand side vector.
:param field_idx: An integer or iterable of integers
                  (if the system is mixed) denoting
                  which field(s) the resulting solution
                  is defined on.
"""


def condense_and_forward_eliminate(A, b, elim_fields):
    """Returns Slate expressions for the operator and
    right-hand side vector after eliminating specified
    unknowns.

    :arg A: a `slate.Tensor` corresponding to the
            mixed UFL operator.
    :arg b: a `firedrake.Function` corresponding
            to the right-hand side.
    :arg elim_fields: a `tuple` of indices denoting
                      which fields to eliminate.
    """

    if not isinstance(A, slate.Tensor):
        raise ValueError("Left-hand operator must be a Slate Tensor")

    # Ensures field indices are in increasing order
    elim_fields = list(as_tuple(elim_fields))
    elim_fields.sort()

    all_fields = list(range(len(A.arg_function_spaces[0])))

    condensed_fields = list(set(all_fields) - set(elim_fields))
    condensed_fields.sort()

    _A = A.blocks
    _b = slate.AssembledVector(b).blocks

    # NOTE: Does not support non-contiguous field elimination
    e_idx0 = elim_fields[0]
    e_idx1 = elim_fields[-1]
    f_idx0 = condensed_fields[0]
    f_idx1 = condensed_fields[-1]

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
    field_idx = [idx for idx in range(f_idx0, f_idx1)]

    return LAContext(lhs=S, rhs=r, field_idx=field_idx)


def backward_solve(A, b, x, reconstruct_fields):
    """Returns a sequence of linear algebra contexts containing
    Slate expressions for backwards substitution.

    :arg A: a `slate.Tensor` corresponding to the
            mixed UFL operator.
    :arg b: a `firedrake.Function` corresponding
            to the right-hand side.
    :arg x: a `firedrake.Function` corresponding
            to the solution.
    :arg reconstruct_fields: a `tuple` of indices denoting
                             which fields to reconstruct.
    """

    if not isinstance(A, slate.Tensor):
        raise ValueError("Left-hand operator must be a Slate Tensor")

    all_fields = list(range(len(A.arg_function_spaces[0])))
    nfields = len(all_fields)
    reconstruct_fields = as_tuple(reconstruct_fields)

    _A = A.blocks
    _b = b.split()
    _x = x.split()

    # Ordering matters
    systems = []

    # Reconstruct one unknown from one determined field:
    #
    # | A_ee  A_ef || x_e |   | b_e |
    # |            ||     | = |     |
    # | A_fe  A_ff || x_f |   | b_f |
    #
    # where x_f is known from a previous computation.
    # Returns the system:
    #
    # A_ee x_e = b_e - A_ef * x_f.
    if nfields == 2:
        id_e, = reconstruct_fields
        id_f, = [idx for idx in all_fields if idx != id_e]

        A_ee = _A[id_e, id_e]
        A_ef = _A[id_e, id_f]
        b_e = slate.AssembledVector(_b[id_e])
        x_f = slate.AssembledVector(_x[id_f])

        r_e = b_e - A_ef * x_f
        local_system = LAContext(lhs=A_ee, rhs=r_e, field_idx=(id_e,))

        systems.append(local_system)

    # Reconstruct two unknowns from one determined field:
    #
    # | A_e0e0  A_e0e1  A_e0f || x_e0 |   | b_e0 |
    # | A_e1e0  A_e1e1  A_e1f || x_e1 | = | b_e1 |
    # | A_fe0   A_fe1   A_ff  || x_f  |   | b_f  |
    #
    # where x_f is the known field. Returns two systems to be
    # solved in order (determined from the reverse order of indices
    # e0 and e1):
    #
    # Solve for e1 first (obtained from eliminating x_e0):
    #
    # S_e1 x_e1 = r_e1
    #
    # where
    #
    # S_e1 = A_e1e1 - A_e1e0 * A_e0e0.inv * A_e0e1, and
    # r_e1 = b_e1 - A_e1e0 * A_e0e0.inv * b_e0
    #      - (A_e1f - A_e1e0 * A_e0e0.inv * A_e0f) * x_f,
    #
    # And then solve for x_e0 given x_f and x_e1:
    #
    # A_e0e0 x_e0 = b_e0 - A_e0e1 * x_e1 - A_e0f * x_f.
    elif nfields == 3:
        if len(reconstruct_fields) != nfields - 1:
            raise NotImplementedError("Implemented for 1 determined field")

        # Order of reconstruction doesn't need to be in order
        # of increasing indices
        id_e0, id_e1 = reconstruct_fields
        id_f, = [idx for idx in all_fields if idx not in reconstruct_fields]

        A_e0e0 = _A[id_e0, id_e0]
        A_e0e1 = _A[id_e0, id_e1]
        A_e1e0 = _A[id_e1, id_e0]
        A_e1e1 = _A[id_e1, id_e1]
        A_e0f = _A[id_e0, id_f]
        A_e1f = _A[id_e1, id_f]

        x_e1 = slate.AssembledVector(_x[id_e1])
        x_f = slate.AssembledVector(_x[id_f])

        b_e0 = slate.AssembledVector(_b[id_e0])
        b_e1 = slate.AssembledVector(_b[id_e1])

        # Solve for e1
        Sf = A_e1f - A_e1e0 * A_e0e0.inv * A_e0f
        S_e1 = A_e1e1 - A_e1e0 * A_e0e0.inv * A_e0e1
        r_e1 = b_e1 - A_e1e0 * A_e0e0.inv * b_e0 - Sf * x_f
        systems.append(LAContext(lhs=S_e1, rhs=r_e1, field_idx=(id_e1,)))

        # Solve for e0
        r_e0 = b_e0 - A_e0e1 * x_e1 - A_e0f * x_f
        systems.append(LAContext(lhs=A_e0e0, rhs=r_e0, field_idx=(id_e0,)))

    else:
        msg = "Not implemented for systems with %s fields" % nfields
        raise NotImplementedError(msg)

    return systems

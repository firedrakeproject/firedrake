"""This module provides assertion functions used by the SLATE language.
They are designed to provide specific and standardized error messages.
"""

__all__ = ['dimension_error', 'expecting_slate_object',
           'expecting_slate_expr', 'rank_error', 'slate_assert']


def dimension_error(shape_A, shape_B):
    raise ValueError("Cannot perform the operation of a (%d, %d)-shaped tensor with a (%d, %d)-shaped tensor." % (shape_A[0], shape_A[1], shape_B[0], shape_B[1]))


def expecting_slate_object(expected, given):
    raise ValueError("Expecting a %s instance, not a %s." % (type(expected), type(given)))


def expecting_slate_expr(given):
    raise ValueError("Expecting a SLATE expression, not %s." % type(given))


def rank_error(expected_rank, given_rank):
    raise ValueError("Expecting a form with %d argument(s) to create a rank-%d tensor. The form provided contains %d argument(s)." % (expected_rank, expected_rank, given_rank))


def slate_assert(condition, message):
    if not condition:
        raise ValueError(message)

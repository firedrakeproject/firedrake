"""This module provides some linear algebra functions for block matrix
algebra.
"""

from __future__ import absolute_import, print_function, division


__all__ = ["upper_schur_complement", "upper_woodbury_identity",
           "lower_schur_complement", "lower_woodbur_identity"]


def upper_schur_complement(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return (A - B * D.inv * C).inv


def lower_schur_complement(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return (D - C * A.inv * B).inv


def upper_woodbury_identity(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return -A.inv * B * (D - C * A.inv * B).inv


def lower_woodbur_identity(tensor):
    """
    """
    A, B, C, D = tensor.blocks.values()

    return -D.inv * C * (A - B * D.inv * C).inv

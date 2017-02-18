"""Provides an interface to Slate for compiling tensor expressions."""

from __future__ import absolute_import, print_function, division
from six import iteritems

from firedrake.slate.slate import TensorBase
from firedrake.slate.slac import compile_expression
from firedrake.tsfc_interface import SplitKernel


def compile_tensor_expr(expr, tsfc_parameters=None):
    """
    """
    if not isinstance(expr, TensorBase):
        raise RuntimeError(
            "Unable to convert object to a Slate tensor expression: %s"
            % repr(expr)
        )

    split_kernels = []
    for idx, tensor_expr in split_tensor(expr):
        if tensor_expr._metakinfo_cache is not None:
            kinfo = tensor_expr._metakinfo_cache

        else:
            kinfo = compile_expression(slate_expr=tensor_expr,
                                       tsfc_parameters=tsfc_parameters)

        split_kernels.append(SplitKernel(idx, kinfo))

    return tuple(split_kernels)


def split_tensor(expr):
    """
    """
    split_tensors = []
    for idx, tensor in iteritems(expr.blocks):
        split_tensors.append((idx, tensor))

    return split_tensors

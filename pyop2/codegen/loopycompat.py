# Everything in this file was formerly in loopy/transform/callable.py
# but was removed in https://github.com/inducer/loopy/pull/327. It has
# been kept here for compatibility but should be phased out.

# Note that since this code is copypasted, the linter has been turned off.

# flake8: noqa

from loopy.kernel.instruction import CallInstruction, MultiAssignmentBase, \
    CInstruction, _DataObliviousInstruction
from loopy.symbolic import CombineMapper, IdentityMapper
from loopy.symbolic import simplify_via_aff
from loopy.kernel.function_interface import CallableKernel
from loopy.translation_unit import TranslationUnit


# Tools to match caller to callee args by (guessed) automatic reshaping
#
# (This is undocumented and not recommended, but it is currently needed
# to support Firedrake.)

class DimChanger(IdentityMapper):
    """
    Mapper to change the dimensions of an argument.
    .. attribute:: callee_arg_dict
        A mapping from the argument name (:class:`str`) to instances of
        :class:`loopy.kernel.array.ArrayBase`.
    .. attribute:: desried_shape
        A mapping from argument name (:class:`str`) to an instance of
        :class:`tuple`.
    """
    def __init__(self, callee_arg_dict, desired_shape):
        self.callee_arg_dict = callee_arg_dict
        self.desired_shape = desired_shape
        super().__init__()

    def map_subscript(self, expr):
        if expr.aggregate.name not in self.callee_arg_dict:
            return super().map_subscript(expr)
        callee_arg_dim_tags = self.callee_arg_dict[expr.aggregate.name].dim_tags
        flattened_index = sum(dim_tag.stride*idx for dim_tag, idx in
                zip(callee_arg_dim_tags, expr.index_tuple))
        new_indices = []

        from operator import mul
        from functools import reduce
        stride = reduce(mul, self.desired_shape[expr.aggregate.name], 1)

        for length in self.desired_shape[expr.aggregate.name]:
            stride /= length
            ind = flattened_index // int(stride)
            flattened_index -= (int(stride) * ind)
            new_indices.append(simplify_via_aff(ind))

        return expr.aggregate[tuple(new_indices)]


def _match_caller_callee_argument_dimension_for_single_kernel(
        caller_knl, callee_knl):
    """
    :returns: a copy of *caller_knl* with the instance of
        :class:`loopy.kernel.function_interface.CallableKernel` addressed by
        *callee_function_name* in the *caller_knl* aligned with the argument
        dimensions required by *caller_knl*.
    """
    from loopy.kernel.array import ArrayBase
    from loopy.kernel.data import auto

    for insn in caller_knl.instructions:
        if not isinstance(insn, CallInstruction) or (
                insn.expression.function.name !=
                callee_knl.name):
            # Call to a callable kernel can only occur through a
            # CallInstruction.
            continue

        def _shape_1_if_empty(shape_caller, shape_callee):
            assert isinstance(shape_caller, tuple)
            if shape_caller == () and shape_caller!=shape_callee:
                return (1,)
            else:
                return shape_caller

        from loopy.kernel.function_interface import (
                ArrayArgDescriptor, get_arg_descriptor_for_expression,
                get_kw_pos_association)
        _, pos_to_kw = get_kw_pos_association(callee_knl)
        arg_id_to_shape = {}
        for arg_id, arg in insn.arg_id_to_arg().items():
            arg_id = pos_to_kw[arg_id]

            arg_descr = get_arg_descriptor_for_expression(caller_knl, arg)
            if isinstance(arg_descr, ArrayArgDescriptor):
                arg_id_to_shape[arg_id] = arg_descr.shape
            else:
                arg_id_to_shape[arg_id] = (1, )

        dim_changer = DimChanger(
                callee_knl.arg_dict,
                arg_id_to_shape)

        new_callee_insns = []
        for callee_insn in callee_knl.instructions:
            if isinstance(callee_insn, MultiAssignmentBase):
                new_callee_insns.append(callee_insn
                        .with_transformed_expressions(dim_changer))

            elif isinstance(callee_insn, (CInstruction,
                    _DataObliviousInstruction)):
                # The layout of the args to a CInstructions is not going to be matched to the caller_kernel,
                # they are appended with unmatched args.
                # We only use Cinstructions exceptionally, e.g. for adding profile instructions,
                # without arguments that required to be matched, so this is ok.
                new_callee_insns.append(callee_insn)
            else:
                raise NotImplementedError("Unknown instruction %s." %
                        type(insn))

        new_args = [arg if not isinstance(arg, ArrayBase)
                    else arg.copy(shape=arg_id_to_shape[arg.name],
                                  dim_tags=None, strides=auto, order="C")
                    for arg in callee_knl.args]

        # subkernel with instructions adjusted according to the new dimensions
        new_callee_knl = callee_knl.copy(instructions=new_callee_insns,
                                         args=new_args)

        return new_callee_knl


class _FunctionCalledChecker(CombineMapper):
    def __init__(self, func_name):
        self.func_name = func_name
        super().__init__()

    def combine(self, values):
        return any(values)

    def map_call(self, expr):
        if expr.function.name == self.func_name:
            return True
        return self.combine(
                tuple(
                    self.rec(child) for child in expr.parameters)
                )

    map_call_with_kwargs = map_call

    def map_constant(self, expr):
        return False

    def map_type_cast(self, expr):
        return self.rec(expr.child)

    def map_algebraic_leaf(self, expr):
        return False

    def map_kernel(self, kernel):
        return any(self.rec(insn.expression) for insn in kernel.instructions if
                isinstance(insn, MultiAssignmentBase))


def _match_caller_callee_argument_dimension_(program, callee_function_name):
    """
    Returns a copy of *program* with the instance of
    :class:`loopy.kernel.function_interface.CallableKernel` addressed by
    *callee_function_name* in the *program* aligned with the argument
    dimensions required by *caller_knl*.
    .. note::
        The callee kernel addressed by *callee_function_name*, should be
        called at only one location throughout the program, as multiple
        invocations would demand complex renaming logic which is not
        implemented yet.
    """
    assert isinstance(program, TranslationUnit)
    assert isinstance(callee_function_name, str)
    assert callee_function_name not in program.entrypoints
    assert callee_function_name in program.callables_table

    is_invoking_callee = _FunctionCalledChecker(
            callee_function_name).map_kernel

    caller_knl,  = [in_knl_callable.subkernel for in_knl_callable in
            program.callables_table.values() if isinstance(in_knl_callable,
                CallableKernel) and
            is_invoking_callee(in_knl_callable.subkernel)]

    from pymbolic.primitives import Call
    assert len([insn for insn in caller_knl.instructions if (isinstance(insn,
        CallInstruction) and isinstance(insn.expression, Call) and
        insn.expression.function.name == callee_function_name)]) == 1
    new_callee_kernel = _match_caller_callee_argument_dimension_for_single_kernel(
            caller_knl, program[callee_function_name])
    return program.with_kernel(new_callee_kernel)

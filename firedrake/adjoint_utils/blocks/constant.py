from pyadjoint import Block, OverloadedType
import numpy

from pyadjoint.reduced_functional_numpy import gather
from .block_utils import isconstant


def constant_from_values(constant, values=None):
    """Returns a new Constant with `constant.values()`.

    If the optional argument `values` is provided, then `values` will be the
    values of the new Constant instead, while still preserving the ufl_shape of
    `constant`.

    Args:
        constant: A constant with the ufl_shape to preserve. values
        (numpy.array): An optional argument to use instead of
        constant.values().

    Returns:
        Constant: The created Constant of the same type as `constant`.

    """
    values = constant.values() if values is None else values
    return type(constant)(numpy.reshape(values, constant.ufl_shape))


class ConstantAssignBlock(Block):
    def __init__(self, other, ad_block_tag=None):
        super(ConstantAssignBlock, self).__init__(ad_block_tag=ad_block_tag)
        self.assigned_float = False
        self.assigned_list = False
        if isinstance(other, OverloadedType):
            self.add_dependency(other)
            self.assigned_float = isinstance(other, float)
        else:
            # Assuming other is supplied as a numpy.ndarray with dtype=object.
            self.assigned_list = True
            self.dependency_to_index = []
            for i, v in enumerate(other.flat):
                if isinstance(v, OverloadedType):
                    self.add_dependency(v)
                    self.dependency_to_index.append(i)
            self.value = other.copy()

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        if self.assigned_list:
            return gather(adj_inputs[0])
        return None

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        adj_output = adj_inputs[0]
        if self.assigned_float:
            # Convert to float
            adj_output = gather(adj_output)
            adj_output = float(adj_output[0])
        elif self.assigned_list:
            adj_output = prepared[self.dependency_to_index[idx]]
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        values = tlm_inputs[0]
        if self.assigned_list:
            values = numpy.zeros(self.value.shape)
            for i, tlm_input in enumerate(tlm_inputs):
                values.flat[self.dependency_to_index[i]] = tlm_input
        elif isconstant(values):
            values = values.values()
        return constant_from_values(block_variable.output, values)

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs,
                                         relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        return self.evaluate_adj_component(inputs, hessian_inputs,
                                           block_variable, idx, prepared)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        values = inputs[0]
        if self.assigned_list:
            for i, inp in enumerate(inputs):
                self.value[self.dependency_to_index[i]] = inp
            values = self.value
        elif isconstant(values):
            values = values.values()
        return constant_from_values(block_variable.output, values)

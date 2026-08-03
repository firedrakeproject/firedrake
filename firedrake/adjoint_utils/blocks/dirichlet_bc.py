import ufl
import firedrake
from pyadjoint import Block, OverloadedType, no_annotations


def extract_bc_subvector(value, Vtarget, bc):
    """Extract from value (a function in a mixed space), the sub
    function corresponding to the part of the space bc applies
    to.  Vtarget is the target (collapsed) space."""
    r = value
    for idx in bc._indices:
        r = r.sub(idx)
    assert Vtarget == r.function_space()
    return r


class DirichletBCBlock(Block):
    def __init__(self, *args, **kwargs):
        Block.__init__(self, ad_block_tag=kwargs.pop('ad_block_tag', None))
        self.function_space = args[0]
        self.parent_space = self.function_space
        while (
            hasattr(self.parent_space, "_ad_parent_space")
            and self.parent_space._ad_parent_space is not None
        ):
            self.parent_space = self.parent_space._ad_parent_space
        self.collapsed_space = self.function_space
        if self.function_space != self.parent_space:
            self.collapsed_space = self.function_space.collapse()

        if len(args) >= 2 and isinstance(args[1], OverloadedType):
            # The dependency is the boundary value actually applied by the
            # BC: a Function on this space, either supplied directly or
            # produced (and taped) by interpolating the supplied expression.
            # Otherwise the value is a UFL zero or literal constant
            # expression and there is nothing to depend on.
            self.add_dependency(args[1])

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        bc = self.get_outputs()[0].saved_output
        c = block_variable.output
        adj_inputs = adj_inputs[0]
        adj_output = None
        for adj_input in adj_inputs:
            adj_value = firedrake.Function(self.parent_space)
            adj_input.apply(adj_value)
            r = extract_bc_subvector(
                adj_value, c.function_space(), bc
            ).riesz_representation("l2")
            if adj_output is None:
                adj_output = r
            else:
                adj_output += r

        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        bc = block_variable.saved_output
        for bv in self.get_dependencies():
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if self.function_space != self.parent_space and not isinstance(
                tlm_input, ufl.Coefficient
            ):
                tlm_input = firedrake.project(tlm_input, self.collapsed_space)

            # TODO: This is gonna crash for dirichletbcs with multiple
            #       dependencies (can't add two bcs) However, if there is
            #       multiple dependencies, we need to AD the expression (i.e if
            #       value=f*g then dvalue = tlm_f * g + f * tlm_g). Right now
            #       we can only handle value=f => dvalue = tlm_f.
            m = bc.reconstruct(g=tlm_input)
        return m

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # The same as evaluate_adj but with hessian values.
        return self.evaluate_adj_component(
            inputs, hessian_inputs, block_variable, idx
        )

    @no_annotations
    def recompute(self):
        # There is nothing to do. The checkpoint is weak,
        # so it changes automatically with the dependency checkpoint.
        return

    def __str__(self):
        return "DirichletBC block"

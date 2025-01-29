import ufl
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.formatting.ufl2unicode import ufl2unicode
from pyadjoint import Block, OverloadedType, AdjFloat
import firedrake
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint, \
    DelegatedFunctionCheckpoint
from .block_utils import isconstant


class FunctionAssignBlock(Block):
    def __init__(self, func, other, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.other = None
        self.expr = None
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        elif isinstance(other, float) or isinstance(other, int):
            other = AdjFloat(other)
            self.add_dependency(other, no_duplicates=True)
        elif not (isinstance(other, float) or isinstance(other, int)):
            # Assume that this is a point-wise evaluated UFL expression
            # (firedrake only)
            for op in traverse_unique_terminals(other):
                if isinstance(op, OverloadedType):
                    self.add_dependency(op, no_duplicates=True)
            self.expr = other

    def _replace_with_saved_output(self):
        if self.expr is None:
            return None

        replace_map = {}
        for dep in self.get_dependencies():
            replace_map[dep.output] = dep.saved_output
        return ufl.replace(self.expr, replace_map)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        adj_input_func, = adj_inputs
        if isinstance(adj_input_func, firedrake.Cofunction):
            adj_input_func = adj_input_func.riesz_representation(riesz_map="l2")

        if self.expr is None:
            return adj_input_func

        expr = self._replace_with_saved_output()
        return expr, adj_input_func

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            if isinstance(block_variable.output, AdjFloat):
                try:
                    # Adjoint of a broadcast is just a sum
                    return adj_inputs[0].dat.data_ro.sum()
                except AttributeError:
                    # Catch the case where adj_inputs[0] is just a float
                    return adj_inputs[0]
            elif isconstant(block_variable.output):
                adj_output = self._adj_assign_constant(
                    prepared, block_variable.output.function_space()
                )
            else:
                adj_output = firedrake.Function(
                    block_variable.output.function_space()
                )
                adj_output.assign(prepared)
            return adj_output.riesz_representation(riesz_map="l2")
        else:
            # Linear combination
            expr, adj_input_func = prepared
            if isconstant(block_variable.output):
                R = block_variable.output._ad_function_space(
                    adj_input_func.function_space().mesh()
                )
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output,
                                   firedrake.Function(R, val=1.0))
                )
                diff_expr_assembled = firedrake.Function(adj_input_func.function_space())
                diff_expr_assembled.interpolate(ufl.conj(diff_expr))
                diff_expr_assembled = diff_expr_assembled.riesz_representation(riesz_map="l2")
                adj_output = firedrake.Function(
                    R, val=firedrake.assemble(ufl.Action(diff_expr_assembled, adj_input_func))
                )
            else:
                adj_output = firedrake.Function(adj_input_func.function_space())
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, adj_input_func)
                )
                adj_output.interpolate(ufl.conj(diff_expr))
            return adj_output.riesz_representation(riesz_map="l2")

    def _adj_assign_constant(self, adj_output, constant_fs):
        r = firedrake.Function(constant_fs)
        shape = r.ufl_shape
        if shape == () or shape[0] == 1:
            # Scalar Constant
            r.dat.data[:] = adj_output.dat.data_ro.sum()
        else:
            # We assume the shape of the constant == shape of the output
            # function if not scalar. This assumption is due to FEniCS not
            # supporting products with non-scalar constants in assign.
            values = []
            for i in range(shape[0]):
                values.append(adj_output.sub(i, deepcopy=True).dat.data_ro.sum())
            r.assign(firedrake.Constant(values))
        return r

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if self.expr is None:
            return None

        return self._replace_with_saved_output()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            return tlm_inputs[0]

        expr = prepared
        dudm = firedrake.Function(block_variable.output.function_space())
        dudmi = firedrake.Function(block_variable.output.function_space())
        for dep in self.get_dependencies():
            if dep.tlm_value:
                dudmi.assign(ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, dep.saved_output,
                                   dep.tlm_value)))
                dudm.dat += 1.0 * dudmi.dat

        return dudm

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs,
                                         relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        # Current implementation assumes lincom in hessian,
        # otherwise we need second-order derivatives here.
        return self.evaluate_adj_component(inputs, hessian_inputs,
                                           block_variable, idx, prepared)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        if self.expr is None:
            return None
        return self._replace_with_saved_output()

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        """Recompute the assignment.

        Parameters
        ----------
        inputs : list of Function or Constant
            The variables in the RHS of the assignment.
        block_variable : pyadjoint.block_variable.BlockVariable
            The output block variable.
        idx : int
            Index associated to the inputs list.
        prepared :
            The precomputed RHS value.

        Notes
        -----
        Recomputes the block_variable only if the checkpoint was not delegated
        to another :class:`~firedrake.function.Function`.

        Returns
        -------
        Function
            Return either the firedrake function or `BlockVariable` checkpoint
            to which was delegated the checkpointing.
        """
        if isinstance(block_variable.checkpoint, DelegatedFunctionCheckpoint):
            return block_variable.checkpoint
        else:
            if self.expr is None:
                prepared = inputs[0]
            output = firedrake.Function(
                block_variable.output.function_space()
            )
            output.assign(prepared)
            return maybe_disk_checkpoint(output)

    def __str__(self):
        rhs = self.expr or self.other or self.get_dependencies()[0].output
        if isinstance(rhs, ufl.core.expr.Expr):
            rhs_str = ufl2unicode(rhs)
        else:
            rhs_str = str(rhs)
        return f"assign({rhs_str})"


class SubfunctionBlock(Block):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        eval_adj = firedrake.Cofunction(block_variable.output.function_space().dual())
        if type(adj_inputs[0]) is firedrake.Cofunction:
            eval_adj.sub(self.idx).assign(adj_inputs[0])
        else:
            eval_adj.sub(self.idx).assign(adj_inputs[0].function)
        return eval_adj

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return firedrake.Function.sub(tlm_inputs[0], self.idx)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        eval_hessian = firedrake.Cofunction(block_variable.output.function_space().dual())
        eval_hessian.sub(self.idx).assign(hessian_inputs[0])
        return eval_hessian

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return maybe_disk_checkpoint(
            firedrake.Function.sub(inputs[0], self.idx)
        )

    def __str__(self):
        return f"{self.get_dependencies()[0]}[{self.idx}]"


class FunctionMergeBlock(Block):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx
        for output in func._ad_outputs:
            self.add_dependency(output)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if idx == 0:
            return adj_inputs[0].subfunctions[self.idx]
        else:
            return adj_inputs[0]

    def evaluate_tlm(self, markings=False):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        output = self.get_outputs()[0]
        if markings and not output.marked_in_path:
            return
        fs = output.output.function_space()
        f = type(output.output)(fs)
        output.add_tlm_output(
            type(output.output).assign(f.sub(self.idx), tlm_input)
        )

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        sub_func = inputs[0]
        parent_in = inputs[1]
        parent_out = type(parent_in)(parent_in)
        parent_out.sub(self.idx).assign(sub_func)
        return maybe_disk_checkpoint(parent_out)

    def __str__(self):
        deps = self.get_dependencies()
        return f"{deps[1]}[{self.idx}].assign({deps[0]})"


class CofunctionAssignBlock(Block):
    """Class specifically for the case b.assign(a).

    All other cofunction assignment operations are annotated via Assemble. In
    effect this means that this is the annotation of an identity operation.

    Parameters
    ----------
    lhs:
        The target of the assignment.
    rhs:
        The cofunction being assigned.
    """

    def __init__(self, lhs: firedrake.Cofunction, rhs: firedrake.Cofunction,
                 ad_block_tag=None, rhs_from_assemble=False):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_output(lhs.block_variable)
        self.add_dependency(rhs)
        if rhs_from_assemble:
            # The `rhs_from_assemble` flag is set to `True` only when the
            # previous block is an Assemble Block, which results from the
            # Firedrake development API and not something implemented for
            # the user.

            # Checkpoint should be created at this point.
            assert self._dependencies[0].checkpoint is not None
            # When `rhs` is a output of an Assemble Block, there is no
            # need to duplicate the output with checkpoint data.
            # For further clarification, see how the `rhs_from_assemble` flag
            # is set in the `firedrake.CoFunction.assign` method.
            self._dependencies[0].output = DelegatedFunctionCheckpoint(
                self._dependencies[0])

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        """Recompute the assignment.

        Parameters
        ----------
        inputs : list of Function or Constant
            The variable in the RHS of the assignment.
        block_variable : pyadjoint.block_variable.BlockVariable
            The output block variable.
        idx : int
            Index associated to the inputs list.
        prepared :
            The precomputed RHS value.

        Notes
        -----
        Recomputes the block_variable only if the checkpoint was not delegated
        to another :class:`~firedrake.function.Function`.

        Returns
        -------
        Cofunction
            Return either the firedrake cofunction or `BlockVariable`
            checkpoint to which was delegated the checkpointing.
        """
        assert idx == 0  # There must be only one RHS.
        if isinstance(block_variable.checkpoint, DelegatedFunctionCheckpoint):
            return block_variable.checkpoint
        else:
            output = firedrake.Cofunction(
                block_variable.output.function_space()
            )
            output.assign(inputs[0])
            return maybe_disk_checkpoint(output)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return adj_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return tlm_inputs[0]

    def __str__(self):
        deps = self.get_dependencies()
        return f"assign({deps[0]})"

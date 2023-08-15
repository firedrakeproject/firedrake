import ufl
from ufl import replace
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.formatting.ufl2unicode import ufl2unicode
from ufl.algorithms.analysis import extract_arguments_and_coefficients
from pyadjoint import Block, OverloadedType, AdjFloat
import firedrake
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint, \
    DelegatedFunctionCheckpoint
from .backend import Backend


class FunctionAssignBlock(Block, Backend):
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
        V = self.get_outputs()[0].output.function_space()
        adj_input_func = self.compat.function_from_vector(V, adj_inputs[0])

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
                    return adj_inputs[0].sum()
                except AttributeError:
                    # Catch the case where adj_inputs[0] is just a float
                    return adj_inputs[0]
            elif self.compat.isconstant(block_variable.output):
                R = block_variable.output._ad_function_space(
                    prepared.function_space().mesh()
                )
                return self._adj_assign_constant(prepared, R)
            else:
                adj_output = self.backend.Function(
                    block_variable.output.function_space())
                adj_output.assign(prepared)
                return adj_output.vector()
        else:
            # Linear combination
            expr, adj_input_func = prepared
            adj_output = self.backend.Function(adj_input_func.function_space())
            if not self.compat.isconstant(block_variable.output):
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        expr, block_variable.saved_output, adj_input_func
                    )
                )
                adj_output.assign(diff_expr)
            else:
                mesh = adj_output.function_space().mesh()
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(
                        expr,
                        block_variable.saved_output,
                        self.compat.create_constant(1., domain=mesh)
                    )
                )
                adj_output.assign(diff_expr)
                return adj_output.vector().inner(adj_input_func.vector())

            if self.compat.isconstant(block_variable.output):
                R = block_variable.output._ad_function_space(
                    adj_output.function_space().mesh()
                )
                return self._adj_assign_constant(adj_output, R)
            else:
                return adj_output.vector()

    def _adj_assign_constant(self, adj_output, constant_fs):
        r = self.backend.Function(constant_fs)
        shape = r.ufl_shape
        if shape == () or shape[0] == 1:
            # Scalar Constant
            r.vector()[:] = adj_output.vector().sum()
        else:
            # We assume the shape of the constant == shape of the output
            # function if not scalar. This assumption is due to FEniCS not
            # supporting products with non-scalar constants in assign.
            values = []
            for i in range(shape[0]):
                values.append(adj_output.sub(i, deepcopy=True).vector().sum())
            r.assign(self.backend.Constant(values))
        return r.vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if self.expr is None:
            return None

        return self._replace_with_saved_output()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if self.expr is None:
            return tlm_inputs[0]

        expr = prepared
        dudm = self.backend.Function(block_variable.output.function_space())
        dudmi = self.backend.Function(block_variable.output.function_space())
        for dep in self.get_dependencies():
            if dep.tlm_value:
                dudmi.assign(ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, dep.saved_output,
                                   dep.tlm_value)))
                dudm.vector().axpy(1.0, dudmi.vector())

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
            output = self.backend.Function(
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


class InterpolateBlock(Block, Backend):
    r"""
    Annotates an interpolator.

    Consider the block as f with 1 forward model output ``v``, and inputs ``u``
    and ``g`` (there can, in principle, be any number of outputs). The adjoint
    input is ``vhat`` (``uhat`` and ``ghat`` are adjoints to ``u`` and ``v``
    respectively and are shown for completeness). The downstream block is ``J``
    which has input ``v``.

    ::

         _             _
        |J|--<--v--<--|f|--<--u--<--...
         ¯      |      ¯      |
               vhat    |     uhat
                       |
                        ---<--g--<--...
                              |
                             ghat

    (Arrows indicate forward model direction)

    ::

        J : V ⟶ R i.e. J(v) ∈ R ∀ v ∈ V

    Interpolation can operate on an expression which may not be linear in its
    arguments.

    ::

        f : W × G ⟶ V i.e. f(u, g) ∈ V ∀ u ∈ W and g ∈ G. f = I ∘ expr I :   X
        ⟶ V i.e. I(;x) ∈ V ∀ x ∈ X.
                          X is infinite dimensional.
        expr: W × G ⟶ X i.e. expr(u, g) ∈ X ∀ u ∈ W and g ∈ G.

    Arguments after a semicolon are linear (i.e. operation I is linear)
    """
    def __init__(self, interpolator, *functions, **kwargs):
        super().__init__(ad_block_tag=kwargs.pop("ad_block_tag", None))

        self.expr = interpolator.expr
        self.arguments, self.coefficients = \
            extract_arguments_and_coefficients(self.expr)

        if isinstance(interpolator.V, firedrake.Function):
            self.V = interpolator.V.function_space()
        else:
            self.V = interpolator.V

        for coefficient in self.coefficients:
            self.add_dependency(coefficient, no_duplicates=True)

        for function in functions:
            self.add_dependency(function, no_duplicates=True)

    def _replace_map(self):
        # Replace the dependencies with checkpointed values
        replace_map = {}
        args = 0
        for block_variable in self.get_dependencies():
            output = block_variable.output
            if output in self.coefficients:
                replace_map[output] = block_variable.saved_output
            else:
                replace_map[self.arguments[args]] = block_variable.saved_output
                args += 1
        return replace_map

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        r"""
        Denote ``d_u[A]`` as the gateaux derivative in the ``u`` direction.
        Arguments after a semicolon are linear.

        This calculates

        ::

            uhat = vhat ⋅ d_u[f](u, g; ⋅) (for inputs[idx] ∈ W) or ghat = vhat
            ⋅ d_g[f](u, g; ⋅) (for inputs[idx] ∈ G)

        where ``inputs[idx]`` specifies the derivative direction, ``vhat`` is
        ``adj_inputs[0]`` (since we assume only one block output) and ``⋅``
        denotes an unspecified operand of ``u'`` (for ``inputs[idx]`` ∈ ``W``)
        or ``g'`` (for ``inputs[idx]`` ∈ ``G``) size (``vhat`` left multiplies
        the derivative).

        ::

            f = I ∘ expr : W × G ⟶ V
                           i.e. I(expr|_{u, g}) ∈ V ∀ u ∈ W, g ∈ G.

        Since ``I`` is linear we get that

        ::

            d_u[I ∘ expr](u, g; u') = I ∘ d_u[expr](u|_u, g|_g; u') d_g[I ∘
            expr](u, g; g') = I ∘ d_u[expr](u|_u, g|_g; g').

        In tensor notation

        ::

            uhat_q^T = vhat_p^T I([dexpr/du|_u]_q)_p or ghat_q^T = vhat_p^T
            I([dexpr/dg|_u]_q)_p

        the output is then

        ::

            uhat_q = I^T([dexpr/du|_u]_q)_p vhat_p or ghat_q =
            I^T([dexpr/dg|_u]_q)_p vhat_p.
        """
        if len(adj_inputs) > 1:
            raise NotImplementedError(
                "Interpolate block must have a single output"
            )
        dJdm = self.backend.derivative(prepared, inputs[idx])
        return self.backend.Interpolator(dJdm, self.V).interpolate(
            adj_inputs[0], transpose=True
        ).vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        r"""
        Denote ``d_u[A]`` as the gateaux derivative in the ``u`` direction.
        Arguments after a semicolon are linear.

        For a block with two inputs this calculates

        ::

            v' = d_u[f](u, g; u') + d_g[f](u, g; g')

        where ``u' = tlm_inputs[0]`` and ``g = tlm_inputs[1]``.

        ::

            f = I ∘ expr : W × G ⟶ V
                           i.e. I(expr|_{u, g}) ∈ V ∀ u ∈ W, g ∈ G.

        Since ``I`` is linear we get that

        ::

            d_u[I ∘ expr](u, g; u') = I ∘ d_u[expr](u|_u, g|_g; u')
            d_g[I ∘ expr](u, g; g') = I ∘ d_u[expr](u|_u, g|_g; g').

        In tensor notation the output is then

        ::

            v'_l = I([dexpr/du|_{u,g}]_k u'_k)_l
                    + I([dexpr/du|_{u,g}]_k g'_k)_l
                 = I([dexpr/du|_{u,g}]_k u'_k + [dexpr/du|_{u,g}]_k g'_k)_l

        since ``I`` is linear.
        """
        dJdm = 0.

        assert len(inputs) == len(tlm_inputs)
        for i, input in enumerate(inputs):
            if tlm_inputs[i] is None:
                continue
            dJdm += self.backend.derivative(prepared, input, tlm_inputs[i])
        return self.backend.Interpolator(dJdm, self.V).interpolate()

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs,
                                         relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        r"""
        Denote ``d_u[A]`` as the gateaux derivative in the ``u`` direction.
        Arguments after a semicolon are linear.

        hessian_input is ``d_v[d_v[J]](v; v', ⋅)`` where the direction ``⋅`` is
        left unspecified so it can be operated upon.

        .. warning::
            NOTE: This comment describes the implementation of 1 block input
            ``u``. (e.g. interpolating from an expression with 1 coefficient).
            Explaining how this works for multiple block inputs (e.g. ``u``
            and ``g``) is currently too complicated for the author to do
            succinctly!

        This function needs to output ``d_u[d_u[J ∘ f]](u; u', ⋅)`` where the
        direction ``⋅`` will be specified in another function and multiplied on
        the right with the output of this function. We will calculate this
        using the chain rule.

        ::

            J : V ⟶ R i.e. J(v) ∈ R ∀ v ∈ V f = I ∘ expr : W ⟶ V J ∘ f : W ⟶ R
            i.e. J(f|u) ∈ R ∀ u ∈ V. d_u[J ∘ f] : W × W ⟶ R i.e. d_u[J ∘ f](u;
            u') d_u[d_u[J ∘ f]] : W × W × W ⟶ R i.e. d_u[d_u[J ∘ f]](u; u',
            u'') d_v[J] : V × V ⟶ R i.e. d_v[J](v; v') d_v[d_v[J]] : V × V × V
            ⟶ R i.e. d_v[d_v[J]](v; v', v'')

        Chain rule:

        ::

            d_u[J ∘ f](u; u') = d_v[J](v = f|u; v' = d_u[f](u; u'))

        Multivariable chain rule:

        ::

            d_u[d_u[J ∘ f]](u; u', u'') = d_v[d_v[J]](v = f|u; v' = d_u[f](u;
            u'), v'' = d_u[f](u; u'')) + d_v'[d_v[J]](v = f|u; v' = d_u[f](u;
            u'), v'' = d_u[d_u[f]](u; u', u'')) = d_v[d_v[J]](v = f|u; v' =
            d_u[f](u; u'), v''=d_u[f](u; u'')) + d_v[J](v = f|u; v' = v'' =
            d_u[d_u[f]](u; u', u''))

        since ``d_v[d_v[J]]`` is linear in ``v'`` so differentiating wrt to it
        leaves its coefficient, the bare d_v[J] operator which acts on the
        ``v''`` term that remains.

        The ``d_u[d_u[f]](u; u', u'')`` term can be simplified further:

        ::

            f = I ∘ expr : W ⟶ V i.e. I(expr|u) ∈ V ∀ u ∈ W d_u[I ∘ expr] : W ×
            W ⟶ V i.e. d_u[I ∘ expr](u; u') d_u[d_u[I ∘ expr]] : W × W × W ⟶ V
            i.e. d_u[I ∘ expr](u; u', u'') d_x[I] : X × X ⟶ V i.e. d_x[I](x;
            x') d_x[d_x[I]] : X × X × X ⟶ V i.e. d_x[d_x[I]](x; x', x'')
            d_u[expr] : W × W ⟶ X i.e. d_u[expr](u; u') d_u[d_u[expr]] : W × W
            × W ⟶ X i.e. d_u[d_u[expr]](u; u', u'')

        Since ``I`` is linear we get that

        ::

            d_u[d_u[I ∘ expr]](u; u', u'') = I ∘ d_u[d_u[expr]](u; u', u'').

        So our full hessian is:

        ::

            d_u[d_u[J ∘ f]](u; u', u'') = d_v[d_v[J]](v = f|u; v' = d_u[f](u;
            u'), v''=d_u[f](u; u'')) + d_v[J](v = f|u; v' = v'' =
            d_u[d_u[f]](u; u', u''))

        In tensor notation

        ::

            [d^2[J ∘ f]/du^2|_u]_{lk} u'_k u''_k = [d^2J/dv^2|_{v=f|_u}]_{ij}
            [df/du|_u]_{jk} u'_k [df/du|_u]_{il} u''_l + [dJ/dv|_{v=f_u}]_i
            I([d^2expr/du^2|_u]_{lk} u'_k)_i u''_l

        In the first term:

        ::

            [df/du|_u]_{jk} u'_k = v'_j => [d^2J/dv^2|_{v=f|_u}]_{ij}
            [df/du|_u]_{jk} u'_k = [d^2J/dv^2|_{v=f|_u}]_{ij} v'_j =
            hessian_input_i => [d^2J/dv^2|_{v=f|_u}]_{ij} [df/du|_u]_{jk} u'_k
            [df/du|_u]_{il} = hessian_input_i [df/du|_u]_{il} =
            self.evaluate_adj_component(inputs, hessian_inputs, ...)_l

        In the second term we calculate everything explicitly though note
        ``[dJ/dv|_{v=f_u}]_i = adj_inputs[0]_i``

        Also, the second term is 0 if ``expr`` is linear.
        """

        if len(hessian_inputs) > 1 or len(adj_inputs) > 1:
            raise NotImplementedError(
                "Interpolate block must have a single output"
            )

        component = self.evaluate_adj_component(inputs, hessian_inputs,
                                                block_variable, idx, prepared)

        # Prepare again by replacing expression
        expr = replace(self.expr, self._replace_map())

        # Calculate first derivative for each relevant block
        dexprdu = 0.
        for _, bv in relevant_dependencies:
            # Only take the derivative if there is a direction to take it in
            if bv.tlm_value is None:
                continue
            dexprdu += self.backend.derivative(expr, bv.saved_output,
                                               bv.tlm_value)

        # Calculate the second derivative w.r.t. the specified coefficient's
        # saved value. Leave argument unspecified so it can be calculated with
        # the eventual inner product with u''.
        d2exprdudu = self.backend.derivative(dexprdu,
                                             block_variable.saved_output)

        # left multiply by dJ/dv (adj_inputs[0]) - i.e. interpolate using the
        # transpose operator
        component += self.backend.Interpolator(d2exprdudu, self.V).interpolate(
            adj_inputs[0], transpose=True
        )
        return component.vector()

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def recompute_component(self, inputs, block_variable, idx, prepared):
        result = self.backend.interpolate(prepared, self.V)
        if isinstance(result, firedrake.Function):
            return maybe_disk_checkpoint(result)
        else:
            return result

    def __str__(self):
        target_string = f"〈{str(self.V.ufl_element().shortstr())}〉"
        return f"interpolate({self.expr},  {target_string})"


class SubfunctionBlock(Block, Backend):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        eval_adj = self.backend.Function(
            block_variable.output.function_space()
        )
        if type(adj_inputs[0]) is self.backend.Function:
            eval_adj.sub(self.idx).assign(adj_inputs[0])
        else:
            eval_adj.sub(self.idx).assign(adj_inputs[0].function)
        return eval_adj.vector()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return self.backend.Function.sub(tlm_inputs[0], self.idx)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        eval_hessian = self.backend.Function(
            block_variable.output.function_space()
        )
        eval_hessian.sub(self.idx).assign(hessian_inputs[0].function)
        return eval_hessian.vector()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return maybe_disk_checkpoint(
            self.backend.Function.sub(inputs[0], self.idx)
        )

    def __str__(self):
        return f"{self.get_dependencies()[0]}[{self.idx}]"


class FunctionMergeBlock(Block, Backend):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx
        for output in func._ad_outputs:
            self.add_dependency(output)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if idx == 0:
            return adj_inputs[0].subfunctions[self.idx].vector()
        else:
            return adj_inputs[0]

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        output = self.get_outputs()[0]
        fs = output.output.function_space()
        f = self.backend.Function(fs)
        output.add_tlm_output(
            self.backend.Function.assign(f.sub(self.idx), tlm_input)
        )

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        sub_func = inputs[0]
        parent_in = inputs[1]
        parent_out = self.backend.Function(parent_in)
        parent_out.sub(self.idx).assign(sub_func)
        return maybe_disk_checkpoint(parent_out)

    def __str__(self):
        deps = self.get_dependencies()
        return f"{deps[1]}[{self.idx}].assign({deps[0]})"

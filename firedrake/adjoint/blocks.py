from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks
from pyadjoint.block import Block
from ufl.algorithms.analysis import extract_arguments_and_coefficients
from ufl import replace

import firedrake
import firedrake.utils as utils


class Backend:
    @utils.cached_property
    def backend(self):
        import firedrake
        return firedrake

    @utils.cached_property
    def compat(self):
        import firedrake
        return compat(firedrake)


class DirichletBCBlock(blocks.DirichletBCBlock, Backend):
    pass


class ConstantAssignBlock(blocks.ConstantAssignBlock, Backend):
    pass


class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
    pass


class AssembleBlock(blocks.AssembleBlock, Backend):
    pass


def solve_init_params(self, args, kwargs, varform):
    if len(self.forward_args) <= 0:
        self.forward_args = args
    if len(self.forward_kwargs) <= 0:
        self.forward_kwargs = kwargs.copy()

    if len(self.adj_args) <= 0:
        self.adj_args = self.forward_args

    if len(self.adj_kwargs) <= 0:
        self.adj_kwargs = self.forward_kwargs.copy()

        if varform:
            if "J" in self.forward_kwargs:
                self.adj_kwargs["J"] = self.backend.adjoint(self.forward_kwargs["J"])
            if "Jp" in self.forward_kwargs:
                self.adj_kwargs["Jp"] = self.backend.adjoint(self.forward_kwargs["Jp"])

            if "M" in self.forward_kwargs:
                raise NotImplementedError("Annotation of adaptive solves not implemented.")
            self.adj_kwargs.pop("appctx", None)

    if "solver_parameters" in kwargs and "mat_type" in kwargs["solver_parameters"]:
        self.assemble_kwargs["mat_type"] = kwargs["solver_parameters"]["mat_type"]

    if varform:
        if "appctx" in kwargs:
            self.assemble_kwargs["appctx"] = kwargs["appctx"]


class GenericSolveBlock(blocks.GenericSolveBlock, Backend):
    pass


class SolveLinearSystemBlock(GenericSolveBlock):
    def __init__(self, A, u, b, *args, **kwargs):
        lhs = A.form
        func = u.function
        rhs = b.form
        bcs = A.bcs if hasattr(A, "bcs") else []
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

        # Set up parameters initialization
        self.ident_zeros_tol = A.ident_zeros_tol if hasattr(A, "ident_zeros_tol") else None
        self.assemble_system = A.assemble_system if hasattr(A, "assemble_system") else False

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=False)


class SolveVarFormBlock(GenericSolveBlock):
    def __init__(self, equation, func, bcs=[], *args, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)


class NonlinearVariationalSolveBlock(GenericSolveBlock):
    def __init__(self, equation, func, bcs, problem_J, solver_params, solver_kwargs, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs

        self.problem_J = problem_J
        self.solver_params = solver_params.copy()
        self.solver_kwargs = solver_kwargs

        super().__init__(lhs, rhs, func, bcs, **{**solver_kwargs, **kwargs})

        if self.problem_J is not None:
            for coeff in self.problem_J.coefficients():
                self.add_dependency(coeff, no_duplicates=True)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)

    def _forward_solve(self, lhs, rhs, func, bcs, **kwargs):
        self._ad_nlvs_replace_forms()
        self._ad_nlvs.parameters.update(self.solver_params)
        self._ad_nlvs.solve()
        func.assign(self._ad_nlvs._problem.u)
        return func

    def _ad_assign_map(self, form):
        count_map = self._ad_nlvs._problem._ad_count_map
        assign_map = {}
        form_ad_count_map = dict((count_map[coeff], coeff) for coeff in form.coefficients())
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if isinstance(coeff, (self.backend.Coefficient, self.backend.Constant)):
                coeff_count = coeff.count()
                if coeff_count in form_ad_count_map:
                    assign_map[form_ad_count_map[coeff_count]] = block_variable.saved_output
        return assign_map

    def _ad_assign_coefficients(self, form):
        assign_map = self._ad_assign_map(form)
        for coeff, value in assign_map.items():
            coeff.assign(value)

    def _ad_nlvs_replace_forms(self):
        problem = self._ad_nlvs._problem
        self._ad_assign_coefficients(problem.F)
        self._ad_assign_coefficients(problem.J)


class ProjectBlock(SolveVarFormBlock):
    def __init__(self, v, V, output, bcs=[], *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = V.mesh()
        dx = self.backend.dx(mesh)
        w = self.backend.TestFunction(V)
        Pv = self.backend.TrialFunction(V)
        a = self.backend.inner(Pv, w) * dx
        L = self.backend.inner(v, w) * dx

        super().__init__(a == L, output, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)


class MeshInputBlock(Block):
    """
    Block which links a MeshGeometry to its coordinates, which is a firedrake
    function.
    """
    def __init__(self, mesh):
        super().__init__()
        self.add_dependency(mesh)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        mesh = self.get_dependencies()[0].saved_output
        return mesh.coordinates


class FunctionSplitBlock(Block, Backend):
    def __init__(self, func, idx):
        super().__init__()
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        eval_adj = self.backend.Function(block_variable.output.function_space())
        eval_adj.sub(self.idx).assign(adj_inputs[0].function)
        return eval_adj.vector()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return self.backend.Function.sub(tlm_inputs[0], self.idx)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        eval_hessian = self.backend.Function(block_variable.output.function_space())
        eval_hessian.sub(self.idx).assign(hessian_inputs[0].function)
        return eval_hessian.vector()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.backend.Function.sub(inputs[0], self.idx)


class FunctionMergeBlock(Block, Backend):
    def __init__(self, func, idx):
        super().__init__()
        self.add_dependency(func)
        self.idx = idx
        for output in func._ad_outputs:
            self.add_dependency(output)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if idx == 0:
            return adj_inputs[0].split()[self.idx]
        else:
            return adj_inputs[0]

    def evaluate_tlm(self):
        tlm_input = self.get_dependencies()[0].tlm_value
        if tlm_input is None:
            return
        output = self.get_outputs()[0]
        fs = output.output.function_space()
        f = self.backend.Function(fs)
        output.add_tlm_output(self.backend.Function.assign(f.sub(self.idx), tlm_input))

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute(self):
        deps = self.get_dependencies()
        sub_func = deps[0].checkpoint
        parent_in = deps[1].checkpoint
        parent_out = self.get_outputs()[0].checkpoint
        parent_out.assign(parent_in)
        parent_out.sub(self.idx).assign(sub_func)


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh are changed.
    """
    def __init__(self, func, mesh):
        super().__init__()
        self.add_dependency(func)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        vector = self.get_dependencies()[0].saved_output
        mesh = vector.function_space().mesh()
        mesh.coordinates.assign(vector, annotate=False)
        return mesh._ad_create_checkpoint()


class InterpolateBlock(Block, Backend):
    r"""
    Annotates an interpolator.

    Consider the block as f with 1 forward model output ``v``, and inputs ``u`` and ``g``
    (there can, in principle, be any number of outputs).
    The adjoint input is ``vhat`` (``uhat`` and ``ghat`` are adjoints to ``u`` and ``v``
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

        f : W × G ⟶ V i.e. f(u, g) ∈ V ∀ u ∈ W and g ∈ G.
        f = I ∘ expr
        I :   X ⟶ V i.e. I(;x) ∈ V ∀ x ∈ X.
                          X is infinite dimensional.
        expr: W × G ⟶ X i.e. expr(u, g) ∈ X ∀ u ∈ W and g ∈ G.

    Arguments after a semicolon are linear (i.e. operation I is linear)
    """
    def __init__(self, interpolator, *functions, **kwargs):
        super().__init__()

        self.expr = interpolator.expr
        self.arguments, self.coefficients = extract_arguments_and_coefficients(self.expr)

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

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        r"""
        Denote ``d_u[A]`` as the gateaux derivative in the ``u`` direction.
        Arguments after a semicolon are linear.

        This calculates

        ::

            uhat = vhat ⋅ d_u[f](u, g; ⋅) (for inputs[idx] ∈ W)
            or
            ghat = vhat ⋅ d_g[f](u, g; ⋅) (for inputs[idx] ∈ G)

        where ``inputs[idx]`` specifies the derivative direction, ``vhat`` is
        ``adj_inputs[0]`` (since we assume only one block output)
        and ``⋅`` denotes an unspecified operand of ``u'`` (for
        ``inputs[idx]`` ∈ ``W``) or ``g'`` (for ``inputs[idx]`` ∈ ``G``) size (``vhat`` left
        multiplies the derivative).

        ::

            f = I ∘ expr : W × G ⟶ V
                           i.e. I(expr|_{u, g}) ∈ V ∀ u ∈ W, g ∈ G.

        Since ``I`` is linear we get that

        ::

            d_u[I ∘ expr](u, g; u') = I ∘ d_u[expr](u|_u, g|_g; u')
            d_g[I ∘ expr](u, g; g') = I ∘ d_u[expr](u|_u, g|_g; g').

        In tensor notation

        ::

            uhat_q^T = vhat_p^T I([dexpr/du|_u]_q)_p
            or
            ghat_q^T = vhat_p^T I([dexpr/dg|_u]_q)_p

        the output is then

        ::

            uhat_q = I^T([dexpr/du|_u]_q)_p vhat_p
            or
            ghat_q = I^T([dexpr/dg|_u]_q)_p vhat_p.
        """
        if len(adj_inputs) > 1:
            raise(NotImplementedError("Interpolate block must have a single output"))
        dJdm = self.backend.derivative(prepared, inputs[idx])
        return self.backend.Interpolator(dJdm, self.V).interpolate(adj_inputs[0], transpose=True).vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
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

            v'_l = I([dexpr/du|_{u,g}]_k u'_k)_l + I([dexpr/du|_{u,g}]_k g'_k)_l
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

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs, relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        r"""
        Denote ``d_u[A]`` as the gateaux derivative in the ``u`` direction.
        Arguments after a semicolon are linear.

        hessian_input is ``d_v[d_v[J]](v; v', ⋅)`` where the direction ``⋅`` is left
        unspecified so it can be operated upon.

        .. warning::
            NOTE: This comment describes the implementation of 1 block input ``u``.
            (e.g. interpolating from an expression with 1 coefficient).
            Explaining how this works for multiple block inputs (e.g. ``u`` and ``g``) is
            currently too complicated for the author to do succinctly!

        This function needs to output ``d_u[d_u[J ∘ f]](u; u', ⋅)`` where
        the direction ``⋅`` will be specified in another function and
        multiplied on the right with the output of this function.
        We will calculate this using the chain rule.

        ::

            J : V ⟶ R i.e. J(v) ∈ R ∀ v ∈ V
            f = I ∘ expr : W ⟶ V
            J ∘ f : W ⟶ R i.e. J(f|u) ∈ R ∀ u ∈ V.
            d_u[J ∘ f] : W × W ⟶ R i.e. d_u[J ∘ f](u; u')
            d_u[d_u[J ∘ f]] : W × W × W ⟶ R i.e. d_u[d_u[J ∘ f]](u; u', u'')
            d_v[J] : V × V ⟶ R i.e. d_v[J](v; v')
            d_v[d_v[J]] : V × V × V ⟶ R i.e. d_v[d_v[J]](v; v', v'')

        Chain rule:

        ::

            d_u[J ∘ f](u; u') = d_v[J](v = f|u; v' = d_u[f](u; u'))

        Multivariable chain rule:

        ::

            d_u[d_u[J ∘ f]](u; u', u'') =
            d_v[d_v[J]](v = f|u; v' = d_u[f](u; u'), v'' = d_u[f](u; u''))
            + d_v'[d_v[J]](v = f|u; v' = d_u[f](u; u'), v'' = d_u[d_u[f]](u; u', u''))
            = d_v[d_v[J]](v = f|u; v' = d_u[f](u; u'), v''=d_u[f](u; u''))
            + d_v[J](v = f|u; v' = v'' = d_u[d_u[f]](u; u', u''))

        since ``d_v[d_v[J]]`` is linear in ``v'`` so differentiating wrt to it leaves
        its coefficient, the bare d_v[J] operator which acts on the ``v''`` term
        that remains.

        The ``d_u[d_u[f]](u; u', u'')`` term can be simplified further:

        ::

            f = I ∘ expr : W ⟶ V i.e. I(expr|u) ∈ V ∀ u ∈ W
            d_u[I ∘ expr] : W × W ⟶ V i.e. d_u[I ∘ expr](u; u')
            d_u[d_u[I ∘ expr]] : W × W × W ⟶ V i.e. d_u[I ∘ expr](u; u', u'')
            d_x[I] : X × X ⟶ V i.e. d_x[I](x; x')
            d_x[d_x[I]] : X × X × X ⟶ V i.e. d_x[d_x[I]](x; x', x'')
            d_u[expr] : W × W ⟶ X i.e. d_u[expr](u; u')
            d_u[d_u[expr]] : W × W × W ⟶ X i.e. d_u[d_u[expr]](u; u', u'')

        Since ``I`` is linear we get that

        ::

            d_u[d_u[I ∘ expr]](u; u', u'') = I ∘ d_u[d_u[expr]](u; u', u'').

        So our full hessian is:

        ::

            d_u[d_u[J ∘ f]](u; u', u'')
            = d_v[d_v[J]](v = f|u; v' = d_u[f](u; u'), v''=d_u[f](u; u''))
            + d_v[J](v = f|u; v' = v'' = d_u[d_u[f]](u; u', u''))

        In tensor notation

        ::

            [d^2[J ∘ f]/du^2|_u]_{lk} u'_k u''_k =
            [d^2J/dv^2|_{v=f|_u}]_{ij} [df/du|_u]_{jk} u'_k [df/du|_u]_{il} u''_l
            + [dJ/dv|_{v=f_u}]_i I([d^2expr/du^2|_u]_{lk} u'_k)_i u''_l

        In the first term:

        ::

            [df/du|_u]_{jk} u'_k = v'_j
            => [d^2J/dv^2|_{v=f|_u}]_{ij} [df/du|_u]_{jk} u'_k
            = [d^2J/dv^2|_{v=f|_u}]_{ij} v'_j
            = hessian_input_i
            => [d^2J/dv^2|_{v=f|_u}]_{ij} [df/du|_u]_{jk} u'_k [df/du|_u]_{il}
            = hessian_input_i [df/du|_u]_{il}
            = self.evaluate_adj_component(inputs, hessian_inputs, ...)_l

        In the second term we calculate everything explicitly though note
        ``[dJ/dv|_{v=f_u}]_i = adj_inputs[0]_i``

        Also, the second term is 0 if ``expr`` is linear.
        """

        if len(hessian_inputs) > 1 or len(adj_inputs) > 1:
            raise(NotImplementedError("Interpolate block must have a single output"))

        component = self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

        # Prepare again by replacing expression
        expr = replace(self.expr, self._replace_map())

        # Calculate first derivative for each relevant block
        dexprdu = 0.
        for _, bv in relevant_dependencies:
            # Only take the derivative if there is a direction to take it in
            if bv.tlm_value is None:
                continue
            dexprdu += self.backend.derivative(expr, bv.saved_output, bv.tlm_value)

        # Calculate the second derivative w.r.t. the specified coefficient's
        # saved value. Leave argument unspecified so it can be calculated with
        # the eventual inner product with u''.
        d2exprdudu = self.backend.derivative(dexprdu, block_variable.saved_output)

        # left multiply by dJ/dv (adj_inputs[0]) - i.e. interpolate using the
        # transpose operator
        component += self.backend.Interpolator(d2exprdudu, self.V).interpolate(adj_inputs[0], transpose=True)
        return component.vector()

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return replace(self.expr, self._replace_map())

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return self.backend.interpolate(prepared, self.V)

from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks
from pyadjoint.block import Block
from pyadjoint import stop_annotating
from ufl.algorithms.analysis import extract_arguments_and_coefficients
from ufl import replace
from .checkpointing import maybe_disk_checkpoint

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
    def recompute_component(self, inputs, block_variable, idx, prepared):
        result = super().recompute_component(inputs, block_variable, idx, prepared)
        return maybe_disk_checkpoint(result)


class AssembleBlock(blocks.AssembleBlock, Backend):
    def recompute_component(self, inputs, block_variable, idx, prepared):
        result = super().recompute_component(inputs, block_variable, idx, prepared)
        if isinstance(result, firedrake.Function):
            return maybe_disk_checkpoint(result)
        else:
            return result


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
    def recompute_component(self, inputs, block_variable, idx, prepared):
        result = super().recompute_component(inputs, block_variable, idx, prepared)
        return maybe_disk_checkpoint(result)


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
    def __init__(self, equation, func, bcs, adj_F, dFdm_cache, problem_J, solver_params, solver_kwargs, **kwargs):
        lhs = equation.lhs
        rhs = equation.rhs

        self.adj_F = adj_F
        self._dFdm_cache = dFdm_cache
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

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        dJdu = adj_inputs[0]

        F_form = self._create_F_form()

        dFdu_form = self.adj_F
        dJdu = dJdu.copy()

        # Replace the form coefficients with checkpointed values.
        replace_map = self._replace_map(dFdu_form)
        replace_map[self.func] = self.get_outputs()[0].saved_output
        dFdu_form = replace(dFdu_form, replace_map)

        compute_bdy = self._should_compute_boundary_adjoint(relevant_dependencies)
        adj_sol, adj_sol_bdy = self._assemble_and_solve_adj_eq(dFdu_form, dJdu, compute_bdy)
        self.adj_sol = adj_sol
        if self.adj_cb is not None:
            self.adj_cb(adj_sol)
        if self.adj_bdy_cb is not None and compute_bdy:
            self.adj_bdy_cb(adj_sol_bdy)

        r = {}
        r["form"] = F_form
        r["adj_sol"] = adj_sol
        r["adj_sol_bdy"] = adj_sol_bdy
        return r

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if not self.linear and self.func == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            return None
        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        adj_sol_bdy = prepared["adj_sol_bdy"]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, firedrake.Function):
            trial_function = firedrake.TrialFunction(c.function_space())
        elif isinstance(c, firedrake.Constant):
            mesh = self.compat.extract_mesh_from_form(F_form)
            trial_function = firedrake.TrialFunction(c._ad_function_space(mesh))
        elif isinstance(c, firedrake.DirichletBC):
            tmp_bc = self.compat.create_bc(c, value=self.compat.extract_subfunction(adj_sol_bdy, c.function_space()))
            return [tmp_bc]
        elif isinstance(c, self.compat.MeshType):
            # Using CoordianteDerivative requires us to do action before
            # differentiating, might change in the future.
            F_form_tmp = firedrake.action(F_form, adj_sol)
            X = firedrake.SpatialCoordinate(c_rep)
            dFdm = firedrake.derivative(-F_form_tmp, X, firedrake.TestFunction(c._ad_function_space()))

            dFdm = self.compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)
            return dFdm

        # dFdm_cache works with original variables, not block saved outputs.
        if c in self._dFdm_cache:
            dFdm = self._dFdm_cache[c]
        else:
            dFdm = -firedrake.derivative(self.lhs, c, trial_function)
            dFdm = firedrake.adjoint(dFdm)
            self._dFdm_cache[c] = dFdm

        # Replace the form coefficients with checkpointed values.
        replace_map = self._replace_map(dFdm)
        replace_map[self.func] = self.get_outputs()[0].saved_output
        dFdm = replace(dFdm, replace_map)

        dFdm = dFdm * adj_sol
        dFdm = self.compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)

        return dFdm


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
    def __init__(self, mesh, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
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
        return maybe_disk_checkpoint(mesh.coordinates)


class SubfunctionBlock(Block, Backend):
    def __init__(self, func, idx, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)
        self.idx = idx

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        eval_adj = self.backend.Function(block_variable.output.function_space())
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
        eval_hessian = self.backend.Function(block_variable.output.function_space())
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
        output.add_tlm_output(self.backend.Function.assign(f.sub(self.idx), tlm_input))

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


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh are changed.
    """
    def __init__(self, func, mesh, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
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
        super().__init__(ad_block_tag=kwargs.pop("ad_block_tag", None))

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
            raise NotImplementedError("Interpolate block must have a single output")
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
            raise NotImplementedError("Interpolate block must have a single output")

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
        result = self.backend.interpolate(prepared, self.V)
        if isinstance(result, firedrake.Function):
            return maybe_disk_checkpoint(result)
        else:
            return result

    def __str__(self):
        target_string = f"〈{str(self.V.ufl_element().shortstr())}〉"
        return f"interpolate({self.expr},  {target_string})"


class SupermeshProjectBlock(Block, Backend):
    r"""
    Annotates supermesh projection.

    Suppose we have a source space, :math:`V_A`, and a target space, :math:`V_B`.
    Projecting a source from :math:`V_A` to :math:`V_B` amounts to solving the
    linear system

    .. math::
        M_B * v_B = M_{AB} * v_A,

    where
      * :math:`M_B` is the mass matrix on :math:`V_B`,
      * :math:`M_{AB}` is the mixed mass matrix for :math:`V_A`
        and :math:`V_B`,
      * :math:`v_A` and :math:`v_B` are vector representations of
        the source and target :class:`.Function` s.

    This can be broken into two steps:
      Step 1. form RHS, multiplying the source with the mixed mass matrix;

      Step 2. solve linear system.
    """
    def __init__(self, source, target_space, target, bcs=[], **kwargs):
        super(SupermeshProjectBlock, self).__init__(ad_block_tag=kwargs.pop("ad_block_tag", None))
        import firedrake.supermeshing as supermesh

        # Process args and kwargs
        if not isinstance(source, self.backend.Function):
            raise NotImplementedError(f"Source function must be a Function, not {type(source)}.")
        if bcs != []:
            raise NotImplementedError("Boundary conditions not yet considered.")

        # Store spaces
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = target_space.mesh()
        self.source_space = source.function_space()
        self.target_space = target_space
        self.projector = firedrake.Projector(source, target_space, **kwargs)

        # Assemble mixed mass matrix
        with stop_annotating():
            self.mixed_mass = supermesh.assemble_mixed_mass_matrix(
                source.function_space(), target_space
            )

        # Add dependencies
        self.add_dependency(source, no_duplicates=True)
        for bc in bcs:
            self.add_dependency(bc, no_duplicates=True)

    def apply_mixedmass(self, a):
        b = self.backend.Function(self.target_space)
        with a.dat.vec_ro as vsrc, b.dat.vec_wo as vrhs:
            self.mixed_mass.mult(vsrc, vrhs)
        return b

    def recompute_component(self, inputs, block_variable, idx, prepared):
        if not isinstance(inputs[0], self.backend.Function):
            raise NotImplementedError(f"Source function must be a Function, not {type(inputs[0])}.")
        target = self.backend.Function(self.target_space)
        rhs = self.apply_mixedmass(inputs[0])      # Step 1
        self.projector.apply_massinv(target, rhs)  # Step 2
        return maybe_disk_checkpoint(target)

    def _recompute_component_transpose(self, inputs):
        if not isinstance(inputs[0], (self.backend.Function, self.backend.Vector)):
            raise NotImplementedError(f"Source function must be a Function, not {type(inputs[0])}.")
        out = self.backend.Function(self.source_space)
        tmp = self.backend.Function(self.target_space)
        self.projector.apply_massinv(tmp, inputs[0])   # Adjoint of step 2 (since mass self-adjoint)
        with tmp.dat.vec_ro as vtmp, out.dat.vec_wo as vout:
            self.mixed_mass.multTranspose(vtmp, vout)  # Adjoint of step 1
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        """
        Recall that the forward propagation can be broken down as
          Step 1. multiply :math:`w := M_{AB} * v_A`;
          Step 2. solve :math:`M_B * v_B = w`.

        For a seed vector :math:`v_B^{seed}` from the target space, the adjoint is given by
          Adjoint of step 2. solve :math:`M_B^T * w = v_B^{seed}` for `w`;
          Adjoint of step 1. multiply :math:`v_A^{adj} := M_{AB}^T * w`.
        """
        if len(adj_inputs) != 1:
            raise NotImplementedError("SupermeshProjectBlock must have a single output")
        return self._recompute_component_transpose(adj_inputs).vector()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        """
        Given that the input is a `Function`, we just have a linear operation. As such,
        the tlm is just the sum of each tlm input projected into the target space.
        """
        dJdm = self.backend.Function(self.target_space)
        for tlm_input in tlm_inputs:
            if tlm_input is None:
                continue
            dJdm += self.recompute_component([tlm_input], block_variable, idx, prepared)
        return dJdm

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        if len(hessian_inputs) != 1:
            raise NotImplementedError("SupermeshProjectBlock must have a single output")
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx).vector()

    def __str__(self):
        target_string = f"〈{str(self.target_space.ufl_element().shortstr())}〉"
        return f"project({self.get_dependencies()[0]}, {target_string}))"

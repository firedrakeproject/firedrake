import numpy
import ufl
from ufl import replace
from ufl.formatting.ufl2unicode import ufl2unicode

from pyadjoint import Block, stop_annotating
from pyadjoint.enlisting import Enlist
import firedrake
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint
from .block_utils import isconstant


def extract_subfunction(u, V):
    """If V is a subspace of the function-space of u, return the component of u that is in that subspace."""
    if V.index is not None:
        # V is an indexed subspace of a MixedFunctionSpace
        return u.sub(V.index)
    elif V.component is not None:
        # V is a vector component subspace.
        # The vector functionspace V.parent may itself be a subspace
        # so call this function recursively
        return extract_subfunction(u, V.parent).sub(V.component)
    else:
        return u


class GenericSolveBlock(Block):
    pop_kwargs_keys = ["adj_cb", "adj_bdy_cb", "adj2_cb", "adj2_bdy_cb",
                       "forward_args", "forward_kwargs", "adj_args",
                       "adj_kwargs"]

    def __init__(self, lhs, rhs, func, bcs, *args, **kwargs):
        super().__init__(ad_block_tag=kwargs.pop('ad_block_tag', None))
        self.adj_cb = kwargs.pop("adj_cb", None)
        self.adj_bdy_cb = kwargs.pop("adj_bdy_cb", None)
        self.adj2_cb = kwargs.pop("adj2_cb", None)
        self.adj2_bdy_cb = kwargs.pop("adj2_bdy_cb", None)
        self.adj_sol = None

        self.forward_args = []
        self.forward_kwargs = {}
        self.adj_args = []
        self.adj_kwargs = {}
        self.assemble_kwargs = {}

        # Equation LHS
        self.lhs = lhs
        # Equation RHS
        self.rhs = rhs
        # Solution function
        self.func = func
        self.function_space = self.func.function_space()
        # Boundary conditions
        self.bcs = []
        if bcs is not None:
            self.bcs = Enlist(bcs)

        if isinstance(self.lhs, ufl.Form) and isinstance(self.rhs, (ufl.Form, ufl.Cofunction)):
            self.linear = True
            for c in self.rhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
        else:
            self.linear = False

        for c in self.lhs.coefficients():
            self.add_dependency(c, no_duplicates=True)

        for bc in self.bcs:
            self.add_dependency(bc, no_duplicates=True)

        mesh = self.lhs.ufl_domain()
        self.add_dependency(mesh)
        self._init_solver_parameters(args, kwargs)

    def _init_solver_parameters(self, args, kwargs):
        self.forward_args = kwargs.pop("forward_args", [])
        self.forward_kwargs = kwargs.pop("forward_kwargs", {})
        self.adj_args = kwargs.pop("adj_args", [])
        self.adj_kwargs = kwargs.pop("adj_kwargs", {})
        self.assemble_kwargs = {}

    def __str__(self):
        return "solve({} = {})".format(ufl2unicode(self.lhs),
                                       ufl2unicode(self.rhs))

    def _create_F_form(self):
        # Process the equation forms, replacing values with checkpoints,
        # and gathering lhs and rhs in one single form.
        if self.linear:
            tmp_u = firedrake.Function(self.function_space)
            F_form = firedrake.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replace_map = self._replace_map(F_form)
        replace_map[tmp_u] = self.get_outputs()[0].saved_output
        return ufl.replace(F_form, replace_map)

    def _homogenize_bcs(self):
        bcs = []
        for bc in self.bcs:
            if isinstance(bc, firedrake.DirichletBC):
                bc = bc.reconstruct(g=0)
            bcs.append(bc)
        return bcs

    def _create_initial_guess(self):
        return firedrake.Function(self.function_space)

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, firedrake.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def _replace_map(self, form):
        replace_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in form.coefficients():
                replace_coeffs[coeff] = block_variable.saved_output
        return replace_coeffs

    def _replace_form(self, form, func=None):
        """Replace the form coefficients with checkpointed values

        func represents the initial guess if relevant.
        """
        replace_map = self._replace_map(form)
        if func is not None and self.func in replace_map:
            firedrake.Function.assign(func, replace_map[self.func])
            replace_map[self.func] = func
        return ufl.replace(form, replace_map)

    def _should_compute_boundary_adjoint(self, relevant_dependencies):
        # Check if DirichletBC derivative is relevant
        bdy = False
        for _, dep in relevant_dependencies:
            if isinstance(dep.output, firedrake.DirichletBC):
                bdy = True
                break
        return bdy

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output

        dJdu = adj_inputs[0]

        F_form = self._create_F_form()

        dFdu = firedrake.derivative(
            F_form,
            fwd_block_variable.saved_output,
            firedrake.TrialFunction(
                u.function_space()
            )
        )
        dFdu_form = firedrake.adjoint(dFdu)
        dJdu = dJdu.copy()

        compute_bdy = self._should_compute_boundary_adjoint(
            relevant_dependencies
        )
        adj_sol, adj_sol_bdy = self._assemble_and_solve_adj_eq(
            dFdu_form, dJdu, compute_bdy
        )
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

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()
        kwargs = self.assemble_kwargs.copy()
        # Homogenize and apply boundary conditions on adj_dFdu and dJdu.
        bcs = self._homogenize_bcs()
        kwargs["bcs"] = bcs
        dFdu = firedrake.assemble(dFdu_adj_form, **kwargs)

        for bc in bcs:
            bc.apply(dJdu)

        adj_sol = firedrake.Function(self.function_space)
        firedrake.solve(
            dFdu, adj_sol, dJdu, *self.adj_args, **self.adj_kwargs
        )

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = firedrake.Function(
                self.function_space.dual(),
                dJdu_copy.dat - firedrake.assemble(
                    firedrake.action(dFdu_adj_form, adj_sol)
                ).dat
            )

        return adj_sol, adj_sol_bdy

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if not self.linear and self.func == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            return None
        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        adj_sol_bdy = prepared["adj_sol_bdy"]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isconstant(c):
            mesh = F_form.ufl_domain()
            trial_function = firedrake.TrialFunction(
                c._ad_function_space(mesh)
            )
        elif isinstance(c, (firedrake.Function, firedrake.Cofunction)):
            trial_function = firedrake.TrialFunction(c.function_space())
        elif isinstance(c, firedrake.DirichletBC):
            tmp_bc = c.reconstruct(
                g=extract_subfunction(adj_sol_bdy, c.function_space())
            )
            return [tmp_bc]
        elif isinstance(c, firedrake.MeshGeometry):
            # Using CoordinateDerivative requires us to do action before
            # differentiating, might change in the future.
            F_form_tmp = firedrake.action(F_form, adj_sol)
            X = firedrake.SpatialCoordinate(c_rep)
            dFdm = firedrake.derivative(
                -F_form_tmp, X,
                firedrake.TestFunction(c._ad_function_space())
            )

            if dFdm == 0:
                return firedrake.Function(c._ad_function_space().dual())

            dFdm = firedrake.assemble(dFdm, **self.assemble_kwargs)
            return dFdm

        dFdm = -firedrake.derivative(F_form, c_rep, trial_function)
        dFdm = firedrake.adjoint(dFdm)
        dFdm = dFdm * adj_sol
        dFdm = firedrake.assemble(dFdm, **self.assemble_kwargs)
        return dFdm

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output

        F_form = self._create_F_form()

        # Obtain dFdu.
        dFdu = firedrake.derivative(
            F_form,
            fwd_block_variable.saved_output,
            firedrake.TrialFunction(u.function_space())
        )

        return {
            "form": F_form,
            "dFdu": dFdu
        }

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        F_form = prepared["form"]
        dFdu = prepared["dFdu"]
        V = self.get_outputs()[idx].output.function_space()

        bcs = []
        dFdm = 0.
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, firedrake.DirichletBC):
                if tlm_value is None:
                    bcs.append(c.reconstruct(g=0))
                else:
                    bcs.append(tlm_value)
                continue
            elif isinstance(c, firedrake.MeshGeometry):
                X = firedrake.SpatialCoordinate(c)
                c_rep = X

            if tlm_value is None:
                continue

            if c == self.func and not self.linear:
                continue

            dFdm += firedrake.derivative(-F_form, c_rep, tlm_value)

        if isinstance(dFdm, float):
            v = dFdu.arguments()[0]
            dFdm = firedrake.inner(
                firedrake.Constant(numpy.zeros(v.ufl_shape)), v
            ) * firedrake.dx

        dFdm = ufl.algorithms.expand_derivatives(dFdm)
        dFdm = firedrake.assemble(dFdm)
        dudm = firedrake.Function(V)
        return self._assemble_and_solve_tlm_eq(
            firedrake.assemble(dFdu, bcs=bcs, **self.assemble_kwargs),
            dFdm, dudm, bcs
        )

    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        return self._assembled_solve(dFdu, dFdm, dudm, bcs)

    def _assemble_soa_eq_rhs(self, dFdu_form, adj_sol, hessian_input, d2Fdu2):
        # Start piecing together the rhs of the soa equation
        b = hessian_input.copy()
        if len(d2Fdu2.integrals()) > 0:
            b_form = firedrake.action(firedrake.adjoint(d2Fdu2), adj_sol)
        else:
            b_form = d2Fdu2

        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value

            if (c == self.func and not self.linear) or tlm_input is None:
                continue

            if isinstance(c, firedrake.MeshGeometry):
                X = firedrake.SpatialCoordinate(c)
                dFdu_adj = firedrake.action(firedrake.adjoint(dFdu_form),
                                            adj_sol)
                d2Fdudm = ufl.algorithms.expand_derivatives(
                    firedrake.derivative(dFdu_adj, X, tlm_input))
                if len(d2Fdudm.integrals()) > 0:
                    b_form += d2Fdudm
            elif not isinstance(c, firedrake.DirichletBC):
                dFdu_adj = firedrake.action(firedrake.adjoint(dFdu_form),
                                            adj_sol)
                b_form += firedrake.derivative(dFdu_adj, c_rep, tlm_input)

        b_form = ufl.algorithms.expand_derivatives(b_form)
        if len(b_form.integrals()) > 0:
            b -= firedrake.assemble(b_form)

        return b

    def _assemble_and_solve_soa_eq(self, dFdu_form, adj_sol, hessian_input,
                                   d2Fdu2, compute_bdy):
        b = self._assemble_soa_eq_rhs(dFdu_form, adj_sol, hessian_input,
                                      d2Fdu2)
        dFdu_form = firedrake.adjoint(dFdu_form)
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_adj_eq(dFdu_form, b,
                                                                 compute_bdy)
        if self.adj2_cb is not None:
            self.adj2_cb(adj_sol2)
        if self.adj2_bdy_cb is not None and compute_bdy:
            self.adj2_bdy_cb(adj_sol2_bdy)
        return adj_sol2, adj_sol2_bdy

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        # First fetch all relevant values
        fwd_block_variable = self.get_outputs()[0]
        hessian_input = hessian_inputs[0]
        tlm_output = fwd_block_variable.tlm_value

        if hessian_input is None:
            return

        if tlm_output is None:
            return

        F_form = self._create_F_form()

        # Using the equation Form derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = firedrake.derivative(F_form,
                                         fwd_block_variable.saved_output)
        d2Fdu2 = ufl.algorithms.expand_derivatives(
            firedrake.derivative(dFdu_form, fwd_block_variable.saved_output,
                                 tlm_output))

        adj_sol = self.adj_sol
        if adj_sol is None:
            raise RuntimeError("Hessian computation was run before adjoint.")
        bdy = self._should_compute_boundary_adjoint(relevant_dependencies)
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_soa_eq(
            dFdu_form, adj_sol, hessian_input, d2Fdu2, bdy
        )

        r = {}
        r["adj_sol2"] = adj_sol2
        r["adj_sol2_bdy"] = adj_sol2_bdy
        r["form"] = F_form
        r["adj_sol"] = adj_sol
        return r

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        c = block_variable.output
        if c == self.func and not self.linear:
            return None

        adj_sol2 = prepared["adj_sol2"]
        adj_sol2_bdy = prepared["adj_sol2_bdy"]
        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        fwd_block_variable = self.get_outputs()[0]
        tlm_output = fwd_block_variable.tlm_value

        c_rep = block_variable.saved_output

        # If m = DirichletBC then d^2F(u,m)/dm^2 = 0 and d^2F(u,m)/dudm = 0,
        # so we only have the term dF(u,m)/dm * adj_sol2
        if isinstance(c, firedrake.DirichletBC):
            tmp_bc = c.reconstruct(
                g=extract_subfunction(adj_sol2_bdy, c.function_space())
            )
            return [tmp_bc]

        if isconstant(c_rep):
            mesh = F_form.ufl_domain()
            W = c._ad_function_space(mesh)
        elif isinstance(c, firedrake.MeshGeometry):
            X = firedrake.SpatialCoordinate(c)
            W = c._ad_function_space()
        else:
            W = c.function_space()

        dc = firedrake.TestFunction(W)
        form_adj = firedrake.action(F_form, adj_sol)
        form_adj2 = firedrake.action(F_form, adj_sol2)
        if isinstance(c, firedrake.MeshGeometry):
            dFdm_adj = firedrake.derivative(form_adj, X, dc)
            dFdm_adj2 = firedrake.derivative(form_adj2, X, dc)
        else:
            dFdm_adj = firedrake.derivative(form_adj, c_rep, dc)
            dFdm_adj2 = firedrake.derivative(form_adj2, c_rep, dc)

        # TODO: Old comment claims this might break on split. Confirm if true
        # or not.
        d2Fdudm = ufl.algorithms.expand_derivatives(
            firedrake.derivative(dFdm_adj, fwd_block_variable.saved_output,
                                 tlm_output))

        d2Fdm2 = 0
        # We need to add terms from every other dependency
        # i.e. the terms d^2F/dm_1dm_2
        for _, bv in relevant_dependencies:
            c2 = bv.output
            c2_rep = bv.saved_output

            if isinstance(c2, firedrake.DirichletBC):
                continue

            tlm_input = bv.tlm_value
            if tlm_input is None:
                continue

            if c2 == self.func and not self.linear:
                continue

            # TODO: If tlm_input is a Sum, this crashes in some instances?
            if isinstance(c2_rep, firedrake.MeshGeometry):
                X = firedrake.SpatialCoordinate(c2_rep)
                d2Fdm2 += ufl.algorithms.expand_derivatives(
                    firedrake.derivative(dFdm_adj, X, tlm_input)
                )
            else:
                d2Fdm2 += ufl.algorithms.expand_derivatives(
                    firedrake.derivative(dFdm_adj, c2_rep, tlm_input)
                )

        hessian_form = ufl.algorithms.expand_derivatives(
            d2Fdm2 + dFdm_adj2 + d2Fdudm
        )
        hessian_output = 0
        if not hessian_form.empty():
            hessian_output = firedrake.assemble(hessian_form)
            hessian_output *= -1.

        return hessian_output

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self._replace_recompute_form()

    def _replace_recompute_form(self):
        func = self._create_initial_guess()

        bcs = self._recover_bcs()
        lhs = self._replace_form(self.lhs, func=func)
        rhs = 0
        if self.linear:
            rhs = self._replace_form(self.rhs)

        return lhs, rhs, func, bcs

    def _forward_solve(self, lhs, rhs, func, bcs):
        firedrake.solve(lhs == rhs, func, bcs, *self.forward_args,
                        **self.forward_kwargs)
        return func

    def _assembled_solve(self, lhs, rhs, func, bcs, **kwargs):
        rhs_func = rhs.riesz_representation(riesz_map="l2")
        for bc in bcs:
            bc.apply(rhs_func)
        rhs.assign(rhs_func.riesz_representation(riesz_map="l2"))
        firedrake.solve(lhs, func, rhs, **kwargs)
        return func

    def recompute_component(self, inputs, block_variable, idx, prepared):
        lhs = prepared[0]
        rhs = prepared[1]
        func = prepared[2]
        bcs = prepared[3]
        result = self._forward_solve(lhs, rhs, func, bcs)
        return maybe_disk_checkpoint(result)


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
                self.adj_kwargs["J"] = firedrake.adjoint(
                    self.forward_kwargs["J"]
                )
            if "Jp" in self.forward_kwargs:
                self.adj_kwargs["Jp"] = firedrake.adjoint(
                    self.forward_kwargs["Jp"]
                )

            if "M" in self.forward_kwargs:
                raise NotImplementedError(
                    "Annotation of adaptive solves not implemented."
                )
            self.adj_kwargs.pop("appctx", None)

    if "mat_type" in kwargs.get("solver_parameters", {}):
        self.assemble_kwargs["mat_type"] = \
            kwargs["solver_parameters"]["mat_type"]

    if varform:
        if "appctx" in kwargs:
            self.assemble_kwargs["appctx"] = kwargs["appctx"]


class SolveLinearSystemBlock(GenericSolveBlock):
    def __init__(self, A, u, b, *args, **kwargs):
        lhs = A.form
        func = u.function
        rhs = b.form
        bcs = A.bcs if hasattr(A, "bcs") else []
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)

        # Set up parameters initialization
        self.ident_zeros_tol = \
            A.ident_zeros_tol if hasattr(A, "ident_zeros_tol") else None
        self.assemble_system = \
            A.assemble_system if hasattr(A, "assemble_system") else False

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
    def __init__(self, equation, func, bcs, adj_F, dFdm_cache, problem_J,
                 solver_params, solver_kwargs, **kwargs):
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
        form_ad_count_map = dict((count_map[coeff], coeff)
                                 for coeff in form.coefficients())
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if isinstance(coeff,
                          (firedrake.Coefficient, firedrake.Constant,
                           firedrake.Cofunction)):
                coeff_count = coeff.count()
                if coeff_count in form_ad_count_map:
                    assign_map[form_ad_count_map[coeff_count]] = \
                        block_variable.saved_output
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

        compute_bdy = self._should_compute_boundary_adjoint(
            relevant_dependencies
        )
        adj_sol, adj_sol_bdy = self._assemble_and_solve_adj_eq(
            dFdu_form, dJdu, compute_bdy
        )
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

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
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
            mesh = F_form.ufl_domain()
            trial_function = firedrake.TrialFunction(
                c._ad_function_space(mesh)
            )
        elif isinstance(c, firedrake.DirichletBC):
            tmp_bc = c.reconstruct(
                g=extract_subfunction(adj_sol_bdy, c.function_space())
            )
            return [tmp_bc]
        elif isinstance(c, firedrake.MeshGeometry):
            # Using CoordianteDerivative requires us to do action before
            # differentiating, might change in the future.
            F_form_tmp = firedrake.action(F_form, adj_sol)
            X = firedrake.SpatialCoordinate(c_rep)
            dFdm = firedrake.derivative(-F_form_tmp, X, firedrake.TestFunction(
                c._ad_function_space())
            )

            dFdm = firedrake.assemble(dFdm, **self.assemble_kwargs)
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
        dFdm = firedrake.assemble(dFdm, **self.assemble_kwargs)

        return dFdm


class ProjectBlock(SolveVarFormBlock):
    def __init__(self, v, V, output, bcs=[], *args, **kwargs):
        mesh = kwargs.pop("mesh", None)
        if mesh is None:
            mesh = V.mesh()
        dx = firedrake.dx(mesh)
        w = firedrake.TestFunction(V)
        Pv = firedrake.TrialFunction(V)
        a = firedrake.inner(Pv, w) * dx
        L = firedrake.inner(v, w) * dx

        super().__init__(a == L, output, bcs, *args, **kwargs)

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)


class SupermeshProjectBlock(Block):
    r"""
    Annotates supermesh projection.

    Suppose we have a source space, :math:`V_A`, and a target space,
    :math:`V_B`. Projecting a source from :math:`V_A` to :math:`V_B` amounts to
    solving the linear system

    .. math::
        M_B * v_B = M_{AB} * v_A,

    where
      * :math:`M_B` is the mass matrix on :math:`V_B`,
      * :math:`M_{AB}` is the mixed mass matrix for :math:`V_A` and
        :math:`V_B`,
      * :math:`v_A` and :math:`v_B` are vector representations of the source
        and target :class:`.Function` s.

    This can be broken into two steps:
      Step 1. form RHS, multiplying the source with the mixed mass matrix;

      Step 2. solve linear system.
    """
    def __init__(self, source, target_space, target, bcs=[], **kwargs):
        super(SupermeshProjectBlock, self).__init__(
            ad_block_tag=kwargs.pop("ad_block_tag", None)
        )
        import firedrake.supermeshing as supermesh

        # Process args and kwargs
        if not isinstance(source, firedrake.Function):
            raise NotImplementedError(
                f"Source function must be a Function, not {type(source)}."
            )
        if bcs != []:
            raise NotImplementedError(
                "Boundary conditions not yet considered."
            )

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
        b = firedrake.Function(self.target_space)
        with a.dat.vec_ro as vsrc, b.dat.vec_wo as vrhs:
            self.mixed_mass.mult(vsrc, vrhs)
        return b

    def recompute_component(self, inputs, block_variable, idx, prepared):
        if not isinstance(inputs[0], firedrake.Function):
            raise NotImplementedError(
                f"Source function must be a Function, not {type(inputs[0])}."
            )
        target = firedrake.Function(self.target_space)
        rhs = self.apply_mixedmass(inputs[0])      # Step 1
        self.projector.apply_massinv(target, rhs)  # Step 2
        return maybe_disk_checkpoint(target)

    def _recompute_component_transpose(self, inputs):
        if not isinstance(inputs[0], firedrake.Cofunction):
            raise NotImplementedError(
                f"Source function must be a Cofunction, not {type(inputs[0])}."
            )
        out = firedrake.Cofunction(self.source_space.dual())
        tmp = firedrake.Function(self.target_space)
        # Adjoint of step 2 (mass is self-adjoint)
        self.projector.apply_massinv(tmp, inputs[0])
        with tmp.dat.vec_ro as vtmp, out.dat.vec_wo as vout:
            self.mixed_mass.multTranspose(vtmp, vout)  # Adjoint of step 1
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        """
        Evaluate the adjoint to one output of the block

        Recall that the forward propagation can be broken down as:
          Step 1. multiply :math:`w := M_{AB} * v_A`;

          Step 2. solve :math:`M_B * v_B = w`.

        For a seed vector :math:`v_B^{seed}` from the target space, the adjoint
        is given by:

          Adjoint of step 2. solve :math:`M_B^T * w = v_B^{seed}` for `w`;

          Adjoint of step 1. multiply :math:`v_A^{adj} := M_{AB}^T * w`.
        """
        if len(adj_inputs) != 1:
            raise NotImplementedError(
                "SupermeshProjectBlock must have a single output"
            )
        return self._recompute_component_transpose(adj_inputs)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        """
        Given that the input is a `Function`, we just have a linear operation.
        As such, the tlm is just the sum of each tlm input projected into the
        target space.
        """
        dJdm = firedrake.Function(self.target_space)
        for tlm_input in tlm_inputs:
            if tlm_input is None:
                continue
            dJdm += self.recompute_component([tlm_input], block_variable, idx,
                                             prepared)
        return dJdm

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx,
                                   relevant_dependencies, prepared=None):
        if len(hessian_inputs) != 1:
            raise NotImplementedError(
                "SupermeshProjectBlock must have a single output"
            )
        return self.evaluate_adj_component(inputs, hessian_inputs,
                                           block_variable, idx)

    def __str__(self):
        target_string = f"〈{str(self.target_space.ufl_element().shortstr())}〉"
        return f"project({self.get_dependencies()[0]}, {target_string}))"

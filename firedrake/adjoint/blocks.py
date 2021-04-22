from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks
from pyadjoint.block import Block
from pyadjoint.enlisting import Enlist
from ufl.algorithms.analysis import extract_arguments_and_coefficients
from ufl import replace

import firedrake
import firedrake.utils as utils

import numpy
import ufl

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


class GenericSolveBlock(Block, Backend):
    pop_kwargs_keys = ["adj_cb", "adj_bdy_cb", "adj2_cb", "adj2_bdy_cb",
                       "forward_args", "forward_kwargs", "adj_args", "adj_kwargs"]

    def __init__(self, lhs, rhs, func, bcs, *args, **kwargs):
        super().__init__()
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

        if isinstance(self.lhs, ufl.Form) and isinstance(self.rhs, ufl.Form):
            self.linear = True
            for c in self.rhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
        else:
            self.linear = False

        for c in self.lhs.coefficients():
            self.add_dependency(c, no_duplicates=True)

        for bc in self.bcs:
            self.add_dependency(bc, no_duplicates=True)

        if self.backend.__name__ != "firedrake":
            mesh = self.lhs.ufl_domain().ufl_cargo()
        else:
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
        return "{} = {}".format(str(self.lhs), str(self.rhs))

    def _create_F_form(self):
        # Process the equation forms, replacing values with checkpoints,
        # and gathering lhs and rhs in one single form.
        if self.linear:
            tmp_u = self.compat.create_function(self.function_space)
            F_form = self.backend.action(self.lhs, tmp_u) - self.rhs
        else:
            tmp_u = self.func
            F_form = self.lhs

        replace_map = self._replace_map(F_form)
        replace_map[tmp_u] = self.get_outputs()[0].saved_output
        return ufl.replace(F_form, replace_map)

    def _process_bcs(self):
        bcs_list = []
        for bc in self.bcs:
            if isinstance(bc, self.backend.DirichletBC):
                bc = self.compat.create_bc(bc, homogenize=True)
                bcs_list.append(bc)
            elif isinstance(bc, self.backend.EquationBC):
                bcs = tuple(bc.extract_form('J'))
                for i in bcs:
                    if isinstance(i, self.backend.DirichletBC):
                        bcs_list.append(self.compat.create_bc(i, homogenize=True))
                    else:
                        bcs_list.append(i)
        bcs = tuple(bcs_list)
        return bcs

    def _create_initial_guess(self):
        return self.backend.Function(self.function_space)

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output
            if isinstance(c, (self.backend.DirichletBC, self.backend.EquationBC)):
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
            self.backend.Function.assign(func, replace_map[self.func])
            replace_map[self.func] = func
        return ufl.replace(form, replace_map)

    def _should_compute_boundary_adjoint(self, relevant_dependencies):

        # Check if DirichletBC derivative is relevant
        bdy = False
        for _, dep in relevant_dependencies:
            if isinstance(dep.output, (self.backend.DirichletBC, self.backend.EquationBC)):
                bdy = True
                break
        return bdy

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output

        dJdu = adj_inputs[0]

        F_form = self._create_F_form()

        dFdu = self.backend.derivative(F_form,
                                       fwd_block_variable.saved_output,
                                       self.backend.TrialFunction(u.function_space()))
        dFdu_form = self.backend.adjoint(dFdu)
        dJdu = dJdu.copy()

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

    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()
        kwargs = self.assemble_kwargs.copy()
        # Homogenize and apply boundary conditions on adj_dFdu and dJdu.
        bcs = self._process_bcs()
        kwargs["bcs"] = bcs
        kwargs["adj"] = True
        import ipdb; ipdb.set_trace()
        dFdu = self.compat.assemble_adjoint_value(dFdu_adj_form, **kwargs)

        for bc in bcs:
            if isinstance(bc, self.backend.DirichletBC):
                bc.apply(dJdu)

        adj_sol = self.compat.create_function(self.function_space)
        self.compat.linalg_solve(dFdu, adj_sol.vector(), dJdu, *self.adj_args, **self.adj_kwargs)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = self.compat.function_from_vector(self.function_space,
                                                           dJdu_copy - self.compat.assemble_adjoint_value(
                                                               self.backend.action(dFdu_adj_form, adj_sol)))

        return adj_sol, adj_sol_bdy

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if not self.linear and self.func == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            return None

        F_form = prepared["form"]
        adj_sol = prepared["adj_sol"]
        adj_sol_bdy = prepared["adj_sol_bdy"]
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, self.backend.Function):
            trial_function = self.backend.TrialFunction(c.function_space())
        elif isinstance(c, self.backend.Constant):
            mesh = self.compat.extract_mesh_from_form(F_form)
            trial_function = self.backend.TrialFunction(c._ad_function_space(mesh))
        elif isinstance(c, self.compat.ExpressionType):
            mesh = F_form.ufl_domain().ufl_cargo()
            c_fs = c._ad_function_space(mesh)
            trial_function = self.backend.TrialFunction(c_fs)
        elif isinstance(c, self.backend.DirichletBC):
            tmp_bc = self.compat.create_bc(c, value=self.compat.extract_subfunction(adj_sol_bdy, c.function_space()))
            return [tmp_bc]
        elif isinstance(c, self.compat.MeshType):
            # Using CoordianteDerivative requires us to do action before
            # differentiating, might change in the future.
            F_form_tmp = self.backend.action(F_form, adj_sol)
            X = self.backend.SpatialCoordinate(c_rep)
            dFdm = self.backend.derivative(-F_form_tmp, X, self.backend.TestFunction(c._ad_function_space()))

            dFdm = self.compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)
            return dFdm

        dFdm = -self.backend.derivative(F_form, c_rep, trial_function)
        dFdm = self.backend.adjoint(dFdm)
        dFdm = dFdm * adj_sol
        dFdm = self.compat.assemble_adjoint_value(dFdm, **self.assemble_kwargs)
        if isinstance(c, self.compat.ExpressionType):
            return [[dFdm, c_fs]]
        else:
            return dFdm

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        fwd_block_variable = self.get_outputs()[0]
        u = fwd_block_variable.output

        F_form = self._create_F_form()

        # Obtain dFdu.
        dFdu = self.backend.derivative(F_form,
                                       fwd_block_variable.saved_output,
                                       self.backend.TrialFunction(u.function_space()))

        return {
            "form": F_form,
            "dFdu": dFdu
        }

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):

        F_form = prepared["form"]
        dFdu = prepared["dFdu"]
        V = self.get_outputs()[idx].output.function_space()

        bcs = []
        dFdm = 0.
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, (self.backend.DirichletBC, self.backend.EquationBC)):
                if tlm_value is None:
                    bcs.append(self.compat.create_bc(c, homogenize=True))
                else:
                    bcs.append(tlm_value)
                continue
            elif isinstance(c, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c)
                c_rep = X

            if tlm_value is None:
                continue

            if c == self.func and not self.linear:
                continue

            dFdm += self.backend.derivative(-F_form, c_rep, tlm_value)

        if isinstance(dFdm, float):
            v = dFdu.arguments()[0]
            dFdm = self.backend.inner(self.backend.Constant(numpy.zeros(v.ufl_shape)), v) * self.backend.dx

        dFdm = ufl.algorithms.expand_derivatives(dFdm)
        dFdm = self.compat.assemble_adjoint_value(dFdm)
        dudm = self.backend.Function(V)
        return self._assemble_and_solve_tlm_eq(
            self.compat.assemble_adjoint_value(dFdu, bcs=bcs, **self.assemble_kwargs), dFdm, dudm, bcs)

    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        return self._assembled_solve(dFdu, dFdm, dudm, bcs)

    def _assemble_soa_eq_rhs(self, dFdu_form, adj_sol, hessian_input, d2Fdu2):
        # Start piecing together the rhs of the soa equation
        b = hessian_input.copy()
        if len(d2Fdu2.integrals()) > 0:
            b_form = self.backend.action(self.backend.adjoint(d2Fdu2), adj_sol)
        else:
            b_form = d2Fdu2

        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value

            if (c == self.func and not self.linear) or tlm_input is None:
                continue

            if isinstance(c, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c)
                dFdu_adj = self.backend.action(self.backend.adjoint(dFdu_form), adj_sol)
                d2Fdudm = ufl.algorithms.expand_derivatives(
                    self.backend.derivative(dFdu_adj, X, tlm_input))
                if len(d2Fdudm.integrals()) > 0:
                    b_form += d2Fdudm
            elif not isinstance(c, (self.backend.DirichletBC, self.backend.EquationBC)):
                dFdu_adj = self.backend.action(self.backend.adjoint(dFdu_form), adj_sol)
                b_form += self.backend.derivative(dFdu_adj, c_rep, tlm_input)

        b_form = ufl.algorithms.expand_derivatives(b_form)
        if len(b_form.integrals()) > 0:
            b -= self.compat.assemble_adjoint_value(b_form)

        return b

    def _assemble_and_solve_soa_eq(self, dFdu_form, adj_sol, hessian_input, d2Fdu2, compute_bdy):
        b = self._assemble_soa_eq_rhs(dFdu_form, adj_sol, hessian_input, d2Fdu2)
        dFdu_form = self.backend.adjoint(dFdu_form)
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_adj_eq(dFdu_form, b, compute_bdy)
        if self.adj2_cb is not None:
            self.adj2_cb(adj_sol2)
        if self.adj2_bdy_cb is not None and compute_bdy:
            self.adj2_bdy_cb(adj_sol2_bdy)
        return adj_sol2, adj_sol2_bdy

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        # First fetch all relevant values
        fwd_block_variable = self.get_outputs()[0]
        hessian_input = hessian_inputs[0]
        tlm_output = fwd_block_variable.tlm_value

        if hessian_input is None:
            return

        if tlm_output is None:
            return

        F_form = self._create_F_form()

        # Using the equation Form we derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = self.backend.derivative(F_form, fwd_block_variable.saved_output)
        d2Fdu2 = ufl.algorithms.expand_derivatives(
            self.backend.derivative(dFdu_form, fwd_block_variable.saved_output, tlm_output))

        adj_sol = self.adj_sol
        if adj_sol is None:
            raise RuntimeError("Hessian computation was run before adjoint.")
        bdy = self._should_compute_boundary_adjoint(relevant_dependencies)
        adj_sol2, adj_sol2_bdy = self._assemble_and_solve_soa_eq(dFdu_form, adj_sol, hessian_input, d2Fdu2, bdy)

        r = {}
        r["adj_sol2"] = adj_sol2
        r["adj_sol2_bdy"] = adj_sol2_bdy
        r["form"] = F_form
        r["adj_sol"] = adj_sol
        return r

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):

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
        import ipdb; ipdb.set_trace()
        if isinstance(c, self.backend.DirichletBC):
            tmp_bc = self.compat.create_bc(c, value=self.compat.extract_subfunction(adj_sol2_bdy, c.function_space()))
            return [tmp_bc]

        if isinstance(c_rep, self.backend.Constant):
            mesh = self.compat.extract_mesh_from_form(F_form)
            W = c._ad_function_space(mesh)
        elif isinstance(c, self.compat.ExpressionType):
            mesh = F_form.ufl_domain().ufl_cargo()
            W = c._ad_function_space(mesh)
        elif isinstance(c, self.compat.MeshType):
            X = self.backend.SpatialCoordinate(c)
            W = c._ad_function_space()
        else:
            W = c.function_space()

        dc = self.backend.TestFunction(W)
        form_adj = self.backend.action(F_form, adj_sol)
        form_adj2 = self.backend.action(F_form, adj_sol2)
        if isinstance(c, self.compat.MeshType):
            dFdm_adj = self.backend.derivative(form_adj, X, dc)
            dFdm_adj2 = self.backend.derivative(form_adj2, X, dc)
        else:
            dFdm_adj = self.backend.derivative(form_adj, c_rep, dc)
            dFdm_adj2 = self.backend.derivative(form_adj2, c_rep, dc)

        # TODO: Old comment claims this might break on split. Confirm if true or not.
        d2Fdudm = ufl.algorithms.expand_derivatives(
            self.backend.derivative(dFdm_adj, fwd_block_variable.saved_output,
                                    tlm_output))

        d2Fdm2 = 0
        # We need to add terms from every other dependency
        # i.e. the terms d^2F/dm_1dm_2
        for _, bv in relevant_dependencies:
            c2 = bv.output
            c2_rep = bv.saved_output

            if isinstance(c2, (self.backend.DirichletBC, self.backend.EquationBC)):
                continue

            tlm_input = bv.tlm_value
            if tlm_input is None:
                continue

            if c2 == self.func and not self.linear:
                continue

            # TODO: If tlm_input is a Sum, this crashes in some instances?
            if isinstance(c2_rep, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c2_rep)
                d2Fdm2 += ufl.algorithms.expand_derivatives(self.backend.derivative(dFdm_adj, X, tlm_input))
            else:
                d2Fdm2 += ufl.algorithms.expand_derivatives(self.backend.derivative(dFdm_adj, c2_rep, tlm_input))

        hessian_form = ufl.algorithms.expand_derivatives(d2Fdm2 + dFdm_adj2 + d2Fdudm)
        hessian_output = 0
        if not hessian_form.empty():
            hessian_output -= self.compat.assemble_adjoint_value(hessian_form)

        if isinstance(c, self.compat.ExpressionType):
            return [(hessian_output, W)]
        else:
            return hessian_output

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self._replace_recompute_form()

    def _replace_recompute_form(self):

        func = self._create_initial_guess()

        bcs = self._recover_bcs()
        bcs_new = []
        for bc in bcs:
            if isinstance(bc, self.backend.EquationBC):
                bc_lhs = self._replace_form(bc.eq.lhs, func=func)
                if bc.is_linear:
                    bc_rhs = self._replace_form(bc.eq.rhs, func=func)
                else:
                    bc_rhs = bc.eq.rhs
                bbcs = []
                for block_variable in bc.block.get_dependencies():
                    c = block_variable.output
                    c_rep = block_variable.saved_output
                    if isinstance(c, (self.backend.DirichletBC)):
                        bbcs.append(c_rep)                
                bcs_new.append(self.backend.EquationBC(bc_lhs == bc_rhs, func, bc.sub_domain, bcs = bbcs))
            else:
                bcs_new.append(bc)
        lhs = self._replace_form(self.lhs, func=func)
        rhs = 0
        if self.linear:
            rhs = self._replace_form(self.rhs)

        return lhs, rhs, func, bcs_new

    def _forward_solve(self, lhs, rhs, func, bcs):
        self.backend.solve(lhs == rhs, func, bcs, *self.forward_args, **self.forward_kwargs)
        return func

    def _assembled_solve(self, lhs, rhs, func, bcs, **kwargs):
        for bc in bcs:
            bc.apply(rhs)
        self.backend.solve(lhs, func.vector(), rhs, **kwargs)
        return func

    def recompute_component(self, inputs, block_variable, idx, prepared):
        lhs = prepared[0]
        rhs = prepared[1]
        func = prepared[2]
        bcs = prepared[3]
        return self._forward_solve(lhs, rhs, func, bcs)


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

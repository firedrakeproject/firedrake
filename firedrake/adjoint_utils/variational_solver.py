import copy
from functools import wraps
from pyadjoint.tape import get_working_tape, stop_annotating, annotate_tape, no_annotations
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock, CachedSolverBlock
from firedrake.ufl_expr import derivative, adjoint, action
from ufl import replace, Action
from ufl.algorithms import expand_derivatives


class NonlinearVariationalProblemMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._ad_F = self.F
            self._ad_u = self.u_restrict
            self._ad_bcs = self.bcs
            self._ad_J = self.J
            try:
                # Some forms (e.g. SLATE tensors) are not currently
                # differentiable.
                dFdu = derivative(self.F, self.u_restrict)
                try:
                    self._ad_adj_F = adjoint(dFdu)
                except ValueError:
                    # Try again without expanding derivatives,
                    # as dFdu might have been simplied to an empty Form
                    self._ad_adj_F = adjoint(dFdu, derivatives_expanded=True)
            except (TypeError, NotImplementedError):
                self._ad_adj_F = None
            self._ad_kwargs = {'Jp': self.Jp, 'form_compiler_parameters': self.form_compiler_parameters, 'is_linear': self.is_linear}
            self._ad_count_map = {}
        return wrapper

    def _ad_count_map_update(self, updated_ad_count_map):
        self._ad_count_map = updated_ad_count_map


class NonlinearVariationalSolverMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, problem, *args, **kwargs):
            self.ad_block_tag = kwargs.pop("ad_block_tag", None)
            init(self, problem, *args, **kwargs)
            self._ad_problem = problem
            self._ad_args = args
            self._ad_kwargs = kwargs
            self._ad_solvers = {"forward_nlvs": None, "adjoint_lvs": None,
                                "recompute_count": 0}
            self._ad_adj_cache = {}

            self._ad_solver_cache = {}

        return wrapper

    def _ad_cache_forward_solver(self):
        from firedrake import (
            DirichletBC,
            NonlinearVariationalProblem,
            NonlinearVariationalSolver)
        from firedrake.adjoint_utils.blocks.solving import FORWARD

        problem = self._ad_problem

        # Build a new form so that we can update the coefficient
        # values without affecting user code.
        # We do this by copying all coefficients in the form and
        # symbolically replacing the old values with the new.
        F = problem.F
        replace_map = {}
        for old_coeff in F.coefficients():
            new_coeff = old_coeff.copy(deepcopy=True)
            replace_map[old_coeff] = new_coeff

        # We need a handle to the new Function being
        # solved for so that we can create an NLVS.
        Fnew = replace(F, replace_map)
        unew = replace_map[problem.u]

        # We also need to "replace" all the bcs in
        # the new NLVS so we can modify those values
        # without affecting user code.
        # Note that ``DirichletBC.reconstruct`` will
        # return ``self`` if V, g, and sub_domain are
        # all unchanged, so we need to explicitly
        # instantiate a new object.
        bcs = problem.bcs
        bcs_new = [
            DirichletBC(V=bc.function_space(),
                        g=bc.function_arg,
                        sub_domain=bc.sub_domain)
            for bc in bcs
        ]

        # This NLVS will be used to recompute the solve.
        # TODO: solver_parameters
        nlvp = NonlinearVariationalProblem(Fnew, unew, bcs=bcs_new)
        nlvs = NonlinearVariationalSolver(nlvp)

        # The original coefficients will be added as
        # dependencies to all solve blocks.
        # The block need handles to the newly created
        # objects to update their values when recomputing.
        self._ad_dependencies_to_add = (*replace_map.keys(), *bcs)
        self._ad_replaced_dependencies = tuple(replace_map.values())
        self._ad_bcs = bcs_new
        self._ad_solver_cache[FORWARD] = nlvs

    def _ad_cache_tlm_solver(self):
        from firedrake import (
            Function, Cofunction, derivative, TrialFunction,
            LinearVariationalProblem, LinearVariationalSolver)
        from firedrake.adjoint_utils.blocks.solving import FORWARD, TLM

        # If we build the TLM form from the cached
        # forward solve form then we can update exactly
        # the same coefficients/boundary conditions.
        nlvp = self._ad_solver_cache[FORWARD]._problem

        F = nlvp.F
        u = nlvp.u
        V = u.function_space()

        # We need gradient of output/input i.e. du/dm.
        # We know F(u; m) = 0 and _total_ dF/dm = 0.
        # Then for the _partial_ derivatives:
        # (dF/du)*(du/dm) + dF/dm = 0 so we calculate:
        # (dF/du)*(du/dm) = -dF/dm
        dFdu = derivative(F, u, TrialFunction(V))
        dFdm = Cofunction(V.dual())
        dudm = Function(V)

        self._ad_dFdu = dFdu

        # Reuse the same bcs as the forward problem.
        # TODO: Think about if we should use new bcs.
        # TODO: solver_parameters
        lvp = LinearVariationalProblem(dFdu, dFdm, dudm, bcs=self._ad_bcs)
        lvs = LinearVariationalSolver(lvp)

        self._ad_solver_cache[TLM] = lvs
        self._ad_tlm_rhs = dFdm

        # Do all the symbolic work for calculating dF/dm up front
        # so we only pay for the numeric calculations at run time.
        replaced_tlms = []
        dFdm_tlm_forms = []
        for m in self._ad_replaced_dependencies:
            mtlm = m.copy(deepcopy=True)
            replaced_tlms.append(mtlm)

            dFdm = derivative(-F, m, mtlm)
            # TODO: Do we need expand_derivatives here? If so, why?
            dFdm = expand_derivatives(dFdm)
            dFdm_tlm_forms.append(dFdm)

        # We'll need to update the replaced_tlm
        # values and assemble the dFdm forms
        self._ad_tlm_dFdm_forms = dFdm_tlm_forms
        self._ad_replaced_tlms = replaced_tlms

    def _ad_cache_adj_solver(self):
        from firedrake import (
            Function, Cofunction, TrialFunction, Argument,
            LinearVariationalProblem, LinearVariationalSolver)
        from firedrake.adjoint_utils.blocks.solving import FORWARD, ADJOINT

        # If we build the adjoint form from the cached
        # forward solve form then we can update exactly
        # the same coefficients/boundary conditions.
        nlvp = self._ad_solver_cache[FORWARD]._problem

        F = nlvp.F
        u = nlvp.u
        V = u.function_space()

        # TODO: rewrite for adjoint not TLM
        # We need gradient of output/input i.e. du/dm.
        # We know F(u; m) = 0 and _total_ dF/dm = 0.
        # Then for the _partial_ derivatives:
        # (dF/du)*(du/dm) + dF/dm = 0 so we calculate:
        # (dF/du)*(du/dm) = -dF/dm
        dFdu = self._ad_dFdu
        try:
            dFdu_adj = adjoint(dFdu)
        except ValueError:
            # Try again without expanding derivatives,
            # as dFdu might have been simplied to an empty Form
            dFdu_adj = adjoint(dFdu, derivatives_expanded=True)

        self._ad_dFdu_adj = dFdu_adj

        # This will be the rhs of the adjoint problem
        dJdu = Cofunction(V.dual())
        adj_sol = Function(V)

        # Reuse the same bcs as the forward problem.
        # TODO: Think about if we should use new bcs.
        # TODO: solver_parameters
        lvp = LinearVariationalProblem(dFdu_adj, dJdu, adj_sol, bcs=self._ad_bcs)
        lvs = LinearVariationalSolver(lvp)

        self._ad_solver_cache[ADJOINT] = lvs
        self._ad_adj_rhs = dJdu

        # Do all the symbolic work for calculating dJ/du up front
        # so we only pay for the numeric calculations at run time.
        dFdm_adj_forms = []
        for m in self._ad_replaced_dependencies:
            # Action of adjoint solution on dFdm
            # TODO: Which of the two implementations should we use?
            dFdm = derivative(-F, m, TrialFunction(m.function_space()))

            # 1. from previous cached implementation
            dFdm = adjoint(dFdm)
            if isinstance(dFdm, Argument):
                #  Corner case. Should be fixed more permanently upstream in UFL.
                #  See: https://github.com/FEniCS/ufl/issues/395
                dFdm = Action(dFdm, adj_sol)
            else:
                dFdm = dFdm * adj_sol

            # 2. from GenericSolveBlock
            # if isinstance(dFdm, ufl.Form):
            #     dFdm = adjoint(dFdm)
            #     dFdm = action(dFdm, adj_sol)
            # else:
            #     dFdm = dFdm(adj_sol)

            dFdm_adj_forms.append(dFdm)

        # To calculate the adjoint component of each DirichletBC
        # we'll need the residual of the adjoint equation without
        # any DirichletBC using the solution calculated with
        # homogeneous DirichletBCs.
        self._ad_adj_residual = dJdu - action(dFdu_adj, adj_sol)

        # We'll need to assemble these forms to calculate
        # the adj_component for each dependency.
        self._ad_adj_dFdm_forms = dFdm_adj_forms

    def _ad_cache_hessian_solver(self):
        from firedrake import (
            Function, TestFunction)
        from firedrake.adjoint_utils.blocks.solving import FORWARD

        nlvp = self._ad_solver_cache[FORWARD]._problem
        F = nlvp.F
        u = nlvp.u
        V = u.function_space()

        # 1. Forms to calculate rhs of Hessian solve

        # Calculate d^2F/du^2 * du/dm * dm
        # where dm is direction for tlm action so du/dm * dm is tlm output
        dFdu = self._ad_dFdu
        tlm_output = Function(V)
        d2Fdu2 = expand_derivatives(
            derivative(dFdu, u, tlm_output))

        self._ad_tlm_output = tlm_output

        adj_sol = Function(V)
        self._ad_adj_sol = adj_sol

        # Contribution from tlm_output
        if len(d2Fdu2.integrals()) > 0:
            d2Fdu2_form = action(adjoint(d2Fdu2), adj_sol)
        else:
            d2Fdu2_form = d2Fdu2
        self._ad_d2Fdu2_form = d2Fdu2_form

        # Contributions from each tlm_input
        dFdu_adj = action(self._ad_dFdu_adj, adj_sol)
        d2Fdmdu_forms = []
        for m, dm in zip(self._ad_replaced_dependencies,
                         self._ad_replaced_tlms):
            d2Fdmdu = expand_derivatives(
                derivative(dFdu_adj, m, dm))

            d2Fdmdu_forms.append(d2Fdmdu)

        self._ad_d2Fdmdu_forms = d2Fdmdu_forms

        # 2. Forms to calculate contribution from each control
        adj2_sol = Function(V)
        self._ad_adj2_sol = adj2_sol

        Fadj = action(F, adj_sol)
        Fadj2 = action(F, adj2_sol)

        dFdm_adj2_forms = []
        d2Fdm2_adj_forms = []
        d2Fdudm_forms = []
        for m in self._ad_replaced_dependencies:
            dm = TestFunction(m.function_space())
            dFdm_adj2 = expand_derivatives(
                derivative(Fadj2, m, dm))

            dFdm_adj2_forms.append(dFdm_adj2)

            dFdm_adj = derivative(Fadj, m, dm)

            d2Fdudm = expand_derivatives(
                derivative(dFdm_adj, u, tlm_output))

            d2Fdudm_forms.append(d2Fdudm)

            d2Fdm2_adj_forms_k = []
            for m2, dm2 in zip(self._ad_replaced_dependencies,
                               self._ad_replaced_tlms):
                d2Fdm2_adj = expand_derivatives(
                    derivative(dFdm_adj, m2, dm2))
                d2Fdm2_adj_forms_k.append(d2Fdm2_adj)

            d2Fdm2_adj_forms.append(d2Fdm2_adj_forms_k)

        self._ad_dFdm_adj2_forms = dFdm_adj2_forms
        self._ad_d2Fdm2_adj_forms = d2Fdm2_adj_forms
        self._ad_d2Fdudm_forms = d2Fdudm_forms

    @staticmethod
    def _ad_annotate_solve(solve):
        @wraps(solve)
        def wrapper(self, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""
            annotate = annotate_tape(kwargs)
            if annotate:
                if kwargs.pop("bounds", None) is not None:
                    raise ValueError(
                        "MissingMathsError: we do not know how to differentiate through a variational inequality")

                if len(self._ad_solver_cache) == 0:
                    self._ad_cache_forward_solver()
                    self._ad_cache_tlm_solver()
                    self._ad_cache_adj_solver()
                    self._ad_cache_hessian_solver()

                block = CachedSolverBlock(self._ad_problem.u,
                                          self._ad_bcs,
                                          self._ad_solver_cache,
                                          self._ad_replaced_dependencies,

                                          self._ad_tlm_rhs,
                                          self._ad_replaced_tlms,
                                          self._ad_tlm_dFdm_forms,

                                          self._ad_adj_rhs,
                                          self._ad_adj_dFdm_forms,
                                          self._ad_adj_residual,

                                          self._ad_adj_sol,
                                          self._ad_adj2_sol,
                                          self._ad_tlm_output,
                                          self._ad_d2Fdu2_form,
                                          self._ad_d2Fdmdu_forms,
                                          self._ad_dFdm_adj2_forms,
                                          self._ad_d2Fdm2_adj_forms,
                                          self._ad_d2Fdudm_forms,

                                          ad_block_tag=self.ad_block_tag)

                for dep in self._ad_dependencies_to_add:
                    block.add_dependency(dep, no_duplicates=True)

                get_working_tape().add_block(block)

            with stop_annotating():
                out = solve(self, **kwargs)

            if annotate:
                block.add_output(self._ad_problem._ad_u.create_block_variable())

            return out

        return wrapper

    @staticmethod
    def _ad_annotate_solve_old(solve):
        @wraps(solve)
        def wrapper(self, **kwargs):
            """To disable the annotation, just pass :py:data:`annotate=False` to this routine, and it acts exactly like the
            Firedrake solve call. This is useful in cases where the solve is known to be irrelevant or diagnostic
            for the purposes of the adjoint computation (such as projecting fields to other function spaces
            for the purposes of visualisation)."""
            from firedrake import LinearVariationalSolver
            annotate = annotate_tape(kwargs)
            if annotate:
                bounds = kwargs.pop("bounds", None)
                if bounds is not None:
                    raise ValueError(
                        "MissingMathsError: we do not know how to differentiate through a variational inequality")

                tape = get_working_tape()
                problem = self._ad_problem
                sb_kwargs = NonlinearVariationalSolveBlock.pop_kwargs(kwargs)
                sb_kwargs.update(kwargs)

                block = NonlinearVariationalSolveBlock(problem._ad_F == 0,
                                                       problem._ad_u,
                                                       problem._ad_bcs,
                                                       adj_cache=self._ad_adj_cache,
                                                       problem_J=problem._ad_J,
                                                       solver_kwargs=self._ad_kwargs,
                                                       ad_block_tag=self.ad_block_tag,
                                                       **sb_kwargs)

                # Forward variational solver.
                if not self._ad_solvers["forward_nlvs"]:
                    self._ad_solvers["forward_nlvs"] = type(self)(
                        self._ad_problem_clone(self._ad_problem, block.get_dependencies()),
                        **self._ad_kwargs
                    )

                # Adjoint variational solver.
                if not self._ad_solvers["adjoint_lvs"]:
                    with stop_annotating():
                        self._ad_solvers["adjoint_lvs"] = LinearVariationalSolver(
                            self._ad_adj_lvs_problem(block, problem._ad_adj_F),
                            *block.adj_args, **block.adj_kwargs)
                        if self._ad_problem._constant_jacobian:
                            self._ad_solvers["update_adjoint"] = False

                block._ad_solvers = self._ad_solvers

                tape.add_block(block)

            with stop_annotating():
                out = solve(self, **kwargs)

            if annotate:
                block.add_output(self._ad_problem._ad_u.create_block_variable())

            return out

        return wrapper

    @no_annotations
    def _ad_problem_clone(self, problem, dependencies):
        """Replaces every coefficient in the residual and jacobian with a deepcopy to return
        a clone of the original NonlinearVariationalProblem instance. We'll be modifying the
        numerical values of the coefficients in the residual and jacobian, so in order not to
        affect the user-defined self._ad_problem.F, self._ad_problem.J and self._ad_problem.u
        expressions, we'll instead create clones of them.
        """
        from firedrake import NonlinearVariationalProblem
        _ad_count_map, J_replace_map, F_replace_map = self._build_count_map(
            problem.J, dependencies, F=problem.F)
        nlvp = NonlinearVariationalProblem(replace(problem.F, F_replace_map),
                                           F_replace_map[problem.u_restrict],
                                           bcs=problem.bcs,
                                           J=replace(problem.J, J_replace_map))
        nlvp.is_linear = problem.is_linear
        nlvp._constant_jacobian = problem._constant_jacobian
        nlvp._ad_count_map_update(_ad_count_map)
        return nlvp

    @no_annotations
    def _ad_adj_lvs_problem(self, block, adj_F):
        """Create the adjoint variational problem."""
        from firedrake import Function, Cofunction, LinearVariationalProblem
        # Homogeneous boundary conditions for the adjoint problem
        # when Dirichlet boundary conditions are applied.
        bcs = block._homogenize_bcs()
        adj_sol = Function(block.function_space)
        right_hand_side = Cofunction(block.function_space.dual())
        tmp_problem = LinearVariationalProblem(
            adj_F, right_hand_side, adj_sol, bcs=bcs,
            constant_jacobian=self._ad_problem._constant_jacobian)
        # The `block.adj_F` coefficients hold the output references.
        # We do not want to modify the user-defined values. Hence, the adjoint
        # linear variational problem is created with a deep copy of the
        # `block.adj_F` coefficients.
        _ad_count_map, J_replace_map, _ = self._build_count_map(
            adj_F, block._dependencies)
        lvp = LinearVariationalProblem(
            replace(tmp_problem.J, J_replace_map), right_hand_side, adj_sol,
            bcs=tmp_problem.bcs,
            constant_jacobian=self._ad_problem._constant_jacobian)
        lvp._ad_count_map_update(_ad_count_map)
        return lvp

    def _build_count_map(self, J, dependencies, F=None):
        from firedrake import Function

        F_replace_map = {}
        J_replace_map = {}
        if F:
            F_coefficients = F.coefficients()
        J_coefficients = J.coefficients()

        _ad_count_map = {}
        for block_variable in dependencies:
            coeff = block_variable.output
            if F:
                if coeff in F_coefficients and coeff not in F_replace_map:
                    if isinstance(coeff, Function) and coeff.ufl_element().family() == "Real":
                        F_replace_map[coeff] = copy.deepcopy(coeff)
                    else:
                        F_replace_map[coeff] = coeff.copy(deepcopy=True)
                    _ad_count_map[F_replace_map[coeff]] = coeff.count()

            if coeff in J_coefficients and coeff not in J_replace_map:
                if coeff in F_replace_map:
                    J_replace_map[coeff] = F_replace_map[coeff]
                elif isinstance(coeff, Function) and coeff.ufl_element().family() == "Real":
                    J_replace_map[coeff] = copy.deepcopy(coeff)
                else:
                    J_replace_map[coeff] = coeff.copy()
                _ad_count_map[J_replace_map[coeff]] = coeff.count()
        return _ad_count_map, J_replace_map, F_replace_map

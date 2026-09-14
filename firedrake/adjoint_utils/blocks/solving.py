from enum import Enum

from pyadjoint import Block, stop_annotating
import firedrake
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint


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


class SolverType(Enum):
    """Enum for solver types."""
    FORWARD = 0
    ADJOINT = 1
    TLM = 2
    HESSIAN = 3


FORWARD = SolverType.FORWARD
ADJOINT = SolverType.ADJOINT
TLM = SolverType.TLM
HESSIAN = SolverType.HESSIAN


class CachedSolverBlock(Block):
    def __init__(
        self, forward_cache, tangent_cache, adjoint_cache, hessian_cache, ad_block_tag=None
    ):
        super().__init__(ad_block_tag=ad_block_tag)

        self.forward_cache = forward_cache
        self.tangent_cache = tangent_cache
        self.adjoint_cache = adjoint_cache
        self.hessian_cache = hessian_cache

        # The adj_sol in the cached forms is shared by all blocks.
        # This adj_sol belongs to this block specifically so we can
        # stash the adjoint solution for the hessian calculation.
        self.adj_sol = adjoint_cache.adj_sol.copy(deepcopy=True, annotate=False)

    def _coefficient_dependencies(self, dependencies=None):
        dependencies = dependencies or self.get_dependencies()
        return dependencies[:len(self.forward_cache.replaced_deps)]

    def _mesh_dependencies(self, dependencies=None):
        dependencies = dependencies or self.get_dependencies()
        len_replaced = len(self.forward_cache.replaced_deps)
        len_meshes = len(self.forward_cache.meshes)
        return dependencies[len_replaced:len_replaced + len_meshes]

    def _bc_dependencies(self, dependencies=None):
        dependencies = dependencies or self.get_dependencies()
        if len(self.forward_cache.bcs) > 0:
            return dependencies[-len(self.forward_cache.bcs):]
        else:
            return []

    def update_dependencies(self, use_output=False):
        """Update all dependencies of the forward solve.
        """
        # Update the coefficients in the form.
        # Use the fact that zip will use the shorter length.
        for replaced_dep, dep in zip(self.forward_cache.replaced_deps,
                                     self._coefficient_dependencies()):
            replaced_dep.assign(dep.saved_output)

        # 1. For forward recomputation the unknown Function should use
        # the incoming value of the dependency as the initial guess.
        # 2. For the adjoint, TLM, and Hessian, the unknown Function
        # should use the computed value so that the linearised
        # Jacobian is correct.
        if use_output:
            output = self.get_outputs()[0].saved_output
            self.forward_cache.solver._problem.u.assign(output)

        # Update the boundary conditions
        for replaced_dep, dep in zip(self.forward_cache.bcs, self._bc_dependencies()):
            replaced_dep.set_value(dep.saved_output.function_arg)

    def update_tlm_dependencies(self):
        """Update all dependencies of the tlm solve.
        """
        for replaced_dep, dep in zip(self.tangent_cache.replaced_tlms,
                                     self._coefficient_dependencies()):
            if dep.output == self.forward_cache.func:
                continue
            if dep.tlm_value is None:  # This dependency doesn't depend on the controls
                continue
            replaced_dep.assign(dep.tlm_value)

        for replaced_dep, dep in zip(self.tangent_cache.mesh_tlms,
                                     self._mesh_dependencies()):
            if dep.tlm_value is None:
                continue
            replaced_dep.assign(dep.tlm_value)

        for replaced_dep, dep in zip(self.forward_cache.bcs, self._bc_dependencies()):
            if dep.tlm_value is None:  # This dependency doesn't depend on the controls
                bc_val = 0
            else:
                bc_val = dep.tlm_value.function_arg
            replaced_dep.set_value(bc_val)

    def update_adj_dependencies(self):
        # TODO: Anything to do here?
        pass

    def update_hessian_dependencies(self):
        # TODO: Anything else to do here?
        self.update_tlm_dependencies()
        # update the adj_sol in the cached forms with
        # the adj_sol value owned by this block.
        self.hessian_cache.adj_sol.assign(self.adj_sol)

    def _compute_boundary(self, relevant_dependencies):
        return any(isinstance(dep.output, firedrake.DirichletBC)
                   for _, dep in relevant_dependencies)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return

    def recompute_component(self, inputs, block_variable, idx, prepared):
        self.update_dependencies(use_output=False)

        solver = self.forward_cache.solver
        solver.solve()
        result = solver._problem.u.copy(deepcopy=True)

        # Possibly checkpoint the result for the adjoint solve later.
        if isinstance(block_variable.checkpoint, firedrake.Function):
            result = block_variable.checkpoint.assign(result)

        return maybe_disk_checkpoint(result)

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        self.update_dependencies(use_output=True)
        self.update_tlm_dependencies()

        # Assemble the rhs of (dF/du)(du/dm) = -dF/dm
        tlm_rhs = self.tangent_cache.rhs
        tlm_rhs.zero()
        for dFdm, dep in zip(self.tangent_cache.dFdm_forms, self.get_dependencies()):
            if dep.tlm_value is None:  # This dependency doesn't depend on the controls
                continue
            if dep.output is self.forward_cache.func:  # Can't compute dependence on initial guess
                continue
            tlm_rhs += firedrake.assemble(dFdm)

        # Solve for dudm
        solver = self.tangent_cache.solver
        solver._problem.u.zero()
        solver.solve()
        result = solver._problem.u.copy(deepcopy=True)
        return result

    def solve_adj_equation(self, rhs, compute_boundary):
        for bc in self.forward_cache.bcs:
            bc.homogenize()

        adj_rhs = self.adjoint_cache.rhs
        adj_sol = self.adjoint_cache.adj_sol

        adj_rhs.assign(rhs)
        adj_sol.zero()
        self.adjoint_cache.solver.solve()

        if compute_boundary:
            adj_sol_bc = firedrake.assemble(self.adjoint_cache.residual)
            adj_sol_bc = adj_sol_bc.riesz_representation("l2")
        else:
            adj_sol_bc = None

        return adj_sol.copy(deepcopy=True), adj_sol_bc

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        self.update_dependencies(use_output=True)
        self.update_adj_dependencies()

        dJdu = adj_inputs[0]

        compute_boundary = self._compute_boundary(relevant_dependencies)

        adj_sol, adj_sol_bc = self.solve_adj_equation(dJdu, compute_boundary)

        # store adj_sol for Hessian computation later, or for inspecting
        # adjoint sensitivities etc.
        # self.adj_sol is owned by this block, whereas self.hessian_cache.adj_sol
        # is shared between all blocks that the NLVS generates because it is the
        # one in the cached forms, so we will update it as necessary if/when each
        # block calculates the Hessian action.
        self.adj_sol.assign(adj_sol)

        prepared = {
            "adj_sol": adj_sol,
            "adj_sol_bc": adj_sol_bc
        }
        return prepared

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if block_variable.output == self.forward_cache.func:
            return None

        if isinstance(block_variable.output, firedrake.DirichletBC):
            bc = block_variable.output
            adj_sol_bc = prepared["adj_sol_bc"]
            return bc.reconstruct(
                g=extract_subfunction(adj_sol_bc, bc.function_space())
            )

        # assemble sensititivy comment
        dFdm = firedrake.assemble(self.adjoint_cache.dFdm_forms[idx])

        return dFdm

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        self.update_dependencies(use_output=True)
        self.update_hessian_dependencies()

        hessian_input = hessian_inputs[0]
        tlm_output = self.get_outputs()[0].tlm_value

        if hessian_input is None:
            return
        if tlm_output is None:
            return

        # 1. Assemble rhs

        # hessian input contribution
        hessian_rhs = hessian_input.copy(deepcopy=True)

        # tlm_output contribution
        self.hessian_cache.tlm_output.assign(tlm_output)
        if not self.hessian_cache.d2Fdu2_form.empty():
            hessian_rhs -= firedrake.assemble(self.hessian_cache.d2Fdu2_form)

        # tlm_input contribution
        for d2Fdmdu, dep in zip(self.hessian_cache.d2Fdmdu_forms,
                                self._coefficient_dependencies() + self._mesh_dependencies()):
            if dep.tlm_value is None:  # This dependency doesn't depend on the controls
                continue
            if dep.output is self.forward_cache.func:  # Can't compute dependence on initial guess
                continue
            if len(d2Fdmdu.integrals()) > 0:
                hessian_rhs -= firedrake.assemble(d2Fdmdu)

        # 2. Solve adjoint system
        compute_boundary = self._compute_boundary(relevant_dependencies)
        adj2_sol, adj2_sol_bc = self.solve_adj_equation(hessian_rhs, compute_boundary)

        self.hessian_cache.adj2_sol.assign(adj2_sol)

        prepared = {
            "adj2_sol": adj2_sol,
            "adj2_sol_bc": adj2_sol_bc,
        }

        return prepared

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):
        m = block_variable.output

        if m is self.forward_cache.func:
            return None

        if isinstance(m, firedrake.DirichletBC):
            bc = block_variable.output
            adj2_sol_bc = prepared["adj2_sol_bc"]
            return bc.reconstruct(
                g=extract_subfunction(adj2_sol_bc, bc.function_space())
            )

        relevant_d2Fdm2_forms = []
        for i, dep in relevant_dependencies:
            if i >= len(self._coefficient_dependencies() + self._mesh_dependencies()):
                continue
            if dep.tlm_value is None:
                continue
            if dep.output is self.forward_cache.func:
                continue
            relevant_d2Fdm2_forms.append(self.hessian_cache.d2Fdm2_adj_forms[idx][i])

        hessian_output = 0

        for form in (self.hessian_cache.d2Fdudm_forms[idx],
                     self.hessian_cache.dFdm_adj2_forms[idx],
                     *relevant_d2Fdm2_forms):
            if not form.empty():
                hessian_output += firedrake.assemble(form)

        return hessian_output


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

    if hasattr(self, "tlm_args") and len(self.tlm_args) <= 0:
        self.tlm_args = self.adj_args

    if hasattr(self, "tlm_kwargs") and len(self.tlm_kwargs) <= 0:
        self.tlm_kwargs = self.adj_kwargs.copy()

    solver_params = kwargs.get("solver_parameters", None)
    if solver_params is not None and "mat_type" in solver_params:
        self.assemble_kwargs["mat_type"] = solver_params["mat_type"]

    if varform:
        if "appctx" in kwargs:
            self.assemble_kwargs["appctx"] = kwargs["appctx"]


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

    pop_kwargs_keys = ["adj_cb", "adj_bdy_cb", "adj2_cb", "adj2_bdy_cb",
                       "forward_args", "forward_kwargs", "adj_args",
                       "adj_kwargs"]

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
        b = firedrake.Function(self.target_space.dual())
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

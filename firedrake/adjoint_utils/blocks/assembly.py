import ufl
import firedrake
from ufl.formatting.ufl2unicode import ufl2unicode
from pyadjoint import Block, create_overloaded_object
from .backend import Backend
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint


class AssembleBlock(Block, Backend):
    def __init__(self, form, ad_block_tag=None):
        super(AssembleBlock, self).__init__(ad_block_tag=ad_block_tag)
        self.form = form
        if self.backend.__name__ != "firedrake":
            mesh = self.form.ufl_domain().ufl_cargo()
        else:
            mesh = self.form.ufl_domain()
        self.add_dependency(mesh)
        for c in self.form.coefficients():
            self.add_dependency(c, no_duplicates=True)

    def __str__(self):
        return f"assemble({ufl2unicode(self.form)})"

    def compute_action_adjoint(self, adj_input, arity_form, form=None,
                               c_rep=None, space=None, dform=None):
        """This computes the action of the adjoint of the derivative of `form`
           wrt `c_rep` on `adj_input`. In other words, it returns:
           `<(dform/dc_rep)*, adj_input>`

           - If `form` has arity 0 => `dform/dc_rep` is a 1-form and
             `adj_input` a foat, we can simply use the `*` operator.

           - If `form` has arity 1 => `dform/dc_rep` is a 2-form and we can
             symbolically take its adjoint and then apply the action on
             `adj_input`, to finally assemble the result.
        """
        if arity_form == 0:
            if dform is None:
                dc = self.backend.TestFunction(space)
                dform = self.backend.derivative(form, c_rep, dc)
            dform_vector = self.compat.assemble_adjoint_value(dform)
            # Return a Vector scaled by the scalar `adj_input`
            return dform_vector * adj_input, dform
        elif arity_form == 1:
            if dform is None:
                dc = self.backend.TrialFunction(space)
                dform = self.backend.derivative(form, c_rep, dc)
            # Get the Function
            adj_input = adj_input.function
            # Symbolic operators such as action/adjoint require derivatives to
            # have been expanded beforehand. However, UFL doesn't support
            # expanding coordinate derivatives of Coefficients in physical
            # space, implying that we can't symbolically take the
            # action/adjoint of the Jacobian for SpatialCoordinates. ->
            # Workaround: Apply action/adjoint numerically (using PETSc).
            if not isinstance(c_rep, self.backend.SpatialCoordinate):
                # Symbolically compute: (dform/dc_rep)^* * adj_input
                adj_output = self.backend.action(self.backend.adjoint(dform),
                                                 adj_input)
                adj_output = self.compat.assemble_adjoint_value(adj_output)
            else:
                # Get PETSc matrix
                dform_mat = self.compat.assemble_adjoint_value(dform).petscmat
                # Action of the adjoint (Hermitian transpose)
                adj_output = self.backend.Function(space)
                with adj_input.dat.vec_ro as v_vec:
                    with adj_output.dat.vec as res_vec:
                        dform_mat.multHermitian(v_vec, res_vec)
            return adj_output, dform
        else:
            raise ValueError('Forms with arity > 1 are not handled yet!')

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        form = prepared
        adj_input = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        from ufl.algorithms.analysis import extract_arguments
        arity_form = len(extract_arguments(form))

        if isinstance(c, self.compat.ExpressionType):
            # Create a FunctionSpace from self.form and Expression.
            # And then make a TestFunction from this space.
            mesh = self.form.ufl_domain().ufl_cargo()
            V = c._ad_function_space(mesh)
            dc = self.backend.TestFunction(V)

            dform = self.backend.derivative(form, c_rep, dc)
            output = self.compat.assemble_adjoint_value(dform)
            return [[adj_input * output, V]]

        if self.compat.isconstant(c):
            mesh = self.compat.extract_mesh_from_form(self.form)
            space = c._ad_function_space(mesh)
        elif isinstance(c, self.backend.Function):
            space = c.function_space()
        elif isinstance(c, self.compat.MeshType):
            c_rep = self.backend.SpatialCoordinate(c_rep)
            space = c._ad_function_space()

        return self.compute_action_adjoint(adj_input, arity_form, form, c_rep,
                                           space)[0]

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, tlm_inputs,
                                         self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        form = prepared
        dform = 0.

        from ufl.algorithms.analysis import extract_arguments
        arity_form = len(extract_arguments(form))
        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue
            if isinstance(c_rep, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c_rep)
                dform += self.backend.derivative(form, X, tlm_value)
            else:
                dform += self.backend.derivative(form, c_rep, tlm_value)
        if not isinstance(dform, float):
            dform = ufl.algorithms.expand_derivatives(dform)
            dform = self.compat.assemble_adjoint_value(dform)
            if arity_form == 1 and dform != 0:
                # Then dform is a Vector
                dform = dform.function
        return dform

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                 relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, adj_inputs,
                                         relevant_dependencies)

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        form = prepared
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        from ufl.algorithms.analysis import extract_arguments
        arity_form = len(extract_arguments(form))

        c1 = block_variable.output
        c1_rep = block_variable.saved_output

        if self.compat.isconstant(c1):
            mesh = self.compat.extract_mesh_from_form(form)
            space = c1._ad_function_space(mesh)
        elif isinstance(c1, self.backend.Function):
            space = c1.function_space()
        elif isinstance(c1, self.compat.ExpressionType):
            mesh = form.ufl_domain().ufl_cargo()
            space = c1._ad_function_space(mesh)
        elif isinstance(c1, self.compat.MeshType):
            c1_rep = self.backend.SpatialCoordinate(c1)
            space = c1._ad_function_space()
        else:
            return None

        hessian_outputs, dform = self.compute_action_adjoint(
            hessian_input, arity_form, form, c1_rep, space
        )

        ddform = 0
        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(c2_rep, self.compat.MeshType):
                X = self.backend.SpatialCoordinate(c2_rep)
                ddform += self.backend.derivative(dform, X, tlm_input)
            else:
                ddform += self.backend.derivative(dform, c2_rep, tlm_input)

        if not isinstance(ddform, float):
            ddform = ufl.algorithms.expand_derivatives(ddform)
            if not ddform.empty():
                hessian_outputs += self.compute_action_adjoint(
                    adj_input, arity_form, dform=ddform
                )[0]

        if isinstance(c1, self.compat.ExpressionType):
            return [(hessian_outputs, space)]
        else:
            return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared
        output = self.backend.assemble(form)
        output = create_overloaded_object(output)
        if isinstance(output, firedrake.Function):
            return maybe_disk_checkpoint(output)
        else:
            return output

import ufl
import firedrake
from ufl.domain import as_domain
from ufl.formatting.ufl2unicode import ufl2unicode
from pyadjoint import Block, AdjFloat, create_overloaded_object
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint
from .block_utils import isconstant


class AssembleBlock(Block):
    def __init__(self, form, ad_block_tag=None):
        super(AssembleBlock, self).__init__(ad_block_tag=ad_block_tag)
        self.form = form
        try:
            mesh = as_domain(form)
        except AttributeError:
            mesh = None

        if mesh and not isinstance(self.form, ufl.Interpolate):
            # Interpolation differentiation wrt spatial coordinates is currently not supported.
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
             `adj_input` a float, we can simply use the `*` operator.

           - If `form` has arity 1 => `dform/dc_rep` is a 2-form and we can
             symbolically take its adjoint and then apply the action on
             `adj_input`, to finally assemble the result.
        """
        if arity_form == 0:
            if dform is None:
                dc = firedrake.TestFunction(space)
                dform = firedrake.derivative(form, c_rep, dc)
            dform_adj = firedrake.assemble(dform)
            if dform_adj == 0:
                # `dform_adj` is a `ZeroBaseForm`
                return AdjFloat(0.), dform
            # Return the adjoint model of `form` scaled by the scalar
            # `adj_input`
            adj_output = dform_adj._ad_mul(adj_input)
            return adj_output, dform
        elif arity_form == 1:
            if dform is None:
                dc = firedrake.TrialFunction(space)
                dform = firedrake.derivative(form, c_rep, dc)
            # Symbolic operators such as action/adjoint require derivatives to
            # have been expanded beforehand. However, UFL doesn't support
            # expanding coordinate derivatives of Coefficients in physical
            # space, implying that we can't symbolically take the
            # action/adjoint of the Jacobian for SpatialCoordinates.
            # -> Workaround: Apply action/adjoint numerically (using PETSc).
            if not isinstance(c_rep, firedrake.SpatialCoordinate):
                # Symbolically compute: (dform/dc_rep)^* * adj_input
                adj_output = firedrake.action(firedrake.adjoint(dform),
                                              adj_input)
                adj_output = firedrake.assemble(adj_output)
            else:
                adj_output = firedrake.Cofunction(space.dual())
                # Assemble `dform`: derivatives are expanded along the way
                # which may lead to a ZeroBaseForm
                assembled_dform = firedrake.assemble(dform)
                if assembled_dform == 0:
                    return adj_output, dform
                # Get PETSc matrix
                dform_mat = assembled_dform.petscmat
                # Action of the adjoint (Hermitian transpose)
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

        if isconstant(c):
            mesh = as_domain(self.form)
            space = c._ad_function_space(mesh)
        elif isinstance(c, (firedrake.Function, firedrake.Cofunction)):
            space = c.function_space()
        elif isinstance(c, firedrake.MeshGeometry):
            c_rep = firedrake.SpatialCoordinate(c_rep)
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

        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value

            if tlm_value is None:
                continue
            if isinstance(c_rep, firedrake.MeshGeometry):
                X = firedrake.SpatialCoordinate(c_rep)
                # Spatial coordinates derivatives cannot be expanded in the
                # physical space, which is required by symbolic operators such
                # as `action`.
                dform += firedrake.derivative(form, X, tlm_value)
            else:
                dform += firedrake.action(firedrake.derivative(form, c_rep),
                                          tlm_value)
        if not isinstance(dform, float):
            dform = ufl.algorithms.expand_derivatives(dform)
            dform = firedrake.assemble(dform)
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

        if isconstant(c1):
            mesh = as_domain(form)
            space = c1._ad_function_space(mesh)
        elif isinstance(c1, (firedrake.Function, firedrake.Cofunction)):
            space = c1.function_space()
        elif isinstance(c1, firedrake.MeshGeometry):
            c1_rep = firedrake.SpatialCoordinate(c1)
            space = c1._ad_function_space()
        else:
            return None

        hessian_outputs, dform = self.compute_action_adjoint(
            hessian_input, arity_form, form, c1_rep, space
        )

        ddform = 0.
        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(c2_rep, firedrake.MeshGeometry):
                X = firedrake.SpatialCoordinate(c2_rep)
                ddform += firedrake.derivative(dform, X, tlm_input)
            else:
                ddform += firedrake.derivative(dform, c2_rep, tlm_input)

        if not isinstance(ddform, float):
            ddform = ufl.algorithms.expand_derivatives(ddform)
            if not (isinstance(ddform, ufl.ZeroBaseForm)
                    or (isinstance(ddform, ufl.Form) and ddform.empty())):
                hessian_outputs += self.compute_action_adjoint(
                    adj_input, arity_form, dform=ddform
                )[0]

        return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared
        output = firedrake.assemble(form)
        output = create_overloaded_object(output)
        if isinstance(output, firedrake.Function):
            return maybe_disk_checkpoint(output)
        else:
            return output

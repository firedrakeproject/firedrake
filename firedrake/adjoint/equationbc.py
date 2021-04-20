from functools import wraps
from pyadjoint.overloaded_type import FloatingType
from pyadjoint.tape import no_annotations, annotate_tape, stop_annotating
import ufl
from pyadjoint import Block, OverloadedType


class EquationBCBlock(Block):
    def __init__(self, *args, **kwargs):
        Block.__init__(self)         
        self.add_dependency(args[1])

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        
        bc = self.get_outputs()[0].saved_output
        import ipdb; ipdb.set_trace()
        c = block_variable.output
        adj_inputs = adj_inputs[0]
        adj_output = None
        for adj_input in adj_inputs:
            if isinstance(c, self.backend.Constant):
                adj_value = self.backend.Function(self.parent_space)
                adj_input.apply(adj_value.vector())
                if self.function_space != self.parent_space:
                    vec = self.compat.extract_bc_subvector(adj_value, self.collapsed_space, bc)
                    adj_value = self.compat.function_from_vector(self.collapsed_space, vec)

                if adj_value.ufl_shape == () or adj_value.ufl_shape[0] <= 1:
                    r = adj_value.vector().sum()
                else:
                    output = []
                    subindices = _extract_subindices(self.function_space)
                    for indices in subindices:
                        current_subfunc = adj_value
                        prev_idx = None
                        for i in indices:
                            if prev_idx is not None:
                                current_subfunc = current_subfunc.sub(prev_idx)
                            prev_idx = i
                        output.append(current_subfunc.sub(prev_idx, deepcopy=True).vector().sum())

                    r = self.backend.cpp.la.Vector(self.backend.MPI.comm_world, len(output))
                    r[:] = output
            elif isinstance(c, self.backend.Function):
                adj_value = self.backend.Function(self.parent_space)
                adj_input.apply(adj_value.vector())
                r = self.compat.extract_bc_subvector(adj_value, c.function_space(), bc)
            elif isinstance(c, self.compat.Expression):
                adj_value = self.backend.Function(self.parent_space)
                adj_input.apply(adj_value.vector())
                output = self.compat.extract_bc_subvector(adj_value, self.collapsed_space, bc)
                r = [[output, self.collapsed_space]]
            if adj_output is None:
                adj_output = r
            else:
                adj_output += r
        return adj_output

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        bc = block_variable.saved_output
        import ipdb; ipdb.set_trace()
        for bv in self.get_dependencies():
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if self.function_space != self.parent_space and not isinstance(tlm_input, ufl.Coefficient):
                tlm_input = self.backend.project(tlm_input, self.collapsed_space)

            m = self.compat.create_bc(bc, value=tlm_input)
        return m

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx,
                                   relevant_dependencies, prepared=None):
        import ipdb; ipdb.set_trace()
        # The same as evaluate_adj but with hessian values.
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx)


    @no_annotations
    def recompute(self):
        # There is nothing to do. The checkpoint is weak,
        # so it changes automatically with the dependency checkpoint.
        return

    def __str__(self):
        return "EquationBC block"


def _extract_subindices(V):
    assert V.num_sub_spaces() > 0
    r = []
    for i in range(V.num_sub_spaces()):
        indices_sequence = [i]
        _build_subindices(indices_sequence, r, V.sub(i))
        indices_sequence.pop()
    return r


def _build_subindices(indices_sequence, r, V):
    if V.num_sub_spaces() <= 0:
        r.append(tuple(indices_sequence))
    else:
        for i in range(V.num_sub_spaces()):
            indices_sequence.append(i)
            _build_subindices(indices_sequence, r, V)
            indices_sequence.pop()


class EquationBCMixin(FloatingType):
    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            FloatingType.__init__(self,
                                  *args,
                                  block_class=EquationBCBlock,
                                  _ad_args=args,
                                  _ad_floating_active=True,
                                  **kwargs)
            init(self, *args, **kwargs)
            self._ad_F = self._F
            self._ad_u = self.u
            self._ad_bcs = self.bcs
            self._ad_J = self._J
            self._ad_kwargs = {'Jp': self._Jp, 'is_linear': self.is_linear}
        return wrapper

    @staticmethod
    def _ad_annotate_reconstruct(reconstruct):
        @wraps(reconstruct)
        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)
            if annotate:
                for arg in args:
                    if not hasattr(arg, "bcs"):
                        arg.bcs = []
                arg.bcs.append(self)
            with stop_annotating():
                ret = reconstruct(self, *args, **kwargs)
            import ipdb; ipdb.set_trace()
            return ret
        return wrapper

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        return deps[0]

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint
    
"""
class EquationBCMixin:
    @staticmethod
    def _ad_annotate_init(init):
        @no_annotations
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            init(self, *args, **kwargs)
            self._ad_F = self._F
            self._ad_u = self.u
            self._ad_bcs = self.bcs
            self._ad_J = self._J
            self._ad_kwargs = {'Jp': self._Jp, 'is_linear': self.is_linear}
            self._ad_count_map = {}
        return wrapper

    def _ad_count_map_update(self, updated_ad_count_map):
        self._ad_count_map = updated_ad_count_map
"""
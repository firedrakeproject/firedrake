from functools import wraps
from pyadjoint import Block
from pyadjoint.overloaded_type import FloatingType
from pyadjoint.tape import no_annotations, annotate_tape, stop_annotating

import firedrake.utils as utils
import ufl

class Backend:
    @utils.cached_property
    def backend(self):
        import firedrake
        return firedrake


class EquationBCBlock(Block, Backend):
    def __init__(self, *args, **kwargs):
        Block.__init__(self)
        self.args = args
        self.add_dependency(args[1])
        self.func = args[1]
        self.function_space = self.args[1].function_space()
        if 'bcs' in kwargs:
            for bc in kwargs['bcs']:
                self.add_dependency(bc, no_duplicates=True)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        raise NotImplementedError("Taking the derivative where EquationBC depends\
                                  on the control is not implemented")

    @no_annotations
    def recompute(self):
        # There is nothing to do. The checkpoint is weak,
        # so it changes automatically with the dependency checkpoint.

        return self

    def __str__(self):
        return "EquationBC block"


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
                                  _ad_kwargs=kwargs)
            init(self, *args, **kwargs)
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

            return ret
        return wrapper

    def _replace_map(self, form, checkpoint):
        replace_coeffs = {}
        coeff = checkpoint.output
        if coeff in form.coefficients():
            replace_coeffs[coeff] = checkpoint.saved_output
        return ufl.replace(form, replace_coeffs)

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        return deps

    def _ad_restore_at_checkpoint(self, checkpoint):
        from firedrake import DirichletBC, Constant, Function
        i_func = []
        i_bc = 0
        for i in range(len(checkpoint)):
              if isinstance(checkpoint[i].saved_output, (type(self), DirichletBC)):
                  i_bc = i
              if isinstance(checkpoint[i].saved_output, (Constant, Function)):
                  i_func.append(i)
            
        if self.is_linear:
            bc_rhs_tmp = self.eq.rhs
            for j in i_func:
                bc_rhs_tmp = self._replace_map(bc_rhs_tmp, checkpoint[j])
            bc_rhs = bc_rhs_tmp
        else:
            bc_rhs = self.eq.rhs

        bc_lhs_tmp = self.eq.lhs
        for j in i_func:
            bc_lhs_tmp = self._replace_map(bc_lhs_tmp, checkpoint[j])
        bc_lhs = bc_lhs_tmp

        #if i_bc != 0:
        #    return type(self)(bc_lhs == bc_rhs, checkpoint[0].saved_output, self.sub_domain, bcs = checkpoint[i_bc].saved_output)
        #else:
        #    return type(self)(bc_lhs == bc_rhs, checkpoint[0].saved_output, self.sub_domain)
        return self
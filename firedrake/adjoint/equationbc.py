from functools import wraps
#from dolfin_adjoint_common import blocks
from pyadjoint import Block
from pyadjoint.overloaded_type import FloatingType
from pyadjoint.tape import no_annotations, annotate_tape, stop_annotating

import firedrake.utils as utils

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
        if len(kwargs) > 0:
            for bc in kwargs['bcs']:
                self.add_dependency(bc, no_duplicates=True)
    # def _create_initial_guess(self):
    #     return self.backend.Function(self.function_space)

    # def _recover_bcs(self):
    #     bcs = []
    #     for block_variable in self.get_dependencies():
    #         c = block_variable.output
    #         c_rep = block_variable.saved_output
    #         if isinstance(c, self.backend.DirichletBC):
    #             bcs.append(c_rep)
    #     return bcs
    
    # def _replace_map(self, form):
    #     replace_coeffs = {}
    #     for block_variable in self.get_dependencies():
    #         coeff = block_variable.output
    #         if coeff in form.coefficients():
    #             replace_coeffs[coeff] = block_variable.saved_output
    #     return replace_coeffs

    # def _replace_form(self, form, func=None):
    #     """Replace the form coefficients with checkpointed values

    #     func represents the initial guess if relevant.
    #     """
    #     replace_map = self._replace_map(form)

    #     if func is not None and self.func in replace_map:
    #         self.backend.Function.assign(func, replace_map[self.func])
    #         replace_map[self.func] = func
    #     return ufl.replace(form, replace_map) 
    
    # def _replace_recompute_form(self):
    #     func = self._create_initial_guess()

    #     bcs = self._recover_bcs()
    #     lhs = self._replace_form(self.args[0].lhs, func=func)

    #     #rhs = 0
    #     #if self.linear:
    #     if self.args[0].rhs != 0:
    #         rhs = self._replace_form(self.args[0].rhs, func=func)
    #     else:
    #         rhs = self.args[0].rhs

    #     return lhs, rhs, func, bcs
        
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
                                  block_class= EquationBCBlock,
                                  _ad_args=args,
                                  _ad_floating_active=True,
                                  _ad_kwargs = kwargs)
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

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        return deps[0]

    def _ad_restore_at_checkpoint(self, checkpoint):
        return self
    
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
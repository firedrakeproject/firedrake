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

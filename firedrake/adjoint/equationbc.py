from functools import wraps
from pyadjoint.overloaded_type import FloatingType
from .blocks import DirichletBCBlock
from pyadjoint.tape import no_annotations, annotate_tape, stop_annotating


class EquationBCSplitMixin(FloatingType):
    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            FloatingType.__init__(self,
                                  *args,
                                  block_class=DirichletBCBlock,
                                  _ad_args=args,
                                  _ad_floating_active=True,
                                  **kwargs)
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
        if len(deps) <= 0:
            # We don't have any dependencies so the supplied value was not an OverloadedType.
            # Most probably it was just a float that is immutable so will never change.
            return None

        return deps[0]

    def _ad_restore_at_checkpoint(self, checkpoint):
        print('check')
        #if checkpoint is not None:
        #    self.set_value(checkpoint.saved_output)
        return self 
    
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
 
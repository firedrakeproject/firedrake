from functools import wraps
from pyadjoint.overloaded_type import FloatingType
from .blocks import DirichletBCBlock
from pyadjoint.tape import stop_annotating, annotate_tape


class DirichletBCMixin(FloatingType):
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
    def _ad_annotate_apply(apply):
        @wraps(apply)
        def wrapper(self, *args, **kwargs):
            annotate = annotate_tape(kwargs)
            if annotate:
                for arg in args:
                    if not hasattr(arg, "bcs"):
                        arg.bcs = []
                arg.bcs.append(self)
            with stop_annotating():
                ret = apply(self, *args, **kwargs)
            return ret
        return wrapper

    @staticmethod
    def _ad_annotate_function_arg(function_arg):
        @wraps(function_arg)
        def wrapper(self, g):
            ret = function_arg(self, g)
            # Keep the floating dependency in sync with the assigned value,
            # so that reusing this DirichletBC (e.g. calling set_value in a
            # time loop) tapes a fresh DirichletBCBlock against the current
            # g, rather than the one originally passed to __init__.
            self._ad_args = (self._ad_args[0], g, self._ad_args[2])
            return ret
        return wrapper

    def _ad_create_checkpoint(self):
        deps = self.block.get_dependencies()
        if len(deps) <= 0:
            # We don't have any dependencies so the supplied value was not an OverloadedType.
            # Most probably it was just a float that is immutable so will never change.
            return None

        return deps[0]

    def _ad_restore_at_checkpoint(self, bv):
        if bv is not None:
            bc = self.reconstruct(g=bv.saved_output)
            bc.block = self.block
            return bc
        return self

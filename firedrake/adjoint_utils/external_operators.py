from firedrake.adjoint_utils.function import FunctionMixin


class ExternalOperatorsMixin(FunctionMixin):

    @staticmethod
    def _ad_annotate_init(init):
        def wrapper(self, *args, **kwargs):
            FunctionMixin.__init__(self, *args, **kwargs)
            init(self, *args, **kwargs)
        return wrapper

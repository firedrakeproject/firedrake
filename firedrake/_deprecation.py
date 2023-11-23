class _fake_module:
    def __init__(self, new_location, functions):
        self.new_location = new_location
        for f in functions:
            setattr(self, f, self._import_error(self, f))

    def _import_error(self, fake_module, function_name):
        def __call__(*args, **kwargs):
            raise ImportError(
                f"The function `{function_name}` has moved to "
                f"`{fake_module.new_location}`, use\n"
                f"\t from {fake_module.new_location} import {function_name}"
            )
        return __call__


plot = _fake_module(
    "firedrake.pyplot",
    [
        "plot", "triplot", "tricontourf", "tricontour", "trisurf",
        "tripcolor", "quiver", "streamplot", "FunctionPlotter", "pgfplot"
    ]
)

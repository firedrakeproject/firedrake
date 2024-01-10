""" Tools for deprecating parts of Firedrake functionality
"""


class _fake_module:
    """ Object which behaves like a module

    Parameters
    ----------
    new_location:
        Where to find old functionality
    functions:
        List of functions that have moved

    """
    def __init__(self, new_location, functions):
        self.new_location = new_location
        for f in functions:
            setattr(self, f, self._import_error(f))

    def _import_error(self, function_name):
        def __call__(*args, **kwargs):
            raise ImportError(
                f"The function `{function_name}` has moved to "
                f"`{self.new_location}`, update your code to use\n\t"
                f"from {self.new_location} import {function_name}"
            )
        return __call__


# Deprecate plotting in the global namespace
plot = _fake_module(
    "firedrake.pyplot",
    [
        "plot", "triplot", "tricontourf", "tricontour", "trisurf",
        "tripcolor", "quiver", "streamplot", "FunctionPlotter", "pgfplot"
    ]
)

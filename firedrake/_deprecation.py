""" Tools for deprecating parts of Firedrake functionality
"""
import importlib

from warnings import warn


class _fake_module:
    """ Object which behaves like a module

    Parameters
    ----------
    new_location:
        Where to find old functionality
    functions:
        List of functions that have moved

    """
    def __init__(self, new_location, functions, new_functions=None):
        # Setup errors for old functions
        self.new_location = new_location
        if new_functions:
            for f, g in zip(functions, new_functions):
                setattr(self, f, self._import_error(f, g))
        else:
            for f in functions:
                setattr(self, f, self._import_error(f))

        # Add any existing functions to the correct namespace
        # This allows us to deprecate a file `output.py` and add its contents
        # to a new file `output/new.py`
        module_name = ".".join(new_location.split(".")[1:])
        try:
            module = importlib.import_module(f".{module_name}", "firedrake")
            for x in dir(module):
                setattr(self, x, getattr(module, x))
        except ImportError:
            pass

    def _import_error(self, function_name, new_function_name=None):
        def __call__(*args, **kwargs):
            raise ImportError(
                f"The function `{function_name}` has moved to "
                f"`{self.new_location}`, update your code to use\n\t"
                f"from {self.new_location} import {new_function_name or function_name}"
            )
        return __call__


# Deprecate output.File in the global namespace
output = _fake_module(
    "firedrake.output",
    ["File", ],
    ["VTKFile", ]
)


# I hate it
def File(*args, **kwargs):
    """Deprecated File constructor.

    Use `VTKFile` from `firedrake.output` instead
    """
    from .output import VTKFile
    warn(
        "The use of `File` for output is deprecated, please update your "
        "code to use `VTKFile` from `firedrake.output`."
    )
    return VTKFile(*args, **kwargs)


# Deprecate plotting in the global namespace
plot = _fake_module(
    "firedrake.pyplot",
    [
        "plot", "triplot", "tricontourf", "tricontour", "trisurf",
        "tripcolor", "quiver", "streamplot", "FunctionPlotter", "pgfplot"
    ]
)

from __future__ import annotations

import collections
import dataclasses
import os
import pathlib
import tempfile
from typing import Any, Callable, Self
import warnings

from immutabledict import immutabledict as idict


def paramclass(cls: type) -> type:
    """Decorator that turns a class into a dataclass for storing parameters."""
    return dataclasses.dataclass(kw_only=True, unsafe_hash=True)(cls)


_default_cache_dir = pathlib.Path(tempfile.gettempdir()) / f"pyop3-cache-uid{os.getuid()}"

# TODO: should live elsewhere
_nothing = object()
"""Sentinel value indicating nothing should be done.

This is useful in cases where `None` holds some meaning.

"""


@dataclasses.dataclass(frozen=True)
class ConfigOption:
    type_: Any
    default_value: Any
    description: str
    # kw only below
    from_str: Callable = lambda x: x
    default_debug_value: Any = _nothing
    value_getter: Callable | None = None
    value_setter: Callable | None = None


def _log_level_setter(self, value: Any, /) -> None:
    from pyop3.log import LOGGER

    LOGGER.setLevel(value)
    self._log_level = value


class Pyop3Configuration:
    """Global configuration options for pyop3."""

    OPTIONS: idict[str, ConfigOption] = idict({

        # {{{ code generation options

        "max_static_array_size": ConfigOption(
            int,
            128,
            """The maximum size of hard-coded constant arrays.

            Constant arrays that exceed this limit are passed to the kernel as
            arguments instead.

            """,
            from_str=lambda x: int(x),
        ),

        # }}}

        # {{{ compilation options

        "extra_cflags": ConfigOption(
            tuple[str, ...],
            (),
            """Extra flags to be passed to the C compiler.""",
            from_str=lambda x: tuple(x.split(" ")),
        ),

        "extra_cxxflags": ConfigOption(
            tuple[str, ...],
            (),
            """Extra flags to be passed to the C++ compiler.""",
            from_str=lambda x: tuple(x.split(" ")),
        ),

        "extra_ldflags": ConfigOption(
            tuple[str, ...],
            (),
            """Extra flags to be passed to the linker.""",
            from_str=lambda x: tuple(x.split(" ")),
        ),

        "cache_dir": ConfigOption(
            pathlib.Path,
            _default_cache_dir,
            """Location of the generated code (libraries).""",
        ),

        "node_local_compilation": ConfigOption(
            bool,
            True,
            """Compile generated code separately on each node.

            If set it is likely that ``cache_dir`` will have to be set to to a
            node-local filesystem too.

            """,
            from_str=lambda x: bool(x),
        ),

        # }}}

        # {{{ logging options

        "log_level": ConfigOption(
            str | int,
            "WARNING",
            """Level used by the pyop3 logger.""",
            default_debug_value="DEBUG",
            value_setter=_log_level_setter,
        ),

        "print_cache_stats": ConfigOption(
            bool,
            False,
            """Print cache statistics at the end of the program.""",
            default_debug_value=True,
            from_str=lambda x: bool(x),
        ),

        # }}}

        # {{{ debugging options

        "debug_checks": ConfigOption(
            bool,
            False,
            """Enable additional correctness checks.

            This option is enabled in debug mode.

            """,
            default_debug_value=True,
            from_str=lambda x: bool(x),
        ),

        "compiler_use_debug_flags": ConfigOption(
            bool,
            False,
            """Pass debugging options (i.e. '-O0' and '-g') to the compiler.

            This option is enabled in debug mode.

            """,
            default_debug_value=True,
            from_str=lambda x: bool(x),
        ),

        "check_src_hashes": ConfigOption(
            bool,
            True,
            """Check that generated code is the same on all processes.""",
            from_str=lambda x: bool(x),
        ),

        "spmd_strict": ConfigOption(
            bool,
            False,
            """Turn on additional parallel correctness checks.

            Setting this option will enable barriers for calls marked with @collective
            and for cache accesses. This adds considerable overhead, but is useful for
            tracking down deadlocks.

            This option is enabled in debug mode.

            """,
            default_debug_value=True,
            from_str=lambda x: bool(x),
        )

        # }}}

    })

    def __init__(self, **kwargs) -> None:
        for option_name, option_type in self.OPTIONS.items():
            assert option_name in kwargs
            setattr(self, option_name, kwargs.pop(option_name))
        assert not kwargs

    def __str__(self) -> str:
        return str(self.as_dict())

    def as_dict(self) -> dict:
        return {option_name: getattr(self, option_name) for option_name in self.OPTIONS}


# programatically add getters and setters for configuration options
def _make_getter(name):
    def _getter(self):
        return getattr(self, f"_{name}")
    return _getter


def _make_setter(name):
    def _setter(self, value):
        setattr(self, f"_{name}", value)
    return _setter


# TODO: Use 'type_' to set annotations for the getter
for option_name, option in Pyop3Configuration.OPTIONS.items():
    option_property = property(
        option.value_getter or _make_getter(option_name),
        option.value_setter or _make_setter(option_name),
        doc=option.description,
    )
    setattr(Pyop3Configuration, option_name, option_property)


# TODO: Included for the PyOP2->pyop3 migration, remove in a later release
_REMOVED_PYOP2_OPTIONS = (
    "PYOP2_COMPUTE_KERNEL_FLOPS",
    "PYOP2_SIMD_WIDTH",
    "PYOP2_TYPE_CHECK",
    "PYOP2_NO_FORK_AVAILABLE",
    "PYOP2_CACHE_INFO",
    "PYOP2_MATNEST",
    "PYOP2_BLOCK_SPARSITY",
)


def _prepare_configuration() -> Pyop3Configuration:
    """Create a configuration object from environment variables.

    This factory method handles the conversion of any non-string types.

    """
    for removed_option in _REMOVED_PYOP2_OPTIONS:
        if removed_option in os.environ:
            warnings.warn(
                f"{removed_option} detected in your environment but is no "
                "longer supported. This option will be ignored."
            )

    # Gather environment variables
    env_options = {}
    for option_name in Pyop3Configuration.OPTIONS.keys():
        if (env_key := f"PYOP3_{option_name.upper()}") in os.environ:
            env_options[option_name] = os.environ[env_key]
        elif (env_key := f"PYOP2_{option_name.upper()}") in os.environ:
            warnings.warn(
                f"{env_key} is deprecated, please use 'PYOP3_{option_name.upper()}' instead.",
                FutureWarning,
            )
            env_options[option_name] = os.environ[env_key]
    debug_mode = bool(os.environ.get("PYOP3_DEBUG", 0))

    # Now parse them
    parsed_options = {}
    for option_name, option_spec in Pyop3Configuration.OPTIONS.items():
        if option_name in env_options:
            option = option_spec.from_str(env_options.pop(option_name))
        elif debug_mode:
            if option_spec.default_debug_value is not _nothing:
                option = option_spec.default_debug_value
            else:
                option = option_spec.default_value
        else:
            option = option_spec.default_value
        parsed_options[option_name] = option
    assert not env_options
    return Pyop3Configuration(**parsed_options)


config = _prepare_configuration()

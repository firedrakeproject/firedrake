from __future__ import annotations

import collections
import dataclasses
import os
import pathlib
import tempfile
import warnings


def paramclass(cls: type) -> type:
    """Decorator that turns a class into a dataclass for storing parameters."""
    return dataclasses.dataclass(kw_only=True)(cls)


_default_cache_dir = pathlib.Path(tempfile.gettempdir()) / f"pyop3-cache-uid{os.getuid()}"


@paramclass
class Pyop3Configuration:
    """Global configuration options for pyop3."""

    # {{{ debugging options

    debug: bool = False
    """Enable debug mode."""

    check_src_hashes: bool = True
    """Check that generated code is the same on all processes.

    This option is always enabled in debug mode.

    """

    spmd_strict: bool = False
    """Turn on additional parallel correctness checks.

    Setting this option will enable barriers for calls marked with @collective
    and for cache accesses. This adds considerable overhead, but is useful for
    tracking down deadlocks.

    """

    # }}}

    # {{{ code generation options

    max_static_array_size: int = 128
    """The maximum size of hard-coded constant arrays.

    Constant arrays that exceed this limit are passed to the kernel as
    arguments instead.

    """

    # }}}

    # {{{ compilation options

    cflags: tuple[str, ...] | None = None
    """Extra flags to be passed to the C compiler."""

    cxxflags: tuple[str, ...] | None = None
    """Extra flags to be passed to the C++ compiler."""

    ldflags: tuple[str, ...] | None = None
    """Extra flags to be passed to the linker."""

    cache_dir: pathlib.Path = _default_cache_dir
    """Location of the generated code (libraries)."""

    node_local_compilation: bool = True
    """Compile generated code separately on each node.

    If set it is likely that ``cache_dir`` will have to be set to to a
    node-local filesystem too.

    """

    # }}}

    # {{{ other options

    log_level: str | int = "WARNING"
    """The logging level of the pyop3 logger."""

    # }}}

    @classmethod
    def _from_env(cls, env_options: collections.abc.Mapping) -> Pyop3Configuration:
        """Create a configuration object from environment variables.

        This factory method handles the conversion of any non-string types.

        """
        env_options = dict(env_options)
        parsed_options = {}

        # debugging options
        if (key := "debug") in env_options:
            parsed_options[key] = bool(env_options.pop(key))
        if (key := "check_src_hashes") in env_options:
            parsed_options[key] = bool(env_options.pop(key))
        if (key := "spmd_strict") in env_options:
            parsed_options[key] = bool(env_options.pop(key))

        # code generation options
        if (key := "max_static_array_size") in env_options:
            parsed_options[key] = int(env_options.pop(key))

        # compilation options
        for key in ["cflags", "cxxflags", "ldflags"]:  # tuple[str, ...]
            if key in env_options:
                parsed_options[key] = tuple(env_options.pop(key).split(" "))
        if (key := "cache_dir") in env_options:
            parsed_options[key] = pathlib.Path(env_options.pop(key))
        if (key := "node_local_compilation") in env_options:
            parsed_options[key] = bool(env_options.pop(key))

        # other options
        if (key := "log_level") in env_options:  # str | int
            env_option = env_options.pop(key)
            try:
                parsed_options[key] = int(env_option)
            except ValueError:  # e.g. 'WARNING'
                parsed_options[key] = env_option

        assert not env_options
        return cls(**parsed_options)


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
    for removed_option in _REMOVED_PYOP2_OPTIONS:
        if removed_option in os.environ:
            warnings.warn(
                f"{removed_option} detected in your environment but is no "
                "longer supported. This option will be ignored."
            )

    env_options = {}
    for field_name in Pyop3Configuration.__dataclass_fields__.keys():
        if (env_key := f"PYOP3_{field_name.upper()}") in os.environ:
            env_options[field_name] = os.environ[env_key]
        elif (env_key := f"PYOP2_{field_name.upper()}") in os.environ:
            warnings.warn(
                f"{env_key} is deprecated, please use 'PYOP3_{field_name}' instead.",
                FutureWarning,
            )
            env_options[field_name] = os.environ[env_key]
    return Pyop3Configuration._from_env(env_options)


CONFIG = _prepare_configuration()

import abc
from dataclasses import dataclass
import hashlib
from typing import Union

import loopy as lp
from loopy.kernel import LoopKernel
from loopy.translation_unit import TranslationUnit
from loopy.tools import LoopyKeyBuilder
import numpy as np

from pyop2 import version
from pyop2.configuration import configuration
from pyop2.datatypes import ScalarType
from pyop2.exceptions import NameTypeError
from pyop2.types import Access
from pyop2.utils import cached_property, validate_type


@dataclass(frozen=True)
class LocalKernelArg:
    """Class representing a kernel argument.

    :param access: Access descriptor for the argument.
    :param dtype: The argument's datatype.
    """

    access: Access
    dtype: Union[np.dtype, str]


@validate_type(("name", str, NameTypeError))
def Kernel(code, name, **kwargs):
    """Construct a local kernel.

    For a description of the arguments to this function please see :class:`LocalKernel`.
    """
    if isinstance(code, str):
        return CStringLocalKernel(code, name, **kwargs)
    elif isinstance(code, (lp.LoopKernel, lp.TranslationUnit)):
        return LoopyLocalKernel(code, name, **kwargs)
    else:
        raise TypeError("code argument is the wrong type")


class LocalKernel(abc.ABC):
    """Class representing the kernel executed per member of the iterset.

    :arg code: Function definition (including signature).
    :arg name: The kernel name. This must match the name of the kernel
        function given in `code`.
    :arg accesses: Optional iterable of :class:`Access` instances describing
        how each argument in the function definition is accessed.

    :kwarg cpp: Is the kernel actually C++ rather than C?  If yes,
        then compile with the C++ compiler (kernel is wrapped in
        extern C for linkage reasons).
    :kwarg flop_count: The number of FLOPs performed by the kernel.
    :kwarg headers: list of system headers to include when compiling the kernel
        in the form ``#include <header.h>`` (optional, defaults to empty)
    :kwarg include_dirs: list of additional include directories to be searched
        when compiling the kernel (optional, defaults to empty)
    :kwarg ldargs: A list of arguments to pass to the linker when
        compiling this Kernel.
    :kwarg opts: An options dictionary for declaring optimisations to apply.
    :kwarg requires_zeroed_output_arguments: Does this kernel require the
        output arguments to be zeroed on entry when called? (default no)
    :kwarg user_code: code snippet to be executed once at the very start of
        the generated kernel wrapper code (optional, defaults to
        empty)
    :kwarg events: Tuple of log event names which are called in the C code of the local kernels

    Consider the case of initialising a :class:`~pyop2.Dat` with seeded random
    values in the interval 0 to 1. The corresponding :class:`~pyop2.Kernel` is
    constructed as follows: ::

      op2.CStringKernel("void setrand(double *x) { x[0] = (double)random()/RAND_MAX); }",
                        name="setrand",
                        headers=["#include <stdlib.h>"], user_code="srandom(10001);")

    .. note::
        When running in parallel with MPI the generated code must be the same
        on all ranks.
    """

    @validate_type(("name", str, NameTypeError))
    def __init__(self, code, name, accesses=None, *,
                 cpp=False,
                 flop_count=None,
                 headers=(),
                 include_dirs=(),
                 ldargs=(),
                 opts=None,
                 requires_zeroed_output_arguments=False,
                 user_code="",
                 events=()):
        self.code = code
        self.name = name
        self.accesses = accesses
        self.cpp = cpp
        self.flop_count = flop_count
        self.headers = headers
        self.include_dirs = include_dirs
        self.ldargs = ldargs
        self.opts = opts or {}
        self.requires_zeroed_output_arguments = requires_zeroed_output_arguments
        self.user_code = user_code
        self.events = events

    @property
    @abc.abstractmethod
    def dtypes(self):
        """Return the dtypes of the arguments to the kernel."""

    @property
    def cache_key(self):
        return self._immutable_cache_key, self.accesses, self.dtypes

    @cached_property
    def _immutable_cache_key(self):
        # We need this function because self.accesses is mutable due to legacy support
        if isinstance(self.code, lp.TranslationUnit):
            code_key = LoopyKeyBuilder()(self.code)
        else:
            code_key = self.code

        key = (code_key, self.name, self.cpp, self.flop_count,
               self.headers, self.include_dirs, self.ldargs, sorted(self.opts.items()),
               self.requires_zeroed_output_arguments, self.user_code, version.__version__)
        return hashlib.md5(str(key).encode()).hexdigest()

    @property
    def _wrapper_cache_key_(self):
        import warnings
        warnings.warn("_wrapper_cache_key is deprecated, use cache_key instead", DeprecationWarning)

        return self.cache_key

    @property
    def arguments(self):
        """Return an iterable of :class:`LocalKernelArg` instances representing
        the arguments expected by the kernel.
        """
        assert len(self.accesses) == len(self.dtypes)

        return tuple(LocalKernelArg(acc, dtype)
                     for acc, dtype in zip(self.accesses, self.dtypes))

    @cached_property
    def num_flops(self):
        """Compute the numbers of FLOPs if not already known."""
        if self.flop_count is not None:
            return self.flop_count

        if not configuration["compute_kernel_flops"]:
            return 0

        if isinstance(self.code, lp.TranslationUnit):
            op_map = lp.get_op_map(
                self.code.copy(options=lp.Options(ignore_boostable_into=True),
                               silenced_warnings=['insn_count_subgroups_upper_bound',
                                                  'get_x_map_guessing_subgroup_size',
                                                  'summing_if_branches_ops']),
                subgroup_size='guess')
            return op_map.filter_by(name=['add', 'sub', 'mul', 'div'],
                                    dtype=[ScalarType]).eval_and_sum({})
        else:
            return 0

    def __eq__(self, other):
        if not isinstance(other, LocalKernel):
            return NotImplemented
        else:
            return self.cache_key == other.cache_key

    def __hash__(self):
        return hash(self.cache_key)

    def __str__(self):
        return f"OP2 Kernel: {self.name}"

    def __repr__(self):
        return 'Kernel("""%s""", %r)' % (self.code, self.name)


class CStringLocalKernel(LocalKernel):
    """:class:`LocalKernel` class where `code` is a string of C code.

    :kwarg dtypes: Iterable of datatypes (either `np.dtype` or `str`) for
        each kernel argument. This is not required for :class:`LoopyLocalKernel`
        because it can be inferred.

    All other `__init__` parameters are the same.
    """

    @validate_type(("code", str, TypeError))
    def __init__(self, code, name, accesses=None, dtypes=None, **kwargs):
        super().__init__(code, name, accesses, **kwargs)
        self._dtypes = dtypes

    @property
    def dtypes(self):
        return self._dtypes

    @dtypes.setter
    def dtypes(self, dtypes):
        self._dtypes = dtypes


class LoopyLocalKernel(LocalKernel):
    """:class:`LocalKernel` class where `code` has type :class:`loopy.LoopKernel`
        or :class:`loopy.TranslationUnit`.
    """

    @validate_type(("code", (LoopKernel, TranslationUnit), TypeError))
    def __init__(self, code, *args, **kwargs):
        super().__init__(code, *args, **kwargs)

    @property
    def dtypes(self):
        return tuple(a.dtype for a in self._loopy_arguments)

    @cached_property
    def _loopy_arguments(self):
        """Return the loopy arguments associated with the kernel."""
        return tuple(a for a in self.code.callables_table[self.name].subkernel.args
                     if isinstance(a, lp.ArrayArg))

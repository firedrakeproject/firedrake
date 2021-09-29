import hashlib

import coffee
import loopy as lp

from . import caching, configuration as conf, datatypes, exceptions as ex, utils, version


class Kernel(caching.Cached):

    """OP2 kernel type.

    :param code: kernel function definition, including signature; either a
        string or an AST :class:`.Node`
    :param name: kernel function name; must match the name of the kernel
        function given in `code`
    :param opts: options dictionary for :doc:`PyOP2 IR optimisations <ir>`
        (optional, ignored if `code` is a string)
    :param include_dirs: list of additional include directories to be searched
        when compiling the kernel (optional, defaults to empty)
    :param headers: list of system headers to include when compiling the kernel
        in the form ``#include <header.h>`` (optional, defaults to empty)
    :param user_code: code snippet to be executed once at the very start of
        the generated kernel wrapper code (optional, defaults to
        empty)
    :param ldargs: A list of arguments to pass to the linker when
        compiling this Kernel.
    :param requires_zeroed_output_arguments: Does this kernel require the
        output arguments to be zeroed on entry when called? (default no)
    :param cpp: Is the kernel actually C++ rather than C?  If yes,
        then compile with the C++ compiler (kernel is wrapped in
        extern C for linkage reasons).

    Consider the case of initialising a :class:`~pyop2.Dat` with seeded random
    values in the interval 0 to 1. The corresponding :class:`~pyop2.Kernel` is
    constructed as follows: ::

      op2.Kernel("void setrand(double *x) { x[0] = (double)random()/RAND_MAX); }",
                 name="setrand",
                 headers=["#include <stdlib.h>"], user_code="srandom(10001);")

    .. note::
        When running in parallel with MPI the generated code must be the same
        on all ranks.
    """

    _cache = {}

    @classmethod
    @utils.validate_type(('name', str, ex.NameTypeError))
    def _cache_key(cls, code, name, opts={}, include_dirs=[], headers=[],
                   user_code="", ldargs=None, cpp=False, requires_zeroed_output_arguments=False,
                   flop_count=None):
        # Both code and name are relevant since there might be multiple kernels
        # extracting different functions from the same code
        # Also include the PyOP2 version, since the Kernel class might change

        if isinstance(code, coffee.base.Node):
            code = code.gencode()
        if isinstance(code, lp.TranslationUnit):
            from loopy.tools import LoopyKeyBuilder
            from hashlib import sha256
            key_hash = sha256()
            code.update_persistent_hash(key_hash, LoopyKeyBuilder())
            code = key_hash.hexdigest()
        hashee = (str(code) + name + str(sorted(opts.items())) + str(include_dirs)
                  + str(headers) + version.__version__ + str(ldargs) + str(cpp) + str(requires_zeroed_output_arguments))
        return hashlib.md5(hashee.encode()).hexdigest()

    @utils.cached_property
    def _wrapper_cache_key_(self):
        return (self._key, )

    def __init__(self, code, name, opts={}, include_dirs=[], headers=[],
                 user_code="", ldargs=None, cpp=False, requires_zeroed_output_arguments=False,
                 flop_count=None):
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return
        self._name = name
        self._cpp = cpp
        # Record used optimisations
        self._opts = opts
        self._include_dirs = include_dirs
        self._ldargs = ldargs if ldargs is not None else []
        self._headers = headers
        self._user_code = user_code
        assert isinstance(code, (str, coffee.base.Node, lp.Program, lp.LoopKernel, lp.TranslationUnit))
        self._code = code
        self._initialized = True
        self.requires_zeroed_output_arguments = requires_zeroed_output_arguments
        self.flop_count = flop_count

    @property
    def name(self):
        """Kernel name, must match the kernel function name in the code."""
        return self._name

    @property
    def code(self):
        return self._code

    @utils.cached_property
    def num_flops(self):
        if self.flop_count is not None:
            return self.flop_count
        if not conf.configuration["compute_kernel_flops"]:
            return 0
        if isinstance(self.code, coffee.base.Node):
            v = coffee.visitors.EstimateFlops()
            return v.visit(self.code)
        elif isinstance(self.code, lp.TranslationUnit):
            op_map = lp.get_op_map(
                self.code.copy(options=lp.Options(ignore_boostable_into=True),
                               silenced_warnings=['insn_count_subgroups_upper_bound',
                                                  'get_x_map_guessing_subgroup_size',
                                                  'summing_if_branches_ops']),
                subgroup_size='guess')
            return op_map.filter_by(name=['add', 'sub', 'mul', 'div'], dtype=[datatypes.ScalarType]).eval_and_sum({})
        else:
            return 0

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", %r)' % (self._code, self._name)

    def __eq__(self, other):
        return self.cache_key == other.cache_key

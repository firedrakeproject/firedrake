# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""OP2 backend for fusion and tiling of parloops."""

from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy as dcopy, copy as scopy
import os

from base import *
import base
import compilation
import sequential
from backends import _make_object
from caching import Cached
from profiling import timed_region
from logger import warning, info as log_info
from mpi import MPI, collective
from configuration import configuration
from utils import flatten, strip, as_tuple

import coffee
from coffee import base as ast
from coffee.utils import ast_make_alias
from coffee.visitors import FindInstances, SymbolReferences

try:
    import slope_python as slope
except ImportError:
    slope = None


class Arg(sequential.Arg):

    @staticmethod
    def specialize(args, gtl_map, loop_id):
        """Given an iterator of :class:`sequential.Arg` objects return an iterator
        of :class:`fusion.Arg` objects.

        :arg args: either a single :class:`sequential.Arg` object or an iterator
             (accepted: list, tuple) of :class:`sequential.Arg` objects.
        :arg gtl_map: a dict associating global map names to local map names.
        :arg loop_id: the position of the loop using ``args`` in the loop chain
        """

        def convert(arg, gtl_map, loop_id):
            # Retrive local maps
            maps = as_tuple(arg.map, Map)
            c_local_maps = [None]*len(maps)
            for i, map in enumerate(maps):
                c_local_maps[i] = [None]*len(map)
                for j, m in enumerate(map):
                    c_local_maps[i][j] = gtl_map["%s%d_%d" % (m.name, i, j)]
            # Instantiate and initialize new, specialized Arg
            _arg = Arg(arg.data, arg.map, arg.idx, arg.access, arg._flatten)
            _arg.loop_position = loop_id
            _arg.position = arg.position
            _arg.indirect_position = arg.indirect_position
            _arg._c_local_maps = c_local_maps
            return _arg

        try:
            return [convert(arg, gtl_map, loop_id) for arg in args]
        except TypeError:
            return convert(args, gtl_map, loop_id)

    @staticmethod
    def filter_args(loop_args):
        """Given an iterator of :class:`Arg` tuples, each tuple representing the
        args in a loop of the chain, create a 'flattened' iterator of ``Args``
        in which: 1) there are no duplicates; 2) access modes are 'adjusted'
        if the same :class:`Dat` is accessed through multiple ``Args``.

        For example, if a ``Dat`` appears twice with access modes ``WRITE`` and
        ``READ``, a single ``Arg`` with access mode ``RW`` will be present in the
        returned iterator."""
        filtered_args = OrderedDict()
        for args in loop_args:
            for a in args:
                fa = filtered_args.setdefault(a.data, a)
                if a.access != fa.access:
                    if READ in [a.access, fa.access]:
                        # If a READ and some sort of write (MIN, MAX, RW, WRITE,
                        # INC), then the access mode becomes RW
                        fa._access = RW
                    elif WRITE in [a.access, fa.access]:
                        # Can't be a READ, so just stick to WRITE regardless of what
                        # the other access mode is
                        fa._access = WRITE
                    else:
                        # Neither READ nor WRITE, so access modes are some
                        # combinations of RW, INC, MIN, MAX. For simplicity,
                        # just make it RW.
                        fa._access = RW
        return filtered_args

    def c_arg_bindto(self, arg):
        """Assign this Arg's c_pointer to ``arg``."""
        if self.ctype != arg.ctype:
            raise RuntimeError("Cannot bind arguments having mismatching types")
        return "%s* %s = %s" % (self.ctype, self.c_arg_name(), arg.c_arg_name())

    def c_map_name(self, i, j):
        return self._c_local_maps[i][j]

    @property
    def name(self):
        """The generated argument name."""
        return "arg_exec_loop%d_%d" % (self.loop_position, self.position)


class Kernel(sequential.Kernel, tuple):

    """A :class:`fusion.Kernel` represents a sequence of kernels.

    The sequence can be:

        * the result of the concatenation of kernel bodies (so a single C function
            is present)
        * a list of separate kernels (multiple C functions, which have to be
            suitably called by the wrapper)."""

    @classmethod
    def _cache_key(cls, kernels, fused_ast=None, loop_chain_index=None):
        keys = "".join([super(Kernel, cls)._cache_key(
            k._original_ast.gencode() if k._original_ast else k._code,
            k._name, k._opts, k._include_dirs, k._headers, k._user_code) for k in kernels])
        return str(loop_chain_index) + keys

    def _ast_to_c(self, asts, opts):
        """Produce a string of C code from an abstract syntax tree representation
        of the kernel."""
        if not isinstance(asts, (ast.FunDecl, ast.Root)):
            asts = ast.Root(asts)
        self._ast = asts
        return super(Kernel, self)._ast_to_c(self._ast, opts)

    def __init__(self, kernels, fused_ast=None, loop_chain_index=None):
        """Initialize a :class:`fusion.Kernel` object.

        :arg kernels: an iterator of some :class:`Kernel` objects. The objects
            can be of class `fusion.Kernel` or of any superclass.
        :arg fused_ast: the abstract syntax tree of the fused kernel. If not
            provided, objects in ``kernels`` are considered "isolated C functions".
        :arg loop_chain_index: index (i.e., position) of the kernel in a loop chain.
            Meaningful only if ``fused_ast`` is specified.
        """
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return

        Kernel._globalcount += 1

        # What sort of Kernel do I have?
        if fused_ast:
            # A single, already fused AST (code generation is then delayed)
            self._ast = fused_ast
            self._code = None
        else:
            # Multiple kernels that should be interpreted as different C functions,
            # in which case duplicates are discarded
            self._ast = None
            kernels = OrderedDict(zip([k.cache_key[1:] for k in kernels], kernels)).values()
            self._code = "\n".join([super(Kernel, k)._ast_to_c(dcopy(k._original_ast), k._opts)
                                    if k._original_ast else k._code for k in kernels])
        self._original_ast = self._ast

        kernels = as_tuple(kernels, (Kernel, sequential.Kernel, base.Kernel))
        self._kernels = kernels
        self._name = "_".join([k.name for k in kernels])
        self._opts = dict(flatten([k._opts.items() for k in kernels]))
        self._applied_blas = any(k._applied_blas for k in kernels)
        self._include_dirs = list(set(flatten([k._include_dirs for k in kernels])))
        self._headers = list(set(flatten([k._headers for k in kernels])))
        self._user_code = "\n".join(list(set([k._user_code for k in kernels])))
        self._attached_info = False

        self._initialized = True

    def __iter__(self):
        for k in self._kernels:
            yield k

    def __str__(self):
        return "OP2 FusionKernel: %s" % self._name


# Parallel loop API

class IterationSpace(base.IterationSpace):

    """A simple bag of :class:`IterationSpace` objects."""

    def __init__(self, all_itspaces):
        self._iterset = [i._iterset for i in all_itspaces]

    def __str__(self):
        output = "OP2 Fused Iteration Space:"
        output += "\n  ".join(["%s with extents %s" % (i._iterset, i._extents)
                               for i in self.iterset])
        return output

    def __repr__(self):
        return "\n".join(["IterationSpace(%r, %r)" % (i._iterset, i._extents)
                          for i in self.iterset])


class JITModule(sequential.JITModule):

    _cppargs = []
    _libraries = []
    _extension = 'cpp'

    _wrapper = """
extern "C" void %(wrapper_name)s(%(executor_arg)s,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(region_flag)s);
void %(wrapper_name)s(%(executor_arg)s,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(region_flag)s) {
  %(user_code)s
  %(wrapper_decs)s;
  %(const_inits)s;

  %(executor_code)s;
}
"""
    _kernel_wrapper = """
%(interm_globals_decl)s;
%(interm_globals_init)s;
%(vec_decs)s;
%(args_binding)s;
%(tile_init)s;
for (int n = %(tile_start)s; n < %(tile_end)s; n++) {
  int i = %(index_expr)s;
  %(vec_inits)s;
  i = %(tile_iter)s;
  %(buffer_decl)s;
  %(buffer_gather)s
  %(kernel_name)s(%(kernel_args)s);
  i = %(index_expr)s;
  %(itset_loop_body)s;
}
%(interm_globals_writeback)s;
"""

    @classmethod
    def _cache_key(cls, kernel, itspace, *args, **kwargs):
        key = (hash(kwargs['executor']),)
        all_kernels = kwargs['all_kernels']
        all_itspaces = kwargs['all_itspaces']
        all_args = kwargs['all_args']
        for kernel, itspace, args in zip(all_kernels, all_itspaces, all_args):
            key += super(JITModule, cls)._cache_key(kernel, itspace, *args)
        return key

    def __init__(self, kernel, itspace, *args, **kwargs):
        if self._initialized:
            return
        self._all_kernels = kwargs.pop('all_kernels')
        self._all_itspaces = kwargs.pop('all_itspaces')
        self._all_args = kwargs.pop('all_args')
        self._executor = kwargs.pop('executor')
        super(JITModule, self).__init__(kernel, itspace, *args, **kwargs)

    def set_argtypes(self, iterset, *args):
        argtypes = [slope.Executor.meta['py_ctype_exec']]
        for itspace in self._all_itspaces:
            if isinstance(itspace.iterset, Subset):
                argtypes.append(itspace.iterset._argtype)
        for arg in args:
            if arg._is_mat:
                argtypes.append(arg.data._argtype)
            else:
                for d in arg.data:
                    argtypes.append(d._argtype)
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        argtypes.append(m._argtype)
        for c in Const._definitions():
            argtypes.append(c._argtype)
        # For the MPI region flag
        argtypes.append(ctypes.c_int)

        self._argtypes = argtypes

    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule not in cache, but has no args associated")

        # Set compiler and linker options
        slope_dir = os.environ['SLOPE_DIR']
        self._kernel._name = 'executor'
        self._kernel._headers.extend(slope.Executor.meta['headers'])
        self._kernel._include_dirs.extend(['%s/%s' % (slope_dir,
                                                      slope.get_include_dir())])
        self._libraries += ['-L%s/%s' % (slope_dir, slope.get_lib_dir()),
                            '-l%s' % slope.get_lib_name()]
        compiler = coffee.plan.compiler.get('name')
        self._cppargs += slope.get_compile_opts(compiler)
        fun = super(JITModule, self).compile()

        if hasattr(self, '_all_args'):
            # After the JITModule is compiled, can drop any reference to now
            # useless fields
            del self._all_kernels
            del self._all_itspaces
            del self._all_args
            del self._executor

        return fun

    def generate_code(self):
        indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))

        args_dict = dict(zip([a.data for a in self._args], self._args))

        # 1) Construct the wrapper arguments
        code_dict = {}
        code_dict['wrapper_name'] = 'wrap_executor'
        code_dict['executor_arg'] = "%s %s" % (slope.Executor.meta['ctype_exec'],
                                               slope.Executor.meta['name_param_exec'])
        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self._args])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self._args])
        code_dict['wrapper_args'] = _wrapper_args
        code_dict['wrapper_decs'] = indent(_wrapper_decs, 1)
        code_dict['region_flag'] = ", %s %s" % (slope.Executor.meta['ctype_region_flag'],
                                                slope.Executor.meta['region_flag'])

        # 2) Construct the kernel invocations
        _loop_body, _user_code, _ssinds_arg = [], [], []
        _const_args, _const_inits = set(), set()
        # For each kernel ...
        for i, (kernel, it_space, args) in enumerate(zip(self._all_kernels,
                                                         self._all_itspaces,
                                                         self._all_args)):
            # ... bind the Executor's arguments to this kernel's arguments
            binding = OrderedDict(zip(args, [args_dict[a.data] for a in args]))
            if len(binding) != len(args):
                raise RuntimeError("Tiling code gen failed due to args mismatching")
            binding = ';\n'.join([a0.c_arg_bindto(a1) for a0, a1 in binding.items()])

            # ... obtain the /code_dict/ as if it were not part of an Executor,
            # since bits of code generation can be reused
            loop_code_dict = sequential.JITModule(kernel, it_space, *args, delay=True)
            loop_code_dict = loop_code_dict.generate_code()

            # ... build the subset indirection array, if necessary
            _ssind_arg, _ssind_decl = '', ''
            if loop_code_dict['ssinds_arg']:
                _ssind_arg = 'ssinds_%d' % i
                _ssind_decl = 'int* %s' % _ssind_arg
                loop_code_dict['index_expr'] = '%s[n]' % _ssind_arg

            # ... finish building up the /code_dict/
            loop_code_dict['args_binding'] = binding
            loop_code_dict['tile_init'] = self._executor.c_loop_init[i]
            loop_code_dict['tile_start'] = slope.Executor.meta['tile_start']
            loop_code_dict['tile_end'] = slope.Executor.meta['tile_end']
            loop_code_dict['tile_iter'] = '%s[n]' % self._executor.gtl_maps[i]['DIRECT']
            if _ssind_arg:
                loop_code_dict['tile_iter'] = '%s[%s]' % (_ssind_arg, loop_code_dict['tile_iter'])

            # ... concatenate the rest, i.e., body, user code, constants, ...
            _loop_body.append(strip(JITModule._kernel_wrapper % loop_code_dict))
            _user_code.append(kernel._user_code)
            _ssinds_arg.append(_ssind_decl)
            _const_args.add(loop_code_dict['const_args'])
            _const_inits.add(loop_code_dict['const_inits'])

        _loop_chain_body = indent("\n\n".join(_loop_body), 2)
        code_dict['const_args'] = "".join(_const_args)
        code_dict['const_inits'] = indent("".join(_const_inits), 1)
        code_dict['user_code'] = indent("\n".join(_user_code), 1)
        code_dict['ssinds_arg'] = "".join(["%s," % s for s in _ssinds_arg if s])
        code_dict['executor_code'] = indent(self._executor.c_code(_loop_chain_body), 1)

        return code_dict


class ParLoop(sequential.ParLoop):

    def __init__(self, kernel, it_space, *args, **kwargs):
        LazyComputation.__init__(self,
                                 kwargs['read_args'] | Const._defs,
                                 kwargs['written_args'],
                                 kwargs['inc_args'])

        # Inspector related stuff
        self._all_kernels = kwargs.get('all_kernels', [kernel])
        self._all_itspaces = kwargs.get('all_itspaces', [kernel])
        self._all_args = kwargs.get('all_args', [args])
        self._inspection = kwargs.get('inspection')
        self._executor = kwargs.get('executor')

        # Global reductions are obviously forbidden when tiling; however, the user
        # might have bypassed this condition because sure about safety. Therefore,
        # we act as in the super class, computing the result in a temporary buffer,
        # and then copying it back into the original input. This is for safety of
        # parallel global reductions (for more details, see base.ParLoop)
        self._reduced_globals = {}
        for _globs, _args in zip(kwargs.get('reduced_globals', []), self._all_args):
            if not _globs:
                continue
            for i, glob in _globs.iteritems():
                shadow_glob = _args[i].data
                for j, data in enumerate([a.data for a in args]):
                    if shadow_glob is data:
                        self._reduced_globals[j] = glob
                        break

        self._kernel = kernel
        self._actual_args = args
        self._it_space = it_space
        self._only_local = False

        for i, arg in enumerate(self._actual_args):
            arg.name = "arg%d" % i  # Override the previously cached_property name
            arg.position = i
            arg.indirect_position = i
        for i, arg1 in enumerate(self._actual_args):
            if arg1._is_dat and arg1._is_indirect:
                for arg2 in self._actual_args[i:]:
                    # We have to check for identity here (we really
                    # want these to be the same thing, not just look
                    # the same)
                    if arg2.data is arg1.data and arg2.map is arg1.map:
                        arg2.indirect_position = arg1.indirect_position

    def prepare_arglist(self, part, *args):
        arglist = [self._inspection]
        for itspace in self._all_itspaces:
            if isinstance(itspace._iterset, Subset):
                arglist.append(itspace._iterset._indices.ctypes.data)
        for arg in args:
            if arg._is_mat:
                arglist.append(arg.data.handle.handle)
            else:
                for d in arg.data:
                    # Cannot access a property of the Dat or we will force
                    # evaluation of the trace
                    arglist.append(d._data.ctypes.data)

            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    for m in map:
                        arglist.append(m._values.ctypes.data)

        for c in Const._definitions():
            arglist.append(c._data.ctypes.data)

        return arglist

    @collective
    def compute(self):
        """Execute the kernel over all members of the iteration space."""
        kwargs = {
            'all_kernels': self._all_kernels,
            'all_itspaces': self._all_itspaces,
            'all_args': self._all_args,
            'executor': self._executor,
        }
        fun = JITModule(self.kernel, self.it_space, *self.args, **kwargs)
        arglist = self.prepare_arglist(None, *self.args)

        with timed_region("ParLoopChain: executor"):
            self.halo_exchange_begin()
            fun(*(arglist + [0]))
            self.halo_exchange_end()
            fun(*(arglist + [1]))

            # Only meaningful if the user is enforcing tiling in presence of
            # global reductions
            self.reduction_begin()
            self.reduction_end()

            self.update_arg_data_state()


# An Inspector produces one of the following Schedules

class Schedule(object):

    """Represent an execution scheme for a sequence of :class:`ParLoop` objects."""

    def __init__(self, kernel):
        self._kernel = list(kernel)

    def __call__(self, loop_chain):
        """Given an iterator of :class:`ParLoop` objects (``loop_chain``),
        return an iterator of new :class:`ParLoop` objects. The input parloops
        are "scheduled" according to the strategy of this Schedule. The Schedule
        itself was produced by an Inspector.

        In the simplest case, the returned value is identical to the input
        ``loop_chain``. That is, the Inspector that created this Schedule could
        not apply any fusion or tiling.

        In general, the Schedule could fuse or tile the loops in ``loop_chain``.
        A sequence of  :class:`fusion.ParLoop` objects would then be returned.
        """
        raise NotImplementedError("Subclass must implement ``__call__`` method")


class PlainSchedule(Schedule):

    def __init__(self):
        super(PlainSchedule, self).__init__([])

    def __call__(self, loop_chain):
        return loop_chain


class FusionSchedule(Schedule):

    """Schedule an iterator of :class:`ParLoop` objects applying soft fusion."""

    def __init__(self, kernels, offsets):
        super(FusionSchedule, self).__init__(kernels)
        # Track the /ParLoop/ indices in the loop chain that each fused kernel maps to
        offsets = [0] + list(offsets)
        loop_indices = [range(offsets[i], o) for i, o in enumerate(offsets[1:])]
        self._info = [{'loop_indices': li} for li in loop_indices]

    def __call__(self, loop_chain):
        fused_par_loops = []
        for kernel, info in zip(self._kernel, self._info):
            loop_indices = info['loop_indices']
            extra_args = info.get('extra_args')
            # Create the ParLoop's arguments. Note that both the iteration set and
            # the iteration region must correspond to that of the /base/ loop
            iterregion = loop_chain[loop_indices[0]].iteration_region
            iterset = loop_chain[loop_indices[0]].it_space.iterset
            loops = [loop_chain[i] for i in loop_indices]
            args = Arg.filter_args([loop.args for loop in loops]).values()
            # Create any ParLoop's additional arguments
            extra_args = [Dat(*d)(*a) for d, a in extra_args] if extra_args else []
            args += extra_args
            # Create the actual ParLoop, resulting from the fusion of some kernels
            fused_par_loops.append(_make_object('ParLoop', kernel, iterset, *args,
                                                **{'iterate': iterregion}))
        return fused_par_loops


class HardFusionSchedule(FusionSchedule):

    """Schedule an iterator of :class:`ParLoop` objects applying hard fusion
    on top of soft fusion."""

    def __init__(self, schedule, fused):
        self._schedule = schedule
        self._fused = fused

        # Set proper loop_indices for this schedule
        self._info = dcopy(schedule._info)
        for i, info in enumerate(schedule._info):
            for k, v in info.items():
                self._info[i][k] = [i] if k == 'loop_indices' else v

        # Update the input schedule to make use of hard fusion kernels
        kernel = scopy(schedule._kernel)
        for fused_kernel, fused_map in fused:
            base, fuse = fused_kernel._kernels
            base_idx, fuse_idx = kernel.index(base), kernel.index(fuse)
            pos = min(base_idx, fuse_idx)
            kernel.insert(pos, fused_kernel)
            self._info[pos]['loop_indices'] = [base_idx, fuse_idx]
            # In addition to the union of the /base/ and /fuse/ sets of arguments,
            # a bitmap, with i-th bit indicating whether the i-th iteration in "fuse"
            # has been executed, will have to be passed in
            self._info[pos]['extra_args'] = [((fused_map.toset, None, np.int32),
                                              (RW, fused_map))]
            kernel.pop(pos+1)
            pos = max(base_idx, fuse_idx)
            self._info.pop(pos)
            kernel.pop(pos)
        self._kernel = kernel

    def __call__(self, loop_chain, only_hard=False):
        # First apply soft fusion, then hard fusion
        if not only_hard:
            loop_chain = self._schedule(loop_chain)
        return super(HardFusionSchedule, self).__call__(loop_chain)


class TilingSchedule(Schedule):

    """Schedule an iterator of :class:`ParLoop` objects applying tiling on top
    of hard fusion and soft fusion."""

    def __init__(self, schedule, inspection, executor):
        self._schedule = schedule
        self._inspection = inspection
        self._executor = executor

    def __call__(self, loop_chain):
        loop_chain = self._schedule(loop_chain)
        # Track the individual kernels, and the args of each kernel
        all_kernels = tuple((loop.kernel for loop in loop_chain))
        all_itspaces = tuple(loop.it_space for loop in loop_chain)
        all_args = tuple((Arg.specialize(loop.args, gtl_map, i) for i, (loop, gtl_map)
                         in enumerate(zip(loop_chain, self._executor.gtl_maps))))
        # Data for the actual ParLoop
        kernel = Kernel(all_kernels)
        it_space = IterationSpace(all_itspaces)
        args = Arg.filter_args([loop.args for loop in loop_chain]).values()
        reduced_globals = [loop._reduced_globals for loop in loop_chain]
        read_args = set(flatten([loop.reads for loop in loop_chain]))
        written_args = set(flatten([loop.writes for loop in loop_chain]))
        inc_args = set(flatten([loop.incs for loop in loop_chain]))
        kwargs = {
            'all_kernels': all_kernels,
            'all_itspaces': all_itspaces,
            'all_args': all_args,
            'read_args': read_args,
            'written_args': written_args,
            'reduced_globals': reduced_globals,
            'inc_args': inc_args,
            'inspection': self._inspection,
            'executor': self._executor
        }
        return [ParLoop(kernel, it_space, *args, **kwargs)]


# Loop chain inspection

class Inspector(Cached):

    """An Inspector constructs a Schedule to fuse or tile a sequence of loops.

    .. note:: For tiling, the Inspector relies on the SLOPE library."""

    _cache = {}
    _modes = ['soft', 'hard', 'tile', 'only_tile']

    @classmethod
    def _cache_key(cls, name, loop_chain, tile_size):
        key = (name, tile_size)
        for loop in loop_chain:
            if isinstance(loop, Mat._Assembly):
                continue
            key += (hash(str(loop.kernel._original_ast)),)
            for arg in loop.args:
                if arg._is_global:
                    key += (arg.data.dim, arg.data.dtype, arg.access)
                elif arg._is_dat:
                    if isinstance(arg.idx, IterationIndex):
                        idx = (arg.idx.__class__, arg.idx.index)
                    else:
                        idx = arg.idx
                    map_arity = arg.map.arity if arg.map else None
                    key += (arg.data.dim, arg.data.dtype, map_arity, idx, arg.access)
                elif arg._is_mat:
                    idxs = (arg.idx[0].__class__, arg.idx[0].index,
                            arg.idx[1].index)
                    map_arities = (arg.map[0].arity, arg.map[1].arity)
                    key += (arg.data.dims, arg.data.dtype, idxs, map_arities, arg.access)
        return key

    def __init__(self, name, loop_chain, tile_size):
        if self._initialized:
            return
        if not hasattr(self, '_inspected'):
            # Initialization can occur more than once (until the inspection is
            # actually performed), but only the first time this attribute is set
            self._inspected = 0
        self._name = name
        self._tile_size = tile_size
        self._loop_chain = loop_chain

    def inspect(self, mode):
        """Inspect this Inspector's loop chain and produce a :class:`Schedule`.

        :arg mode: can take any of the values in ``Inspector._modes``, namely
            ``soft``, ``hard``, ``tile``, ``only_tile``.

            * ``soft``: consecutive loops over the same iteration set that do
                not present RAW or WAR dependencies through indirections
                are fused.
            * ``hard``: ``soft`` fusion; then, loops over different iteration sets
                are also fused, provided that there are no RAW or WAR
                dependencies.
            * ``tile``: ``soft`` and ``hard`` fusion; then, tiling through the
                SLOPE library takes place.
            * ``only_tile``: only tiling through the SLOPE library (i.e., no fusion)
        """
        self._inspected += 1
        if self._heuristic_skip_inspection(mode):
            # Heuristically skip this inspection if there is a suspicion the
            # overhead is going to be too much; for example, when the loop
            # chain could potentially be executed only once or a few times.
            # Blow away everything we don't need any more
            del self._name
            del self._loop_chain
            del self._tile_size
            return PlainSchedule()
        elif hasattr(self, '_schedule'):
            # An inspection plan is in cache.
            # It should not be possible to pull a jit module out of the cache
            # /with/ the loop chain
            if hasattr(self, '_loop_chain'):
                raise RuntimeError("Inspector is holding onto loop_chain, memory leaks!")
            # The fusion mode was recorded, and must match the one provided for
            # this inspection
            if self.mode != mode:
                raise RuntimeError("Cached Inspector mode doesn't match")
            return self._schedule
        elif not hasattr(self, '_loop_chain'):
            # The inspection should be executed /now/. We weren't in the cache,
            # so we /must/ have a loop chain
            raise RuntimeError("Inspector must have a loop chain associated with it")
        # Finally, we check the legality of `mode`
        if mode not in Inspector._modes:
            raise TypeError("Inspection accepts only %s fusion modes",
                            str(Inspector._modes))
        self._mode = mode

        with timed_region("ParLoopChain `%s`: inspector" % self._name):
            self._schedule = PlainSchedule()
            if mode in ['soft', 'hard', 'tile']:
                self._soft_fuse()
            if mode in ['hard', 'tile']:
                self._hard_fuse()
            if mode in ['tile', 'only_tile']:
                self._tile()

        # A schedule has been computed by any of /_soft_fuse/, /_hard_fuse/ or
        # or /_tile/; therefore, consider this Inspector initialized, and
        # retrievable from cache in subsequent calls to inspect().
        self._initialized = True

        # Blow away everything we don't need any more
        del self._name
        del self._loop_chain
        del self._tile_size
        return self._schedule

    def _heuristic_skip_inspection(self, mode):
        """Decide heuristically whether to run an inspection or not."""
        # At the moment, a simple heuristic is used. If tiling is not requested,
        # then inspection is performed. If tiling is, on the other hand, requested,
        # then inspection is performed on the third time it is requested, which
        # would suggest the inspection is being asked in a loop chain context; this
        # is for amortizing the cost of data flow analysis performed by SLOPE.
        if mode in ['tile', 'only_tile'] and self._inspected < 3:
            return True
        return False

    def _filter_kernel_args(self, loops, fundecl):
        """Eliminate redundant arguments in the fused kernel signature."""
        fused_loop_args = list(flatten([l.args for l in loops]))
        unique_fused_loop_args = Arg.filter_args([l.args for l in loops])
        fused_kernel_args = fundecl.args
        binding = OrderedDict(zip(fused_loop_args, fused_kernel_args))
        new_fused_kernel_args, args_maps = [], []
        for fused_loop_arg, fused_kernel_arg in binding.items():
            unique_fused_loop_arg = unique_fused_loop_args[fused_loop_arg.data]
            if fused_loop_arg is unique_fused_loop_arg:
                new_fused_kernel_args.append(fused_kernel_arg)
                continue
            tobind_fused_kernel_arg = binding[unique_fused_loop_arg]
            if tobind_fused_kernel_arg.is_const:
                # Need to remove the /const/ qualifier from the C declaration
                # if the same argument is written to, somewhere, in the fused
                # kernel. Otherwise, /const/ must be appended, if not present
                # already, to the alias' qualifiers
                if fused_loop_arg._is_written:
                    tobind_fused_kernel_arg.qual.remove('const')
                elif 'const' not in fused_kernel_arg.qual:
                    fused_kernel_arg.qual.append('const')
            # Update the /binding/, since might be useful for the caller
            binding[fused_loop_arg] = tobind_fused_kernel_arg
            # Aliases may be created instead of changing symbol names
            if fused_kernel_arg.sym.symbol == tobind_fused_kernel_arg.sym.symbol:
                continue
            alias = ast_make_alias(dcopy(fused_kernel_arg),
                                   dcopy(tobind_fused_kernel_arg))
            args_maps.append(alias)
        fundecl.children[0].children = args_maps + fundecl.children[0].children
        fundecl.args = new_fused_kernel_args
        return binding

    def _soft_fuse(self):
        """Fuse consecutive loops over the same iteration set by concatenating
        kernel bodies and creating new :class:`ParLoop` objects representing
        the fused sequence.

        The conditions under which two loops over the same iteration set can
        be soft fused are:

            * They are both direct, OR
            * One is direct and the other indirect

        This is detailed in the paper::

            "Mesh Independent Loop Fusion for Unstructured Mesh Applications"

        from C. Bertolli et al.
        """

        def fuse(self, loops, loop_chain_index):
            # Naming convention: here, we are fusing ASTs in /fuse_asts/ within
            # /base_ast/. Same convention will be used in the /hard_fuse/ method
            kernels = [l.kernel for l in loops]
            fuse_asts = [k._original_ast if k._code else k._ast for k in kernels]
            # Fuse the actual kernels' bodies
            base_ast = dcopy(fuse_asts[0])
            retval = FindInstances.default_retval()
            base_fundecl = FindInstances(ast.FunDecl).visit(base_ast, ret=retval)[ast.FunDecl]
            if len(base_fundecl) != 1:
                raise RuntimeError("Fusing kernels, but found unexpected AST")
            base_fundecl = base_fundecl[0]
            for unique_id, _fuse_ast in enumerate(fuse_asts[1:], 1):
                fuse_ast = dcopy(_fuse_ast)
                retval = FindInstances.default_retval()
                fuse_fundecl = FindInstances(ast.FunDecl).visit(fuse_ast, ret=retval)[ast.FunDecl]
                if len(fuse_fundecl) != 1:
                    raise RuntimeError("Fusing kernels, but found unexpected AST")
                fuse_fundecl = fuse_fundecl[0]
                # 1) Extend function name
                base_fundecl.name = "%s_%s" % (base_fundecl.name, fuse_fundecl.name)
                # 2) Concatenate the arguments in the signature
                base_fundecl.args.extend(fuse_fundecl.args)
                # 3) Uniquify symbols identifiers
                retval = SymbolReferences.default_retval()
                fuse_symbols = SymbolReferences().visit(fuse_ast, ret=retval)
                for decl in fuse_fundecl.args:
                    for symbol, _ in fuse_symbols[decl.sym.symbol]:
                        symbol.symbol = "%s_%d" % (symbol.symbol, unique_id)
                # 4) Scope and concatenate bodies
                base_fundecl.children[0] = ast.Block(
                    [ast.Block(base_fundecl.children[0].children, open_scope=True),
                     ast.FlatBlock("\n\n// Begin of fused kernel\n\n"),
                     ast.Block(fuse_fundecl.children[0].children, open_scope=True)])
            # Eliminate redundancies in the /fused/ kernel signature
            self._filter_kernel_args(loops, base_fundecl)
            # Naming convention
            fused_ast = base_ast
            return Kernel(kernels, fused_ast, loop_chain_index)

        fused, fusing = [], [self._loop_chain[0]]
        for i, loop in enumerate(self._loop_chain[1:]):
            base_loop = fusing[-1]
            if base_loop.it_space != loop.it_space or \
                    (base_loop.is_indirect and loop.is_indirect):
                # Fusion not legal
                fused.append((fuse(self, fusing, len(fused)), i+1))
                fusing = [loop]
            elif (base_loop.is_direct and loop.is_direct) or \
                    (base_loop.is_direct and loop.is_indirect) or \
                    (base_loop.is_indirect and loop.is_direct):
                # This loop is fusible. Also, can speculative go on searching
                # for other loops to fuse
                fusing.append(loop)
            else:
                raise RuntimeError("Unexpected loop chain structure while fusing")
        if fusing:
            fused.append((fuse(self, fusing, len(fused)), len(self._loop_chain)))

        fused_kernels, offsets = zip(*fused)
        self._schedule = FusionSchedule(fused_kernels, offsets)
        self._loop_chain = self._schedule(self._loop_chain)

    def _hard_fuse(self):
        """Fuse consecutive loops over different iteration sets that do not
        present RAW, WAR or WAW dependencies. For examples, two loops like: ::

            par_loop(kernel_1, it_space_1,
                     dat_1_1(INC, ...),
                     dat_1_2(READ, ...),
                     ...)

            par_loop(kernel_2, it_space_2,
                     dat_2_1(INC, ...),
                     dat_2_2(READ, ...),
                     ...)

        where ``dat_1_1 == dat_2_1`` and, possibly (but not necessarily),
        ``it_space_1 != it_space_2``, can be hard fused. Note, in fact, that
        the presence of ``INC`` does not imply a real WAR dependency, because
        increments are associative."""

        def has_raw_or_war(loop1, loop2):
            # Note that INC after WRITE is a special case of RAW dependency since
            # INC cannot take place before WRITE.
            return loop2.reads & loop1.writes or loop2.writes & loop1.reads or \
                loop1.incs & (loop2.writes - loop2.incs) or \
                loop2.incs & (loop1.writes - loop1.incs)

        def has_iai(loop1, loop2):
            return loop1.incs & loop2.incs

        def fuse(base_loop, loop_chain, fused):
            """Try to fuse one of the loops in ``loop_chain`` with ``base_loop``."""
            for loop in loop_chain:
                if has_raw_or_war(loop, base_loop):
                    # Can't fuse across loops preseting RAW or WAR dependencies
                    return
                if loop.it_space == base_loop.it_space:
                    warning("Ignoring unexpected sequence of loops in loop fusion")
                    continue
                # Is there an overlap in any incremented regions? If that is
                # the case, then fusion can really be useful, by allowing to
                # save on the number of indirect increments or matrix insertions
                common_inc_data = has_iai(base_loop, loop)
                if not common_inc_data:
                    continue
                common_incs = [a for a in base_loop.args + loop.args
                               if a.data in common_inc_data]
                # Hard fusion potentially doable provided that we own a map between
                # the iteration spaces involved
                maps = list(set(flatten([a.map for a in common_incs])))
                maps += [m.factors for m in maps if hasattr(m, 'factors')]
                maps = list(flatten(maps))
                set1, set2 = base_loop.it_space.iterset, loop.it_space.iterset
                fused_map = [m for m in maps if set1 == m.iterset and set2 == m.toset]
                if fused_map:
                    fused.append((base_loop, loop, fused_map[0], common_incs[1]))
                    return
                fused_map = [m for m in maps if set1 == m.toset and set2 == m.iterset]
                if fused_map:
                    fused.append((loop, base_loop, fused_map[0], common_incs[0]))
                    return

        # First, find fusible kernels
        fused = []
        for i, l in enumerate(self._loop_chain, 1):
            fuse(l, self._loop_chain[i:], fused)
        if not fused:
            return

        # Then, create a suitable hard-fusion kernel
        # The hard fused kernel will have the following structure:
        #
        # wrapper (args: Union(kernel1, kernel2, extra):
        #   staging of pointers
        #   ...
        #   fusion (staged pointers, ..., extra)
        #   insertion (...)
        #
        # Where /extra/ represents additional arguments, like the map from
        # /kernel1/ iteration space to /kernel2/ iteration space. The /fusion/
        # function looks like:
        #
        # fusion (...):
        #   kernel1 (buffer, ...)
        #   for i = 0 to arity:
        #     if not already_executed[i]:
        #       kernel2 (buffer[..], ...)
        #
        # Where /arity/ is the number of /kernel2/ iterations incident to
        # /kernel1/ iterations.
        _fused = []
        for base_loop, fuse_loop, fused_map, fused_inc_arg in fused:
            # Start with analyzing the kernel ASTs. Note: fusion occurs on fresh
            # copies of the /base/ and /fuse/ ASTs. This is because the optimization
            # of the /fused/ AST should be independent of that of individual ASTs,
            # and subsequent cache hits for non-fused ParLoops should always retrive
            # the original, unmodified ASTs. This is important not just for the
            # sake of performance, but also for correctness of padding, since hard
            # fusion changes the signature of /fuse/ (in particular, the buffers that
            # are provided for computation on iteration spaces)
            base, fuse = base_loop.kernel, fuse_loop.kernel
            base_ast = dcopy(base._original_ast) if base._code else dcopy(base._ast)
            retval = FindInstances.default_retval()
            base_info = FindInstances((ast.FunDecl, ast.PreprocessNode)).visit(base_ast, ret=retval)
            base_headers = base_info[ast.PreprocessNode]
            base_fundecl = base_info[ast.FunDecl]
            fuse_ast = dcopy(fuse._original_ast) if fuse._code else dcopy(fuse._ast)
            retval = FindInstances.default_retval()
            fuse_info = FindInstances((ast.FunDecl, ast.PreprocessNode)).visit(fuse_ast, ret=retval)
            fuse_headers = fuse_info[ast.PreprocessNode]
            fuse_fundecl = fuse_info[ast.FunDecl]
            retval = SymbolReferences.default_retval()
            fuse_symbol_refs = SymbolReferences().visit(fuse_ast, ret=retval)
            if len(base_fundecl) != 1 or len(fuse_fundecl) != 1:
                raise RuntimeError("Fusing kernels, but found unexpected AST")
            base_fundecl = base_fundecl[0]
            fuse_fundecl = fuse_fundecl[0]

            # Craft the /fusion/ kernel #

            # 1A) Create /fusion/ arguments and signature
            body = ast.Block([])
            fusion_name = '%s_%s' % (base_fundecl.name, fuse_fundecl.name)
            fusion_args = base_fundecl.args + fuse_fundecl.args
            fusion_fundecl = ast.FunDecl(base_fundecl.ret, fusion_name,
                                         fusion_args, body)

            # 1B) Filter out duplicate arguments, and append extra arguments to
            # the function declaration
            binding = self._filter_kernel_args([base_loop, fuse_loop], fusion_fundecl)
            fusion_fundecl.args += [ast.Decl('int**', ast.Symbol('executed'))]

            # 1C) Create /fusion/ body
            base_funcall_syms = [ast.Symbol(d.sym.symbol)
                                 for d in base_fundecl.args]
            base_funcall = ast.FunCall(base_fundecl.name, *base_funcall_syms)
            fuse_funcall_syms = [ast.Symbol(binding[arg].sym.symbol)
                                 for arg in fuse_loop.args]
            fuse_funcall = ast.FunCall(fuse_fundecl.name, *fuse_funcall_syms)
            if_cond = ast.Not(ast.Symbol('executed', ('i', 0)))
            if_update = ast.Assign(ast.Symbol('executed', ('i', 0)), ast.Symbol('1'))
            if_exec = ast.If(if_cond, [ast.Block([fuse_funcall, if_update],
                                                 open_scope=True)])
            fuse_body = ast.Block([if_exec], open_scope=True)
            fuse_for = ast.c_for('i', fused_map.arity, fuse_body, pragma=None)
            body.children.extend([base_funcall, fuse_for.children[0]])

            # Modify the /fuse/ kernel #
            # This is to take into account that many arguments are shared with
            # /base/, so they will only staged once for /base/. This requires
            # tweaking the way the arguments are declared and accessed in /fuse/
            # kernel. For example, the shared incremented array (called /buffer/
            # in the pseudocode in the comment above) now needs to take offsets
            # to be sure the locations that /base/ is supposed to increment are
            # actually accessed. The same concept apply to indirect arguments.
            ofs_syms, ofs_decls, ofs_vals = [], [], []
            init = lambda v: '{%s}' % ', '.join([str(j) for j in v])
            for i, fuse_args in enumerate(zip(fuse_loop.args, fuse_fundecl.args)):
                fuse_loop_arg, fuse_kernel_arg = fuse_args
                sym_id = fuse_kernel_arg.sym.symbol
                if fuse_loop_arg == fused_inc_arg:
                    # 2A) The shared incremented argument. A 'buffer' of statically
                    # known size is expected by the kernel, so the offset is used
                    # to index into it
                    # Note: the /fused_map/ is a factor of the /base/ iteration
                    # set map, so the order the /fuse/ loop iterations are executed
                    # (in the /for i=0 to arity/ loop) reflects the order of the
                    # entries in /fused_map/
                    fuse_inc_refs = fuse_symbol_refs[sym_id]
                    fuse_inc_refs = [sym for sym, parent in fuse_inc_refs
                                     if not isinstance(parent, ast.Decl)]
                    # Handle the declaration
                    fuse_kernel_arg.sym.rank = binding[fused_inc_arg].sym.rank
                    for b in fused_inc_arg._block_shape:
                        for rc in b:
                            _ofs_vals = [[0] for j in range(len(rc))]
                            for j, ofs in enumerate(rc):
                                ofs_sym_id = 'm_ofs_%d_%d' % (i, j)
                                ofs_syms.append(ofs_sym_id)
                                ofs_decls.append(ast.Decl('int', ast.Symbol(ofs_sym_id)))
                                _ofs_vals[j].append(ofs)
                            for s in fuse_inc_refs:
                                s.offset = tuple((1, o) for o in ofs_syms)
                            ofs_vals.extend([init(o) for o in _ofs_vals])
                    # Tell COFFEE that the argument is not an empty buffer anymore,
                    # so any write to it must actually be an increment
                    fuse_kernel_arg.pragma = set([ast.INC])
                elif fuse_loop_arg._is_indirect:
                    # 2B) All indirect arguments. At the C level, these arguments
                    # are of pointer type, so simple pointer arithmetic is used
                    # to ensure the kernel accesses are to the correct locations
                    fuse_arity = fuse_loop_arg.map.arity
                    base_arity = fuse_arity*fused_map.arity
                    cdim = fuse_loop_arg.data.dataset.cdim
                    size = fuse_arity*cdim
                    if fuse_loop_arg._flatten and cdim > 1:
                        # Set the proper storage layout before invoking /fuse/
                        ofs_tmp = '_%s' % fuse_kernel_arg.sym.symbol
                        ofs_tmp_sym = ast.Symbol(ofs_tmp, (size,))
                        ofs_tmp_decl = ast.Decl('%s*' % fuse_loop_arg.ctype, ofs_tmp_sym)
                        _ofs_vals = [[base_arity*j + k for k in range(fuse_arity)]
                                     for j in range(cdim)]
                        _ofs_vals = [[fuse_arity*j + k for k in flatten(_ofs_vals)]
                                     for j in range(fused_map.arity)]
                        _ofs_vals = list(flatten(_ofs_vals))
                        ofs_idx_sym = 'v_i_ofs_%d' % i
                        body.children.insert(0, ast.Decl(
                            'int', ast.Symbol(ofs_idx_sym, (len(_ofs_vals),)),
                            ast.ArrayInit(init(_ofs_vals)), ['static', 'const']))
                        ofs_idx_syms = [ast.Symbol(ofs_idx_sym, ('i',), ((size, j),))
                                        for j in range(size)]
                        ofs_assigns = [ofs_tmp_decl]
                        ofs_assigns += [ast.Assign(ast.Symbol(ofs_tmp, (j,)),
                                                   ast.Symbol(sym_id, (k,)))
                                        for j, k in enumerate(ofs_idx_syms)]
                        # Need to reflect this onto the invocation of /fuse/
                        fuse_funcall.children[fuse_loop.args.index(fuse_loop_arg)] = \
                            ast.Symbol(ofs_tmp)
                    else:
                        # In this case, can just use offsets since it's not a
                        # multi-dimensional Dat
                        ofs_sym = ast.Symbol('ofs', (len(ofs_vals), 'i'))
                        ofs_assigns = [ast.Assign(sym_id, ast.Sum(sym_id, ofs_sym))]
                        ofs_vals.append(init([j*size for j in range(fused_map.arity)]))
                    if_exec.children[0].children[0:0] = ofs_assigns
            # Now change the /fusion/ kernel body accordingly
            body.children.insert(0, ast.Decl(
                'int', ast.Symbol('ofs', (len(ofs_vals), fused_map.arity)),
                ast.ArrayInit(init(ofs_vals)), ['static', 'const']))
            if_exec.children[0].children[0:0] = \
                [ast.Decl('int', ast.Symbol(s), ast.Symbol('ofs', (i, 'i')))
                 for i, s in enumerate(ofs_syms)]

            # 2C) Change /fuse/ kernel invocation, declaration, and body
            fuse_funcall.children.extend([ast.Symbol(s) for s in ofs_syms])
            fuse_fundecl.args.extend(ofs_decls)

            # 2D) Hard fusion breaks any padding applied to the /fuse/ kernel, so
            # this transformation pass needs to be re-performed;

            # Create a /fusion.Kernel/ object to be used to update the schedule
            fused_headers = set([str(h) for h in base_headers + fuse_headers])
            fused_ast = ast.Root([ast.PreprocessNode(h) for h in fused_headers] +
                                 [base_fundecl, fuse_fundecl, fusion_fundecl])
            kernels = [base, fuse]
            loop_chain_index = (self._loop_chain.index(base_loop),
                                self._loop_chain.index(fuse_loop))
            _fused.append((Kernel(kernels, fused_ast, loop_chain_index), fused_map))

        # Finally, generate a new schedule
        self._schedule = HardFusionSchedule(self._schedule, _fused)
        self._loop_chain = self._schedule(self._loop_chain, only_hard=True)

    def _tile(self):
        """Tile consecutive loops over different iteration sets characterized
        by RAW and WAR dependencies. This requires interfacing with the SLOPE
        library."""

        def format_set(s):
            superset, s_name = None, s.name
            if isinstance(s, Subset):
                superset = s.superset.name
                s_name = "%s_ss" % s.name
            return s_name, s.core_size, s.exec_size - s.core_size, \
                s.total_size - s.exec_size, superset

        # Set the SLOPE backend
        global MPI
        if not MPI.parallel:
            if configuration['backend'] == 'sequential':
                slope_backend = 'SEQUENTIAL'
            if configuration['backend'] == 'openmp':
                slope_backend = 'OMP'
        elif configuration['backend'] == 'sequential':
            slope_backend = 'ONLY_MPI'
        elif configuration['backend'] == 'openmp':
            slope_backend = 'OMP_MPI'
        else:
            warning("Could not find a valid SLOPE backend, tiling skipped")
            return
        slope.set_exec_mode(slope_backend)
        log_info("SLOPE backend set to %s" % slope_backend)

        # The SLOPE inspector, which needs be populated with sets, maps,
        # descriptors, and loop chain structure
        inspector = slope.Inspector()

        # Build inspector and argument types and values
        arguments = []
        insp_sets, insp_maps, insp_loops = set(), {}, []
        for loop in self._loop_chain:
            slope_desc = set()
            # Add sets
            iterset = loop.it_space.iterset
            is_name, is_cs, is_es, is_ts, superset = format_set(iterset)
            insp_sets.add((is_name, is_cs, is_es, is_ts, superset))
            # If iterating over a subset, we fake an indirect parloop from the
            # (iteration) subset to the superset. This allows the propagation of
            # tiling across the hierarchy of sets (see SLOPE for further info)
            if superset:
                map_name = "%s_tosuperset" % is_name
                insp_maps[is_name] = (map_name, is_name, iterset.superset.name,
                                      iterset.indices)
                slope_desc.add((map_name, INC._mode))
            for a in loop.args:
                # Add access descriptors
                maps = as_tuple(a.map, Map)
                if not maps:
                    # Simplest case: direct loop
                    slope_desc.add(('DIRECT', a.access._mode))
                else:
                    # Add maps (there can be more than one per argument if the arg
                    # is actually a Mat - in which case there are two maps - or if
                    # a MixedMap) and relative descriptors
                    for i, map in enumerate(maps):
                        for j, m in enumerate(map):
                            map_name = "%s%d_%d" % (m.name, i, j)
                            insp_maps[m.name] = (map_name, m.iterset.name,
                                                 m.toset.name, m.values_with_halo)
                            slope_desc.add((map_name, a.access._mode))
                            insp_sets.add(format_set(m.iterset))
                            insp_sets.add(format_set(m.toset))
            # Add loop
            insp_loops.append((loop.kernel.name, is_name, list(slope_desc)))
        # Provide structure of loop chain to SLOPE
        arguments.extend([inspector.add_sets(insp_sets)])
        arguments.extend([inspector.add_maps(insp_maps.values())])
        inspector.add_loops(insp_loops)

        # Get type and value of additional arguments that SLOPE can exploit
        arguments.extend([inspector.set_external_dats()])

        # Set a specific tile size
        arguments.extend([inspector.set_tile_size(self._tile_size)])

        # Tell SLOPE the rank of the MPI process
        arguments.extend([inspector.set_mpi_rank(MPI.comm.rank)])

        # Arguments types and values
        argtypes, argvalues = zip(*arguments)

        # Generate the C code
        src = inspector.generate_code()

        # Return type of the inspector
        rettype = slope.Executor.meta['py_ctype_exec']

        # Compiler and linker options
        slope_dir = os.environ['SLOPE_DIR']
        compiler = coffee.plan.compiler.get('name')
        cppargs = slope.get_compile_opts(compiler)
        cppargs += ['-I%s/%s' % (slope_dir, slope.get_include_dir())]
        ldargs = ['-L%s/%s' % (slope_dir, slope.get_lib_dir()),
                  '-l%s' % slope.get_lib_name()]

        # Compile and run inspector
        fun = compilation.load(src, "cpp", "inspector", cppargs, ldargs,
                               argtypes, rettype, compiler)
        inspection = fun(*argvalues)

        # Finally, get the Executor representation, to be used at executor
        # code generation time
        executor = slope.Executor(inspector)

        self._schedule = TilingSchedule(self._schedule, inspection, executor)

    @property
    def mode(self):
        return self._mode


# Interface for triggering loop fusion

def fuse(name, loop_chain, **kwargs):
    """Apply fusion (and possibly tiling) to an iterator of :class:`ParLoop`
    obecjts, which we refer to as ``loop_chain``. Return an iterator of
    :class:`ParLoop` objects, in which some loops may have been fused or tiled.
    If fusion could not be applied, return the unmodified ``loop_chain``.

    .. note::
       At the moment, the following features are not supported, in which
       case the unmodified ``loop_chain`` is returned.

        * mixed ``Datasets`` and ``Maps``;
        * extruded ``Sets``

    .. note::
       Tiling cannot be applied if any of the following conditions verifies:

        * a global reduction/write occurs in ``loop_chain``
    """
    tile_size = kwargs.get('tile_size', 0)
    force_glb = kwargs.get('force_glb', False)
    mode = kwargs.get('mode', 'hard')

    # If there is nothing to fuse, just return
    if len(loop_chain) in [0, 1]:
        return loop_chain

    # Search for _Assembly objects since they introduce a synchronization point;
    # that is, loops cannot be fused across an _Assembly object. In that case, try
    # to fuse only the segment of loop chain right before the synchronization point
    remainder = []
    synch_points = [l for l in loop_chain if isinstance(l, Mat._Assembly)]
    if synch_points:
        if len(synch_points) > 1:
            warning("Fusing loops and found more than one synchronization point")
        synch_point = loop_chain.index(synch_points[0])
        remainder, loop_chain = loop_chain[synch_point:], loop_chain[:synch_point]

    # If there is nothing left to fuse (e.g. only _Assembly objects were present), return
    if len(loop_chain) in [0, 1]:
        return loop_chain + remainder

    # Skip if loops in /loop_chain/ are already /fusion/ objects: this could happen
    # when loops had already been fused in a /loop_chain/ context
    if any([isinstance(l, ParLoop) for l in loop_chain]):
        return loop_chain + remainder

    # Global reductions are dangerous for correctness, so avoid fusion unless the
    # user is forcing it
    if not force_glb and any([l._reduced_globals for l in loop_chain]):
        return loop_chain + remainder

    # Loop fusion requires modifying kernels, so ASTs must be present...
    if not mode == 'only_tile':
        if any([not hasattr(l.kernel, '_ast') or not l.kernel._ast for l in loop_chain]):
            return loop_chain + remainder
        # ...and must not be "fake" ASTs
        if any([isinstance(l.kernel._ast, ast.FlatBlock) for l in loop_chain]):
            return loop_chain + remainder

    # Mixed still not supported
    if any(a._is_mixed for a in flatten([l.args for l in loop_chain])):
        return loop_chain + remainder

    # Extrusion still not supported
    if any([l.is_layered for l in loop_chain]):
        return loop_chain + remainder

    # Check if tiling needs be applied
    if mode in ['tile', 'only_tile']:
        # Loop tiling requires the SLOPE library to be available on the system.
        if slope is None:
            warning("Requested tiling, but couldn't locate SLOPE. Check the PYTHONPATH")
            return loop_chain + remainder
        try:
            os.environ['SLOPE_DIR']
        except KeyError:
            warning("Set the env variable SLOPE_DIR to the location of SLOPE")
            warning("Loops won't be fused, and plain ParLoops will be executed")
            return loop_chain + remainder

    # Get an inspector for fusing this loop_chain, possibly retrieving it from
    # the cache, and obtain the fused ParLoops through the schedule it produces
    inspector = Inspector(name, loop_chain, tile_size)
    schedule = inspector.inspect(mode)
    return schedule(loop_chain) + remainder


@contextmanager
def loop_chain(name, tile_size=1, **kwargs):
    """Analyze the sub-trace of loops lazily evaluated in this contextmanager ::

        [loop_0, loop_1, ..., loop_n-1]

    and produce a new sub-trace (``m <= n``) ::

        [fused_loops_0, fused_loops_1, ..., fused_loops_m-1, peel_loops]

    which is eventually inserted in the global trace of :class:`ParLoop` objects.

    That is, sub-sequences of :class:`ParLoop` objects are potentially replaced by
    new :class:`ParLoop` objects representing the fusion or the tiling of the
    original trace slice.

    :arg name: identifier of the loop chain
    :arg tile_size: suggest a starting average tile size
    :arg kwargs:
        * num_unroll (default=1): in a time stepping loop, the length of the loop
            chain is given by ``num_loops * num_unroll``, where ``num_loops`` is the
            number of loops per time loop iteration. Therefore, setting this value
            to a number >1 enables tiling longer chains.
        * force_glb (default=False): force tiling even in presence of global
            reductions. In this case, the user becomes responsible of semantic
            correctness.
        * mode (default='tile'): the fusion/tiling mode (accepted: soft, hard,
            tile, only_tile)
    """
    num_unroll = kwargs.get('num_unroll', 1)
    force_glb = kwargs.get('force_glb', False)
    mode = kwargs.get('mode', 'tile')

    # Get a snapshot of the trace before new par loops are added within this
    # context manager
    from base import _trace
    stamp = list(_trace._trace)

    yield

    trace = _trace._trace
    if num_unroll < 1 or stamp == trace:
        return

    # What's the first item /B/ that appeared in the trace /before/ entering the
    # context manager and that still has to be executed ?
    # The loop chain will be (B, end_of_current_trace]
    bottom = 0
    for i in reversed(stamp):
        if i in trace:
            bottom = trace.index(i) + 1
            break
    extracted_loop_chain = trace[bottom:]

    # Unroll the loop chain /num_unroll/ times before fusion/tiling
    total_loop_chain = loop_chain.unrolled_loop_chain + extracted_loop_chain
    if len(total_loop_chain) / len(extracted_loop_chain) == num_unroll:
        bottom = trace.index(total_loop_chain[0])
        trace[bottom:] = fuse(name, total_loop_chain,
                              tile_size=tile_size, force_glb=force_glb, mode=mode)
        loop_chain.unrolled_loop_chain = []
    else:
        loop_chain.unrolled_loop_chain.extend(extracted_loop_chain)
loop_chain.unrolled_loop_chain = []

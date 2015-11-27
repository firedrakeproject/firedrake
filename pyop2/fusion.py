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
from itertools import groupby
import os
import sys

from base import *
import base
import compilation
import sequential
import host
from backends import _make_object
from caching import Cached
from profiling import timed_region
from logger import warning, info as log_info
from mpi import MPI, collective
from utils import flatten, strip, as_tuple

import coffee
from coffee import base as ast
from coffee.utils import ast_make_alias, ItSpace
from coffee.visitors import FindInstances, SymbolReferences


try:
    """Is SLOPE accessible ?"""
    sys.path.append(os.path.join(os.environ['SLOPE_DIR'], 'python'))
    import slope_python as slope
    os.environ['SLOPE_METIS']

    # Set the SLOPE backend
    backend = os.environ.get('SLOPE_BACKEND')
    if backend not in ['SEQUENTIAL', 'OMP']:
        backend = 'SEQUENTIAL'
    if MPI.parallel:
        if backend == 'SEQUENTIAL':
            backend = 'ONLY_MPI'
        if backend == 'OMP':
            backend = 'OMP_MPI'
    slope.set_exec_mode(backend)
    log_info("SLOPE backend set to %s" % backend)
except:
    warning("Couldn't locate SLOPE, no tiling possible. Check SLOPE_{DIR,METIS} env vars")
    slope = None


lazy_trace_name = 'lazy_trace'
"""The default name for sequences of par loops extracted from the trace produced
by lazy evaluation."""


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
                        fa.access = RW
                    elif WRITE in [a.access, fa.access]:
                        # Can't be a READ, so just stick to WRITE regardless of what
                        # the other access mode is
                        fa.access = WRITE
                    else:
                        # Neither READ nor WRITE, so access modes are some
                        # combinations of RW, INC, MIN, MAX. For simplicity,
                        # just make it RW.
                        fa.access = RW
        return filtered_args

    def c_arg_bindto(self, arg):
        """Assign this Arg's c_pointer to ``arg``."""
        if self.ctype != arg.ctype:
            raise RuntimeError("Cannot bind arguments having mismatching types")
        return "%s* %s = %s" % (self.ctype, self.c_arg_name(), arg.c_arg_name())

    def c_ind_data(self, idx, i, j=0, is_top=False, layers=1, offset=None):
        return "%(name)s + (%(map_name)s[n * %(arity)s + %(idx)s]%(top)s%(off_mul)s%(off_add)s)* %(dim)s%(off)s" % \
            {'name': self.c_arg_name(i),
             'map_name': self.c_map_name(i, 0),
             'arity': self.map.split[i].arity,
             'idx': idx,
             'top': ' + start_layer' if is_top else '',
             'dim': self.data[i].cdim,
             'off': ' + %d' % j if j else '',
             'off_mul': ' * %d' % offset if is_top and offset is not None else '',
             'off_add': ' + %d' % offset if not is_top and offset is not None else ''}

    def c_map_name(self, i, j):
        return self._c_local_maps[i][j]

    def c_global_reduction_name(self, count=None):
        return "%(name)s_l%(count)d[0]" % {
            'name': self.c_arg_name(),
            'count': count}

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
            suitably called within the wrapper function)."""

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
        self._original_ast = dcopy(self._ast)
        return super(Kernel, self)._ast_to_c(self._ast, opts)

    def _multiple_ast_to_c(self, kernels):
        """Glue together different ASTs (or strings) such that: ::

            * clashes due to identical function names are avoided;
            * duplicate functions (same name, same body) are avoided.
        """
        code = ""
        identifier = lambda k: k.cache_key[1:]
        unsorted_kernels = sorted(kernels, key=identifier)
        for i, (_, kernel_group) in enumerate(groupby(unsorted_kernels, identifier)):
            duplicates = list(kernel_group)
            main = duplicates[0]
            if main._original_ast:
                main_ast = dcopy(main._original_ast)
                finder = FindInstances((ast.FunDecl, ast.FunCall))
                found = finder.visit(main_ast, ret=FindInstances.default_retval())
                for fundecl in found[ast.FunDecl]:
                    new_name = "%s_%d" % (fundecl.name, i)
                    # Need to change the name of any inner functions too
                    for funcall in found[ast.FunCall]:
                        if fundecl.name == funcall.funcall.symbol:
                            funcall.funcall.symbol = new_name
                    fundecl.name = new_name
                function_name = "%s_%d" % (main._name, i)
                code += host.Kernel._ast_to_c(main, main_ast, main._opts)
            else:
                # AST not available so can't change the name, hopefully there
                # will not be compile time clashes.
                function_name = main._name
                code += main._code
            # Finally track the function name within this /fusion.Kernel/
            for k in duplicates:
                try:
                    k._function_names[self.cache_key] = function_name
                except AttributeError:
                    k._function_names = {
                        k.cache_key: k.name,
                        self.cache_key: function_name
                    }
            code += "\n"
        return code

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

        # We need to distinguish between the kernel name and the function name(s).
        # Since /fusion.Kernel/ are, in general, collections of functions, the same
        # function (which is itself associated a Kernel) can appear in different
        # /fusion.Kernel/ objects, but possibly under a different name (to avoid
        # name clashes)
        self._name = "_".join([k.name for k in kernels])
        self._function_names = {self.cache_key: self._name}

        self._cpp = any(k._cpp for k in kernels)
        self._opts = dict(flatten([k._opts.items() for k in kernels]))
        self._applied_blas = any(k._applied_blas for k in kernels)
        self._include_dirs = list(set(flatten([k._include_dirs for k in kernels])))
        self._headers = list(set(flatten([k._headers for k in kernels])))
        self._user_code = "\n".join(list(set([k._user_code for k in kernels])))
        self._attached_info = False

        # What sort of Kernel do I have?
        if fused_ast:
            # A single, already fused AST (code generation is then delayed)
            self._ast = fused_ast
            self._code = None
        else:
            # Multiple kernels, interpreted as different C functions
            self._ast = None
            self._code = self._multiple_ast_to_c(kernels)
        self._original_ast = self._ast
        self._kernels = kernels

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
  int i = %(tile_iter)s;
  %(vec_inits)s;
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

            # ... use the proper function name (the function name of the kernel
            # within *this* specific loop chain)
            loop_code_dict['kernel_name'] = kernel._function_names[self._kernel.cache_key]

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

    def __init__(self, kernels=None):
        super(PlainSchedule, self).__init__(kernels or [])

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
            # Create any ParLoop additional arguments
            extra_args = [Dat(*d)(*a) for d, a in extra_args] if extra_args else []
            args += extra_args
            # Remove now incorrect cached properties:
            for a in args:
                a.__dict__.pop('name', None)
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
        for ofs, (fused_kernel, fused_map) in enumerate(fused):
            # Find the position of the /fused/ kernel in the new loop chain.
            base, fuse = fused_kernel._kernels
            base_idx, fuse_idx = kernel.index(base), kernel.index(fuse)
            pos = min(base_idx, fuse_idx)
            self._info[pos]['loop_indices'] = [base_idx + ofs, fuse_idx + ofs]
            # We also need a bitmap, with the i-th bit indicating whether the i-th
            # iteration in "fuse" has been executed or not
            self._info[pos]['extra_args'] = [((fused_map.toset, None, np.int32),
                                              (RW, fused_map))]
            # Now we can modify the kernel sequence
            kernel.insert(pos, fused_kernel)
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

    def __init__(self, kernel, schedule, inspection, executor):
        self._schedule = schedule
        self._inspection = inspection
        self._executor = executor
        self._kernel = kernel

    def __call__(self, loop_chain):
        loop_chain = self._schedule(loop_chain)
        # Track the individual kernels, and the args of each kernel
        all_itspaces = tuple(loop.it_space for loop in loop_chain)
        all_args = tuple((Arg.specialize(loop.args, gtl_map, i) for i, (loop, gtl_map)
                         in enumerate(zip(loop_chain, self._executor.gtl_maps))))
        # Data for the actual ParLoop
        it_space = IterationSpace(all_itspaces)
        args = Arg.filter_args([loop.args for loop in loop_chain]).values()
        reduced_globals = [loop._reduced_globals for loop in loop_chain]
        read_args = set(flatten([loop.reads for loop in loop_chain]))
        written_args = set(flatten([loop.writes for loop in loop_chain]))
        inc_args = set(flatten([loop.incs for loop in loop_chain]))
        kwargs = {
            'all_kernels': self._kernel._kernels,
            'all_itspaces': all_itspaces,
            'all_args': all_args,
            'read_args': read_args,
            'written_args': written_args,
            'reduced_globals': reduced_globals,
            'inc_args': inc_args,
            'inspection': self._inspection,
            'executor': self._executor
        }
        return [ParLoop(self._kernel, it_space, *args, **kwargs)]


# Loop chain inspection

class Inspector(Cached):

    """An Inspector constructs a Schedule to fuse or tile a sequence of loops.

    .. note:: For tiling, the Inspector relies on the SLOPE library."""

    _cache = {}
    _modes = ['soft', 'hard', 'tile', 'only_tile', 'only_omp']

    @classmethod
    def _cache_key(cls, name, loop_chain, **options):
        key = (name,)
        if name != lazy_trace_name:
            # Special case: the Inspector comes from a user-defined /loop_chain/
            key += (options['mode'], options['tile_size'], options['partitioning'])
            key += (loop_chain[0].kernel.cache_key,)
            return key
        # Inspector extracted from lazy evaluation trace
        for loop in loop_chain:
            if isinstance(loop, Mat._Assembly):
                continue
            key += (loop.kernel.cache_key,)
            key += (loop.it_space.cache_key, loop.it_space.iterset.sizes)
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

    def __init__(self, name, loop_chain, **options):
        """Initialize an Inspector object.

        :arg name: a name for the Inspector
        :arg loop_chain: an iterator for the loops that will be fused/tiled
        :arg options: a set of parameters to drive fusion/tiling
            * mode: can take any of the values in ``Inspector._modes``, namely
                soft, hard, tile, only_tile, only_omp:
                * soft: consecutive loops over the same iteration set that do
                    not present RAW or WAR dependencies through indirections
                    are fused.
                * hard: ``soft`` fusion; then, loops over different iteration sets
                    are also fused, provided that there are no RAW or WAR
                    dependencies.
                * tile: ``soft`` and ``hard`` fusion; then, tiling through the
                    SLOPE library takes place.
                * only_tile: only tiling through the SLOPE library (i.e., no fusion)
                * only_omp: ompize individual parloops through the SLOPE library
            * tile_size: starting average tile size
            * partitioning: strategy for tile partitioning
            * extra_halo: are we providing SLOPE with extra halo to be efficient
                and allow it to minimize redundant computation ?
        """
        if self._initialized:
            return
        self._name = name
        self._loop_chain = loop_chain
        self._mode = options.pop('mode')
        self._options = options
        self._schedule = PlainSchedule([loop.kernel for loop in self._loop_chain])

    def inspect(self):
        """Inspect the loop chain and produce a :class:`Schedule`."""
        if self._initialized:
            # An inspection plan is in cache.
            return self._schedule
        elif self._heuristic_skip_inspection():
            # Not in cache, and too premature for running a potentially costly inspection
            del self._name
            del self._loop_chain
            del self._mode
            del self._options
            return self._schedule

        # Is `mode` legal ?
        if self.mode not in Inspector._modes:
            raise RuntimeError("Inspection accepts only %s fusion modes", Inspector._modes)

        with timed_region("ParLoopChain `%s`: inspector" % self._name):
            if self.mode in ['soft', 'hard', 'tile']:
                self._soft_fuse()
            if self.mode in ['hard', 'tile']:
                self._hard_fuse()
            if self.mode in ['tile', 'only_tile', 'only_omp']:
                self._tile()

        # A schedule has been computed. The Inspector is initialized and therefore
        # retrievable from cache. We then blow away everything we don't need any more.
        self._initialized = True
        del self._name
        del self._loop_chain
        del self._mode
        del self._options
        return self._schedule

    def _heuristic_skip_inspection(self):
        """Decide, heuristically, whether to run an inspection or not.
        If tiling is not requested, then inspection is performed.
        If tiling is requested, then inspection is performed on the third
        invocation. The fact that an inspection for the same loop chain
        is requested multiple times suggests the parloops originate in a
        time stepping loop. The cost of building tiles in SLOPE-land would
        then be amortized over several iterations."""
        self._ninsps = self._ninsps + 1 if hasattr(self, '_ninsps') else 1
        if self.mode in ['tile', 'only_tile'] and self._ninsps < 3:
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

        reads = lambda l: set([a.data for a in l.args if a.access in [READ, RW]])
        writes = lambda l: set([a.data for a in l.args if a.access in [RW, WRITE, MIN, MAX]])
        incs = lambda l: set([a.data for a in l.args if a.access in [INC]])

        def has_raw_or_war(loop1, loop2):
            # Note that INC after WRITE is a special case of RAW dependency since
            # INC cannot take place before WRITE.
            return reads(loop2) & writes(loop1) or writes(loop2) & reads(loop1) or \
                incs(loop1) & (writes(loop2) - incs(loop2)) or \
                incs(loop2) & (writes(loop1) - incs(loop1))

        def has_iai(loop1, loop2):
            return incs(loop1) & incs(loop2)

        def fuse(base_loop, loop_chain, fused):
            """Try to fuse one of the loops in ``loop_chain`` with ``base_loop``."""
            for loop in loop_chain:
                if has_raw_or_war(loop, base_loop):
                    # Can't fuse across loops preseting RAW or WAR dependencies
                    return []
                if loop.it_space == base_loop.it_space:
                    warning("Ignoring unexpected sequence of loops in loop fusion")
                    continue
                # Is there an overlap in any of the incremented regions? If that is
                # the case, then fusion can really be beneficial
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
                    return loop_chain[:loop_chain.index(loop)+1]
                fused_map = [m for m in maps if set1 == m.toset and set2 == m.iterset]
                if fused_map:
                    fused.append((loop, base_loop, fused_map[0], common_incs[0]))
                    return loop_chain[:loop_chain.index(loop)+1]
            return []

        # First, find fusible kernels
        fusible, skip = [], []
        for i, l in enumerate(self._loop_chain, 1):
            if l in skip:
                # /l/ occurs between (hard) fusible loops, let's leave it where
                # it is for safeness
                continue
            skip = fuse(l, self._loop_chain[i:], fusible)
        if not fusible:
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
        fused = []
        for base_loop, fuse_loop, fused_map, fused_inc_arg in fusible:
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
            if len(base_fundecl) != 1 or len(fuse_fundecl) != 1:
                raise RuntimeError("Fusing kernels, but found unexpected AST")
            base_fundecl = base_fundecl[0]
            fuse_fundecl = fuse_fundecl[0]

            # 1) Craft the /fusion/ kernel #

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

            # 2) Modify the /fuse/ kernel #
            # This is to take into account that many arguments are shared with
            # /base/, so they will only staged once for /base/. This requires
            # tweaking the way the arguments are declared and accessed in /fuse/
            # kernel. For example, the shared incremented array (called /buffer/
            # in the pseudocode in the comment above) now needs to take offsets
            # to be sure the locations that /base/ is supposed to increment are
            # actually accessed. The same concept apply to indirect arguments.
            init = lambda v: '{%s}' % ', '.join([str(j) for j in v])
            for i, fuse_args in enumerate(zip(fuse_loop.args, fuse_fundecl.args)):
                fuse_loop_arg, fuse_kernel_arg = fuse_args
                sym_id = fuse_kernel_arg.sym.symbol
                # 2A) Use temporaries to invoke the /fuse/ kernel
                buffer = '_%s' % fuse_kernel_arg.sym.symbol
                # 2B) How should I use the temporaries ?
                if fuse_loop_arg.access == INC:
                    op = ast.Incr
                    lvalue, rvalue = sym_id, buffer
                    extend_if_body = lambda body, block: body.children.extend(block)
                    buffer_decl = ast.Decl('%s' % fuse_loop_arg.ctype, ast.Symbol(buffer))
                elif fuse_loop_arg.access == READ:
                    op = ast.Assign
                    lvalue, rvalue = buffer, sym_id
                    extend_if_body = lambda body, block: \
                        [body.children.insert(0, b) for b in reversed(block)]
                    buffer_decl = ast.Decl('%s*' % fuse_loop_arg.ctype, ast.Symbol(buffer))
                # 2C) Now handle arguments depending on their type ...
                if fuse_loop_arg._is_mat:
                    # ... Handle Mats
                    staging = []
                    for b in fused_inc_arg._block_shape:
                        for rc in b:
                            lvalue = ast.Symbol(lvalue, ('i', 'i'),
                                                ((rc[0], 'j'), (rc[1], 'k')))
                            rvalue = ast.Symbol(rvalue, ('j', 'k'))
                            staging = ItSpace(mode=0).to_for([(0, rc[0]), (0, rc[1])],
                                                             ('j', 'k'),
                                                             [op(lvalue, rvalue)])[:1]
                    # Set up the temporary
                    buffer_decl.sym.rank = fuse_kernel_arg.sym.rank
                    if fuse_loop_arg.access == INC:
                        buffer_decl.init = ast.ArrayInit(init([init([0.0])]))
                elif fuse_loop_arg._is_indirect:
                    # ... Handle indirect arguments. At the C level, these arguments
                    # are of pointer type, so simple pointer arithmetic is used
                    # to ensure the kernel accesses are to the correct locations
                    fuse_arity = fuse_loop_arg.map.arity
                    base_arity = fuse_arity*fused_map.arity
                    cdim = fuse_loop_arg.data.dataset.cdim
                    size = fuse_arity*cdim
                    # Set the proper storage layout before invoking /fuse/
                    ofs_vals = [[base_arity*j + k for k in range(fuse_arity)]
                                for j in range(cdim)]
                    ofs_vals = [[fuse_arity*j + k for k in flatten(ofs_vals)]
                                for j in range(fused_map.arity)]
                    ofs_vals = list(flatten(ofs_vals))
                    ofs_idx_sym = 'v_ofs_%d' % i
                    body.children.insert(0, ast.Decl(
                        'int', ast.Symbol(ofs_idx_sym, (len(ofs_vals),)),
                        ast.ArrayInit(init(ofs_vals)), ['static', 'const']))
                    ofs_idx_syms = [ast.Symbol(ofs_idx_sym, ('i',), ((size, j),))
                                    for j in range(size)]
                    # Set up the temporary and stage data into it
                    buffer_decl.sym.rank = (size,)
                    if fuse_loop_arg.access == INC:
                        buffer_decl.init = ast.ArrayInit(init([0.0]))
                        staging = [op(ast.Symbol(lvalue, (k,)), ast.Symbol(rvalue, (j,)))
                                   for j, k in enumerate(ofs_idx_syms)]
                    else:
                        staging = [op(ast.Symbol(lvalue, (j,)), ast.Symbol(rvalue, (k,)))
                                   for j, k in enumerate(ofs_idx_syms)]
                else:
                    # Nothing special to do for direct arguments
                    continue
                # Update the If body to use the temporary
                extend_if_body(if_exec.children[0], staging)
                if_exec.children[0].children.insert(0, buffer_decl)
                fuse_funcall.children[fuse_loop.args.index(fuse_loop_arg)] = \
                    ast.Symbol(buffer)

            # 3) Create a /fusion.Kernel/ object to be used to update the schedule
            fused_headers = set([str(h) for h in base_headers + fuse_headers])
            fused_ast = ast.Root([ast.PreprocessNode(h) for h in fused_headers] +
                                 [base_fundecl, fuse_fundecl, fusion_fundecl])
            kernels = [base, fuse]
            loop_chain_index = (self._loop_chain.index(base_loop),
                                self._loop_chain.index(fuse_loop))
            fused.append((Kernel(kernels, fused_ast, loop_chain_index), fused_map))

        # Finally, generate a new schedule
        self._schedule = HardFusionSchedule(self._schedule, fused)
        self._loop_chain = self._schedule(self._loop_chain, only_hard=True)

    def _tile(self):
        """Tile consecutive loops over different iteration sets characterized
        by RAW and WAR dependencies. This requires interfacing with the SLOPE
        library."""

        def inspect_set(s, insp_sets, extra_halo):
            """Inspect the iteration set of a loop and store set info suitable
            for SLOPE in /insp_sets/. Further, check that such iteration set has
            a sufficiently depth halo region for correct execution in the case a
            SLOPE MPI backend is enabled."""
            # Get and format some iterset info
            superset, s_name = None, s.name
            if isinstance(s, Subset):
                superset = s.superset.name
                s_name = "%s_ss" % s.name
            # If not an MPI backend, return "standard" values for core, exec, and
            # non-exec regions (recall that SLOPE expects owned to be part of exec)
            if slope.get_exec_mode() not in ['OMP_MPI', 'ONLY_MPI']:
                infoset = s_name, s.core_size, s.exec_size - s.core_size, \
                    s.total_size - s.exec_size, superset

            else:
                if not hasattr(s, '_deep_size'):
                    raise RuntimeError("SLOPE backend (%s) requires deep halos",
                                       slope.get_exec_mode())
                # Assume [1, ..., N] levels of halo depth
                levelN = s._deep_size[-1] if not extra_halo else s._deep_size[-2]
                core_size = levelN[0]
                exec_size = levelN[2] - core_size
                nonexec_size = levelN[3] - levelN[2]
                infoset = s_name, core_size, exec_size, nonexec_size, superset
            insp_sets[infoset] = infoset
            return infoset

        tile_size = self._options.get('tile_size', 1)
        partitioning = self._options.get('partitioning', 'chunk')
        extra_halo = self._options.get('extra_halo', False)

        # The SLOPE inspector, which needs be populated with sets, maps,
        # descriptors, and loop chain structure
        inspector = slope.Inspector()

        # Build inspector and argument types and values
        # Note: we need ordered containers to be sure that SLOPE generates
        # identical code for all ranks
        arguments = []
        insp_sets, insp_maps, insp_loops = OrderedDict(), OrderedDict(), []
        for loop in self._loop_chain:
            slope_desc = set()
            # 1) Add sets
            iterset = loop.it_space.iterset
            iterset = iterset.subset if hasattr(iterset, 'subset') else iterset
            infoset = inspect_set(iterset, insp_sets, extra_halo)
            iterset_name, is_superset = infoset[0], infoset[4]
            # If iterating over a subset, we fake an indirect parloop from the
            # (iteration) subset to the superset. This allows the propagation of
            # tiling across the hierarchy of sets (see SLOPE for further info)
            if is_superset:
                inspect_set(iterset.superset, insp_sets, extra_halo)
                map_name = "%s_tosuperset" % iterset_name
                insp_maps[iterset_name] = (map_name, iterset_name,
                                           iterset.superset.name, iterset.indices)
                slope_desc.add((map_name, INC._mode))
            for a in loop.args:
                # 2) Add access descriptors
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
                            inspect_set(m.iterset, insp_sets, extra_halo)
                            inspect_set(m.toset, insp_sets, extra_halo)
            # 3) Add loop
            insp_loops.append((loop.kernel.name, iterset_name, list(slope_desc)))
        # Provide structure of loop chain to SLOPE
        arguments.extend([inspector.add_sets(insp_sets.values())])
        arguments.extend([inspector.add_maps(insp_maps.values())])
        inspector.add_loops(insp_loops)

        # Set a specific tile size
        arguments.extend([inspector.set_tile_size(tile_size)])

        # Tell SLOPE the rank of the MPI process
        arguments.extend([inspector.set_mpi_rank(MPI.comm.rank)])

        # Get type and value of additional arguments that SLOPE can exploit
        arguments.extend(inspector.add_extra_info())

        # Arguments types and values
        argtypes, argvalues = zip(*arguments)

        # Set a tile partitioning strategy
        inspector.set_partitioning(partitioning)

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
                  '-l%s' % slope.get_lib_name(),
                  '-L%s/lib' % os.environ['SLOPE_METIS'],
                  '-lmetis']

        # Compile and run inspector
        fun = compilation.load(src, "cpp", "inspector", cppargs, ldargs,
                               argtypes, rettype, compiler)
        inspection = fun(*argvalues)

        # Finally, get the Executor representation, to be used at executor
        # code generation time
        executor = slope.Executor(inspector)

        kernel = Kernel(tuple(loop.kernel for loop in self._loop_chain))
        self._schedule = TilingSchedule(kernel, self._schedule, inspection, executor)

    @property
    def mode(self):
        return self._mode

    @property
    def schedule(self):
        return self._schedule


# Loop fusion interface

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
    # If there is nothing to fuse, just return
    if len(loop_chain) in [0, 1]:
        return loop_chain

    # Are there _Assembly objects (i.e., synch points) preventing fusion?
    remainder = []
    synch_points = [l for l in loop_chain if isinstance(l, Mat._Assembly)]
    if synch_points:
        if len(synch_points) > 1:
            warning("Fusing loops and found more than one synchronization point")
        # Fuse only the sub-sequence before the first synch point
        synch_point = loop_chain.index(synch_points[0])
        remainder, loop_chain = loop_chain[synch_point:], loop_chain[:synch_point]

    # Get an inspector for fusing this /loop_chain/. If there's a cache hit,
    # return the fused par loops straight away. Otherwise, try to run an inspection.
    options = {
        'mode': kwargs.get('mode', 'hard'),
        'tile_size': kwargs.get('tile_size', 1),
        'partitioning': kwargs.get('partitioning', 'chunk'),
        'extra_halo': kwargs.get('extra_halo', False)
    }
    inspector = Inspector(name, loop_chain, **options)
    if inspector._initialized:
        return inspector.schedule(loop_chain) + remainder

    # Otherwise, is the inspection legal ?
    mode = kwargs.get('mode', 'hard')
    force_glb = kwargs.get('force_glb', False)

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

    # If tiling is requested, SLOPE must be visible
    if mode in ['tile', 'only_tile'] and not slope:
        return loop_chain + remainder

    schedule = inspector.inspect()
    return schedule(loop_chain) + remainder


@contextmanager
def loop_chain(name, **kwargs):
    """Analyze the sub-trace of loops lazily evaluated in this contextmanager ::

        [loop_0, loop_1, ..., loop_n-1]

    and produce a new sub-trace (``m <= n``) ::

        [fused_loops_0, fused_loops_1, ..., fused_loops_m-1, peel_loops]

    which is eventually inserted in the global trace of :class:`ParLoop` objects.

    That is, sub-sequences of :class:`ParLoop` objects are potentially replaced by
    new :class:`ParLoop` objects representing the fusion or the tiling of the
    original trace slice.

    :arg name: identifier of the loop chain
    :arg kwargs:
        * mode (default='tile'): the fusion/tiling mode (accepted: soft, hard,
            tile, only_tile)
        * tile_size: (default=1) suggest a starting average tile size
        * num_unroll (default=1): in a time stepping loop, the length of the loop
            chain is given by ``num_loops * num_unroll``, where ``num_loops`` is the
            number of loops per time loop iteration. Therefore, setting this value
            to a number >1 enables tiling longer chains.
        * force_glb (default=False): force tiling even in presence of global
            reductions. In this case, the user becomes responsible of semantic
            correctness.
        * partitioning (default='chunk'): select a partitioning mode for crafting
            tiles. The partitioning modes available are those accepted by SLOPE;
            refer to the SLOPE documentation for more info.
    """
    assert name != lazy_trace_name, "Loop chain name must differ from %s" % lazy_trace_name

    num_unroll = kwargs.setdefault('num_unroll', 1)
    tile_size = kwargs.setdefault('tile_size', 1)
    partitioning = kwargs.setdefault('partitioning', 'chunk')

    # Get a snapshot of the trace before new par loops are added within this
    # context manager
    from base import _trace
    stamp = list(_trace._trace)

    yield

    trace = _trace._trace
    if trace == stamp:
        return

    # What's the first item /B/ that appeared in the trace /before/ entering the
    # context manager and that still has to be executed ?
    # The loop chain will be (B, end_of_current_trace]
    bottom = 0
    for i in reversed(stamp):
        if i in trace:
            bottom = trace.index(i) + 1
            break
    extracted_trace = trace[bottom:]

    if num_unroll < 1:
        # No fusion, but openmp parallelization could still occur through SLOPE
        if slope and slope.get_exec_mode() in ['OMP', 'OMP_MPI'] and tile_size > 0:
            block_size = tile_size    # This is rather a 'block' size (no tiling)
            options = {'mode': 'only_omp',
                       'tile_size': block_size,
                       'partitioning': partitioning}
            new_trace = [Inspector(name, [loop], **options).inspect()([loop])
                         for loop in extracted_trace]
            trace[bottom:] = list(flatten(new_trace))
            _trace.evaluate_all()
        return

    # Unroll the loop chain /num_unroll/ times before fusion/tiling
    total_loop_chain = loop_chain.unrolled_loop_chain + extracted_trace
    if len(total_loop_chain) / len(extracted_trace) == num_unroll:
        bottom = trace.index(total_loop_chain[0])
        trace[bottom:] = fuse(name, total_loop_chain, **kwargs)
        loop_chain.unrolled_loop_chain = []
        # We can now force the evaluation of the trace. This frees resources.
        _trace.evaluate_all()
    else:
        loop_chain.unrolled_loop_chain.extend(extracted_trace)
loop_chain.unrolled_loop_chain = []

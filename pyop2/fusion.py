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

"""OP2 OpenMP backend for fused/tiled loops."""

from contextlib import contextmanager
from collections import OrderedDict
from copy import deepcopy as dcopy
import os

from base import _trace
from base import *
import openmp
import compilation
import host
from caching import Cached
from profiling import lineprof, timed_region, profile
from logger import warning
from mpi import collective
from op2 import par_loop
from utils import flatten, strip, as_tuple

import coffee
from coffee import base as coffee_ast
from coffee.utils import visit as coffee_ast_visit, \
    ast_update_id as coffee_ast_update_id

import slope_python as slope

# hard coded value to max openmp threads
_max_threads = 32


class Arg(openmp.Arg):

    @staticmethod
    def specialize(args, gtl_map, loop_id):
        """Given ``args`` instances of some :class:`fusion.Arg` superclass,
        create and return specialized :class:`fusion.Arg` objects.

        :param args: either a single :class:`host.Arg` object or an iterator
                     (accepted: list, tuple) of :class:`host.Arg` objects.
        :gtl_map: a dict associating global maps' names to local maps' c_names.
        :param loop_id: indicates the position of the args` loop in the loop
                        chain
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
            _arg._loop_position = loop_id
            _arg._position = arg._position
            _arg._indirect_position = arg._indirect_position
            _arg._c_local_maps = c_local_maps
            return _arg

        if isinstance(args, (list, tuple)):
            return [convert(arg, gtl_map, loop_id) for arg in args]
        return convert(args, gtl_map, loop_id)

    def c_arg_bindto(self, arg):
        """Assign c_pointer of this Arg to ``arg``."""
        if self.ctype != arg.ctype:
            raise RuntimeError("Cannot bind arguments having mismatching types")
        return "%s* %s = %s" % (self.ctype, self.c_arg_name(), arg.c_arg_name())

    def c_map_name(self, i, j):
        return self._c_local_maps[i][j]

    @property
    def name(self):
        """The generated argument name."""
        return "arg_exec_loop%d_%d" % (self._loop_position, self._position)


class Kernel(openmp.Kernel, tuple):

    @classmethod
    def _cache_key(cls, kernels, fuse=True):
        return "".join([super(Kernel, cls)._cache_key(k.code, k.name, k._opts,
                                                      k._include_dirs, k._headers,
                                                      k._user_code) for k in kernels])

    def _ast_to_c(self, asts, opts):
        """Fuse Abstract Syntax Trees of a collection of kernels and transform
        them into a string of C code."""
        asts = as_tuple(asts, (coffee_ast.FunDecl, coffee_ast.Root))

        if len(asts) == 1 or not opts['fuse']:
            self._ast = coffee_ast.Root(asts)
            return self._ast.gencode()

        # Fuse the actual kernels' bodies
        fused_ast = dcopy(asts[0])
        if not isinstance(fused_ast, coffee_ast.FunDecl):
            # Need to get the Function declaration, so inspect the children
            fused_ast = [n for n in fused_ast.children
                         if isinstance(n, coffee_ast.FunDecl)][0]
        for unique_id, _ast in enumerate(asts[1:], 1):
            ast = dcopy(_ast)
            # 1) Extend function name
            fused_ast.name = "%s_%s" % (fused_ast.name, ast.name)
            # 2) Concatenate the arguments in the signature
            fused_ast.args.extend(ast.args)
            # 3) Uniquify symbols identifiers
            ast_info = coffee_ast_visit(ast, None)
            ast_decls = ast_info['decls']
            ast_symbols = ast_info['symbols']
            for str_sym, decl in ast_decls.items():
                for symbol in ast_symbols.keys():
                    coffee_ast_update_id(symbol, str_sym, unique_id)
            # 4) Concatenate bodies
            marker_ast_node = coffee_ast.FlatBlock("\n\n// Begin of fused kernel\n\n")
            fused_ast.children[0].children.extend([marker_ast_node] + ast.children)

        self._ast = fused_ast
        return self._ast.gencode()

    def __init__(self, kernels, fuse=True):
        # Protect against re-initialization when retrieved from cache
        if self._initialized:
            return
        kernels = as_tuple(kernels, (Kernel, host.Kernel))
        self._kernels = kernels

        Kernel._globalcount += 1
        self._name = "_".join([kernel.name for kernel in kernels])
        self._opts = dict(flatten([kernel._opts.items() for kernel in kernels]))
        self._opts['fuse'] = fuse
        self._applied_blas = any(kernel._applied_blas for kernel in kernels)
        self._applied_ap = any(kernel._applied_ap for kernel in kernels)
        self._include_dirs = list(set(flatten([kernel._include_dirs for kernel
                                               in kernels])))
        self._headers = list(set(flatten([kernel._headers for kernel in kernels])))
        self._user_code = "\n".join([kernel._user_code for kernel in kernels])
        self._code = self._ast_to_c([kernel._ast for kernel in kernels], self._opts)
        self._initialized = True

    def __iter__(self):
        for kernel in self._kernels:
            yield kernel

    def __str__(self):
        return "OP2 FusionKernel: %s" % self._name


# Parallel loop API

class JITModule(openmp.JITModule):

    _cppargs = []
    _libraries = []
    _extension = 'cpp'

    _wrapper = """
extern "C" void %(wrapper_name)s(%(executor_arg)s,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s);
void %(wrapper_name)s(%(executor_arg)s,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s) {
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
  int i = %(tile_iter)s[%(index_expr)s];
  %(vec_inits)s;
  %(buffer_decl)s;
  %(buffer_gather)s
  %(kernel_name)s(%(kernel_args)s);
  %(layout_decl)s;
  %(layout_loop)s
      %(layout_assign)s;
  %(layout_loop_close)s
  %(itset_loop_body)s;
}
%(interm_globals_writeback)s;
"""

    @classmethod
    def _cache_key(cls, kernel, it_space, *args, **kwargs):
        key = (hash(kwargs['executor']),)
        all_args = kwargs['all_args']
        for kernel_i, it_space_i, args_i in zip(kernel, it_space, all_args):
            key += super(JITModule, cls)._cache_key(kernel_i, it_space_i, *args_i)
        return key

    def __init__(self, kernel, it_space, *args, **kwargs):
        if self._initialized:
            return
        self._all_args = kwargs.pop('all_args')
        self._executor = kwargs.pop('executor')
        super(JITModule, self).__init__(kernel, it_space, *args, **kwargs)

    def compile(self, argtypes=None, restype=None):
        if hasattr(self, '_fun'):
            # It should not be possible to pull a jit module out of
            # the cache /with/ arguments
            if hasattr(self, '_args'):
                raise RuntimeError("JITModule is holding onto args, memory leak!")
            self._fun.argtypes = argtypes
            self._fun.restype = restype
            return self._fun
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule not in cache, but has no args associated")

        # Prior to the instantiation and compilation of the JITModule, a fusion
        # kernel object needs be created. This is because the superclass' method
        # expects a single kernel, not a list as we have at this point.
        self._kernel = Kernel(self._kernel, fuse=False)
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
        fun = super(JITModule, self).compile(argtypes, restype)

        if hasattr(self, '_all_args'):
            # After the JITModule is compiled, can drop any reference to now
            # useless fields, which would otherwise cause memory leaks
            del self._all_args
            del self._executor

        return fun

    def generate_code(self):
        indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))
        code_dict = {}

        code_dict['wrapper_name'] = 'wrap_executor'
        code_dict['executor_arg'] = "%s %s" % (slope.Executor.meta['ctype_exec'],
                                               slope.Executor.meta['name_param_exec'])

        # Construct the wrapper
        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self._args])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self._args])
        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

        code_dict['wrapper_args'] = _wrapper_args
        code_dict['const_args'] = _const_args
        code_dict['wrapper_decs'] = indent(_wrapper_decs, 1)
        code_dict['const_inits'] = indent(_const_inits, 1)

        # Construct kernels invocation
        _loop_chain_body, _user_code, _ssinds_arg = [], [], []
        for i, loop in enumerate(zip(self._kernel, self._itspace, self._all_args)):
            kernel, it_space, args = loop

            # Obtain code_dicts of individual kernels, since these have pieces of
            # code that can be straightforwardly reused for this code generation
            loop_code_dict = host.JITModule(kernel, it_space, *args).generate_code()

            # Need to bind executor arguments to this kernel's arguments
            # Using a dict because need comparison on identity, not equality
            args_dict = dict(zip([_a.data for _a in self._args], self._args))
            binding = OrderedDict(zip(args, [args_dict[a.data] for a in args]))
            if len(binding) != len(args):
                raise RuntimeError("Tiling code gen failed due to args mismatching")
            binding = ';\n'.join([a0.c_arg_bindto(a1) for a0, a1 in binding.items()])

            loop_code_dict['args_binding'] = binding
            loop_code_dict['tile_iter'] = self._executor.gtl_maps[i]['DIRECT']
            loop_code_dict['tile_init'] = self._executor.c_loop_init[i]
            loop_code_dict['tile_start'] = slope.Executor.meta['tile_start']
            loop_code_dict['tile_end'] = slope.Executor.meta['tile_end']

            _loop_chain_body.append(strip(JITModule._kernel_wrapper % loop_code_dict))
            _user_code.append(kernel._user_code)
            _ssinds_arg.append(loop_code_dict['ssinds_arg'])
        _loop_chain_body = "\n\n".join(_loop_chain_body)
        _user_code = "\n".join(_user_code)
        _ssinds_arg = ", ".join([s for s in _ssinds_arg if s])

        code_dict['user_code'] = indent(_user_code, 1)
        code_dict['ssinds_arg'] = _ssinds_arg
        executor_code = indent(self._executor.c_code(indent(_loop_chain_body, 2)), 1)
        code_dict['executor_code'] = executor_code

        return code_dict


class ParLoop(openmp.ParLoop):

    def __init__(self, kernel, it_space, *args, **kwargs):
        read_args = [a.data for a in args if a.access in [READ, RW]]
        written_args = [a.data for a in args if a.access in [RW, WRITE, MIN, MAX, INC]]
        LazyComputation.__init__(self, set(read_args) | Const._defs, set(written_args))

        self._kernel = kernel
        self._actual_args = args
        self._it_space = it_space

        for i, arg in enumerate(self._actual_args):
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

        # These parameters are expected in a ParLoop based on tiling
        self._inspection = kwargs['inspection']
        self._all_args = kwargs['all_args']
        self._executor = kwargs['executor']

    @collective
    @profile
    def compute(self):
        """Execute the kernel over all members of the iteration space."""
        with timed_region("ParLoopChain: compute"):
            self._compute()

    @collective
    @lineprof
    def _compute(self):
        with timed_region("ParLoopChain: executor"):
            kwargs = {
                'all_args': self._all_args,
                'executor': self._executor,
            }
            fun = JITModule(self.kernel, self.it_space, *self.args, **kwargs)

            # Build restype, argtypes and argvalues
            self._restype = None
            self._argtypes = [slope.Executor.meta['py_ctype_exec']]
            self._jit_args = [self._inspection]
            for it_space in self.it_space:
                if isinstance(it_space._iterset, Subset):
                    self._argtypes.append(it_space._iterset._argtype)
                    self._jit_args.append(it_space._iterset._indices)
            for arg in self.args:
                if arg._is_mat:
                    self._argtypes.append(arg.data._argtype)
                    self._jit_args.append(arg.data.handle.handle)
                else:
                    for d in arg.data:
                        # Cannot access a property of the Dat or we will force
                        # evaluation of the trace
                        self._argtypes.append(d._argtype)
                        self._jit_args.append(d._data)

                if arg._is_indirect or arg._is_mat:
                    maps = as_tuple(arg.map, Map)
                    for map in maps:
                        for m in map:
                            self._argtypes.append(m._argtype)
                            self._jit_args.append(m.values_with_halo)

            for c in Const._definitions():
                self._argtypes.append(c._argtype)
                self._jit_args.append(c.data)

            # Compile and run the JITModule
            fun = fun.compile(argtypes=self._argtypes, restype=self._restype)


# Possible Schedules as produced by an Inspector

class Schedule(object):
    """Represent an execution scheme for a sequence of :class:`ParLoop` objects."""

    def __init__(self, kernel):
        self._kernel = kernel

    def __call__(self, loop_chain):
        """The argument ``loop_chain`` is a list of :class:`ParLoop` objects,
        which is expected to be mapped onto an optimized scheduling.

        In the simplest case, this Schedule's kernels exactly match the :class:`Kernel`
        objects in ``loop_chain``; the default PyOP2 execution model should then be
        used, and an unmodified ``loop_chain`` therefore be returned.

        In other scenarios, this Schedule's kernels could represent the fused
        version, or the tiled version, of the provided ``loop_chain``; a sequence
        of new :class:`ParLoop` objects using the fused/tiled kernels should be
        returned.
        """
        raise NotImplementedError("Subclass must implement ``__call__`` method")


class PlainSchedule(Schedule):

    def __init__(self):
        super(PlainSchedule, self).__init__([])

    def __call__(self, loop_chain):
        return loop_chain


class FusionSchedule(Schedule):
    """Schedule for a sequence of soft/hard fused :class:`ParLoop` objects."""

    def __init__(self, kernel, ranges):
        super(FusionSchedule, self).__init__(kernel)
        self._ranges = ranges

    def __call__(self, loop_chain):
        offset = 0
        fused_par_loops = []
        for kernel, range in zip(self._kernel, self._ranges):
            iterset = loop_chain[offset].it_space.iterset
            args = flatten([loop.args for loop in loop_chain[offset:range]])
            fused_par_loops.append(par_loop(kernel, iterset, *args))
            offset = range
        return fused_par_loops


class TilingSchedule(Schedule):
    """Schedule for a sequence of tiled :class:`ParLoop` objects."""

    def __init__(self, schedule, inspection, executor):
        self._schedule = schedule
        self._inspection = inspection
        self._executor = executor

    def _filter_args(self, loop_chain):
        """Uniquify arguments and access modes"""
        args = OrderedDict()
        for loop in loop_chain:
            # 1) Analyze the Args in each loop composing the chain and produce a
            # new sequence of Args for the tiled ParLoop. For example, consider
            # Arg X, and be X.DAT written to in ParLoop_0 (access mode WRITE) and
            # read from in ParLoop_1 (access mode READ); this means that in the
            # tiled ParLoop, X will have access mode RW
            for a in loop.args:
                args[a.data] = args.get(a.data, a)
                if a.access != args[a.data].access:
                    if READ in [a.access, args[a.data].access]:
                        # If a READ and some sort of write (MIN, MAX, RW, WRITE,
                        # INC), then the access mode becomes RW
                        args[a.data]._access = RW
                    elif WRITE in [a.access, args[a.data].access]:
                        # Can't be a READ, so just stick to WRITE regardless of what
                        # the other access mode is
                        args[a.data]._access = WRITE
                    else:
                        # Neither READ nor WRITE, so access modes are some
                        # combinations of RW, INC, MIN, MAX. For simplicity,
                        # just make it RW.
                        args[a.data]._access = RW
        return args.values()

    def __call__(self, loop_chain):
        loop_chain = self._schedule(loop_chain)
        args = self._filter_args(loop_chain)
        kernel = tuple((loop.kernel for loop in loop_chain))
        all_args = tuple((Arg.specialize(loop.args, gtl_map, i) for i, (loop, gtl_map)
                         in enumerate(zip(loop_chain, self._executor.gtl_maps))))
        it_space = tuple((loop.it_space for loop in loop_chain))
        kwargs = {
            'inspection': self._inspection,
            'all_args': all_args,
            'executor': self._executor
        }
        return [ParLoop(kernel, it_space, *args, **kwargs)]


# Loop chain inspection

class Inspector(Cached):
    """An inspector is used to fuse or tile a sequence of :class:`ParLoop` objects.

    The inspector is implemented by the SLOPE library, which the user makes
    visible by setting the environment variable ``SLOPE_DIR`` to the value of
    the root SLOPE directory."""

    _cache = {}
    _modes = ['soft', 'hard', 'tile']

    @classmethod
    def _cache_key(cls, name, loop_chain, tile_size):
        key = (name, tile_size)
        for loop in loop_chain:
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
        """Inspect this Inspector's loop chain and produce a Schedule object.

        :param mode: can take any of the values in ``Inspector._modes``, namely
                     ``soft``, ``hard``, and ``tile``. If ``soft`` is specified,
                     only soft fusion takes place; that is, only consecutive loops
                     over the same iteration set that do not present RAW or WAR
                     dependencies through indirections are fused. If ``hard`` is
                     specified, then first ``soft`` is applied, followed by fusion
                     of loops over different iteration sets, provided that RAW or
                     WAR dependencies are not present. If ``tile`` is specified,
                     than tiling through the SLOPE library takes place just after
                     ``soft`` and ``hard`` fusion.
        """
        self._inspected += 1
        if self._heuristic_skip_inspection():
            # Heuristically skip this inspection if there is a suspicion the
            # overhead is going to be too much; for example, when the loop
            # chain could potentially be execution only once or a few time.
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
                raise RuntimeError("Cached Inspector's mode doesn't match")
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
        mode = Inspector._modes.index(mode)

        with timed_region("ParLoopChain `%s`: inspector" % self._name):
            self._soft_fuse()
            if mode > 0:
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

    def _heuristic_skip_inspection(self):
        """Decide heuristically whether to run an inspection or not."""
        # At the moment, a simple heuristic is used: if the inspection is
        # requested more than once, then it is performed
        if self._inspected < 2:
            return True
        return False

    def _soft_fuse(self):
        """Fuse consecutive loops over the same iteration set by concatenating
        kernel bodies and creating new :class:`ParLoop` objects representing
        the fused sequence.

        The conditions under which two loops over the same iteration set are
        hardly fused are:

            * They are both direct, OR
            * One is direct and the other indirect

        This is detailed in the paper::

            "Mesh Independent Loop Fusion for Unstructured Mesh Applications"

        from C. Bertolli et al.
        """
        fuse = lambda fusing: par_loop(Kernel([l.kernel for l in fusing]),
                                       fusing[0].it_space.iterset,
                                       *flatten([l.args for l in fusing]))

        fused, fusing = [], [self._loop_chain[0]]
        for i, loop in enumerate(self._loop_chain[1:]):
            base_loop = fusing[-1]
            if base_loop.it_space != loop.it_space or \
                    (base_loop.is_indirect and loop.is_indirect):
                # Fusion not legal
                fused.append((fuse(fusing), i+1))
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
            fused.append((fuse(fusing), len(self._loop_chain)))

        fused_loops, offsets = zip(*fused)
        self._loop_chain = fused_loops
        self._schedule = FusionSchedule([l.kernel for l in fused_loops], offsets)

    def _tile(self):
        """Tile consecutive loops over different iteration sets characterized
        by RAW and WAR dependencies. This requires interfacing with the SLOPE
        library."""
        inspector = slope.Inspector('OMP')

        # Build arguments types and values
        arguments = []
        insp_sets, insp_maps, insp_loops = set(), {}, []
        for loop in self._loop_chain:
            slope_desc = set()
            # Add sets
            insp_sets.add((loop.it_space.name, loop.it_space.core_size))
            for a in loop.args:
                maps = as_tuple(a.map, Map)
                # Add maps (there can be more than one per argument if the arg
                # is actually a Mat - in which case there are two maps - or if
                # a MixedMap) and relative descriptors
                if not maps:
                    slope_desc.add(('DIRECT', a.access._mode))
                    continue
                for i, map in enumerate(maps):
                    for j, m in enumerate(map):
                        map_name = "%s%d_%d" % (m.name, i, j)
                        insp_maps[m.name] = (map_name, m.iterset.name,
                                             m.toset.name, m.values)
                        slope_desc.add((map_name, a.access._mode))
            # Add loop
            insp_loops.append((loop.kernel.name, loop.it_space.name, list(slope_desc)))
        # Provide structure of loop chain to the SLOPE's inspector
        arguments.extend([inspector.add_sets(insp_sets)])
        arguments.extend([inspector.add_maps(insp_maps.values())])
        inspector.add_loops(insp_loops)
        # Get type and value of any additional arguments that the SLOPE's inspector
        # expects
        arguments.extend([inspector.set_external_dats()])

        # Set a specific tile size
        arguments.extend([inspector.set_tile_size(self._tile_size)])

        # Arguments types and values
        argtypes, argvalues = zip(*arguments)

        # Generate inspector C code
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

        # Finally, get the Executor representation, to be used at executor's
        # code generation time
        executor = slope.Executor(inspector)

        self._schedule = TilingSchedule(self._schedule, inspection, executor)

    @property
    def mode(self):
        return self._mode


# Interface for triggering loop fusion

def reschedule_loops(name, loop_chain, tile_size, mode='tile'):
    """Given a list of :class:`ParLoop` in ``loop_chain``, return a list of new
    :class:`ParLoop` objects implementing an optimized scheduling of the loop chain.

    .. note:: The unmodified loop chain is instead returned if any of these
    conditions verify:

        * a global reduction is present;
        * at least one loop iterates over an extruded set
    """
    # Loop fusion is performed through the SLOPE library, which must be accessible
    # by reading the environment variable SLOPE_DIR
    try:
        os.environ['SLOPE_DIR']
    except KeyError:
        warning("Set the env variable SLOPE_DIR to the location of SLOPE")
        warning("Loops won't be fused, and plain pyop2.ParLoops will be executed")
        return loop_chain

    # If there are global reduction or extruded sets are present, return
    if any([l._reduced_globals for l in loop_chain]) or \
            any([l.is_layered for l in loop_chain]):
        return loop_chain

    # Get an inspector for fusing this loop_chain, possibly retrieving it from
    # the cache, and obtain the fused ParLoops through the schedule it produces
    inspector = Inspector(name, loop_chain, tile_size)
    schedule = inspector.inspect(mode)
    return schedule(loop_chain)


@contextmanager
def loop_chain(name, time_unroll=1, tile_size=0):
    """Analyze the sub-trace of loops lazily evaluated in this contextmanager ::

        [loop_0, loop_1, ..., loop_n-1]

    and produce a new sub-trace (``m <= n``) ::

        [fused_loops_0, fused_loops_1, ..., fused_loops_m-1, peel_loops]

    which is eventually inserted in the global trace of :class:`ParLoop` objects.

    That is, sub-sequences of :class:`ParLoop` objects are potentially replaced by
    new :class:`ParLoop` objects representing the fusion or the tiling of the
    original trace slice.

    :param name: identifier of the loop chain
    :param time_unroll: in a time stepping loop, the length of the loop chain
                        is given by ``num_loops * time_unroll``, where ``num_loops``
                        is the number of loops per time loop iteration. Therefore,
                        setting this value to a number greater than 1 enables
                        fusing/tiling longer loop chains (optional, defaults to 1).
    :param tile_size: suggest a tile size in case loop tiling is used (optional).
    """
    trace = _trace._trace
    stamp = trace[-1:]

    yield

    if time_unroll < 1:
        return

    start_point = trace.index(stamp[0])+1 if stamp else 0
    extracted_loop_chain = trace[start_point:]

    # Unroll the loop chain ``time_unroll`` times before fusion/tiling
    total_loop_chain = loop_chain.unrolled_loop_chain + extracted_loop_chain
    if len(total_loop_chain) / len(extracted_loop_chain) == time_unroll:
        trace[start_point:] = reschedule_loops(name, total_loop_chain, tile_size)
        loop_chain.unrolled_loop_chain = []
    else:
        loop_chain.unrolled_loop_chain.extend(total_loop_chain)
loop_chain.unrolled_loop_chain = []

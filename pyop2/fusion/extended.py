# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2016, Imperial College London and
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

"""Classes for fusing parallel loops and for executing fused parallel loops,
derived from ``base.py``."""

import sys
import ctypes
from copy import deepcopy as dcopy
from itertools import groupby
from collections import OrderedDict
from hashlib import md5

import pyop2.base as base
import pyop2.sequential as sequential
import pyop2.host as host
from pyop2.utils import flatten, strip, as_tuple
from pyop2.mpi import collective
from pyop2.profiling import timed_region

from pyop2.fusion.interface import slope, lazy_trace_name

import coffee
from coffee import base as ast
from coffee.visitors import FindInstances


class FusionArg(sequential.Arg):

    """An Arg specialized for kernels and loops subjected to any kind of fusion."""

    def __init__(self, arg, gather=None, c_index=False):
        """Initialize a :class:`FusionArg`.

        :arg arg: a supertype of :class:`FusionArg`, from which this Arg is derived.
        :arg gather: recognized values: ``postponed``, ``onlymap``. With ``postponed``,
            the gather is performed in a callee of the wrapper function; with
            ``onlymap``, the gather is performed as usual in the wrapper, but
            only the map values are staged.
        :arg c_index: if True, will provide the kernel with the iteration index of this
            Arg's set. Otherwise, code generation is unaffected.
        """
        super(FusionArg, self).__init__(arg.data, arg.map, arg.idx, arg.access, arg._flatten)
        self.gather = gather or arg.gather
        self.c_index = c_index or arg.c_index

    def c_map_name(self, i, j, fromvector=False):
        map_name = super(FusionArg, self).c_map_name(i, j)
        return map_name if not fromvector else "&%s[0]" % map_name

    def c_vec_dec(self, is_facet=False):
        if self.gather == 'onlymap':
            facet_mult = 2 if is_facet else 1
            cdim = self.data.cdim if self._flatten else 1
            return "%(type)s %(vec_name)s[%(arity)s];\n" % \
                {'type': self.ctype,
                 'vec_name': self.c_vec_name(),
                 'arity': self.map.arity * cdim * facet_mult}
        else:
            return super(FusionArg, self).c_vec_dec(is_facet)

    def c_vec_init(self, is_top, is_facet=False, force_gather=False):
        if self.gather == 'postponed' and not force_gather:
            return ''
        elif self.gather == 'onlymap':
            vec_name = self.c_vec_name()
            map_name = self.c_map_name(0, 0)
            arity = self.map.arity
            return ';\n'.join(["%s[%s] = %s[%s*%s+%s]" %
                               (vec_name, i, map_name, self.c_def_index(), arity, i)
                               for i in range(self.map.arity)])
        else:
            return super(FusionArg, self).c_vec_init(is_top, is_facet)

    def c_kernel_arg(self, count, i=0, j=0, shape=(0,), layers=1):
        if self.gather == 'postponed':
            if self._is_indirect:
                c_args = "%s, %s" % (self.c_arg_name(i),
                                     self.c_map_name(i, 0, self.c_map_is_vector()))
            else:
                c_args = self.c_arg_name(i)
        elif self.gather == 'onlymap':
            c_args = "%s, %s" % (self.c_arg_name(i), self.c_vec_name())
        else:
            c_args = super(FusionArg, self).c_kernel_arg(count, i, j, shape, layers)
        if self.c_index:
            c_args += ", %s" % self.c_def_index()
        return c_args

    def c_def_index(self):
        return 'i'

    def c_map_is_vector(self):
        return False


class TilingArg(FusionArg):

    """An Arg specialized for kernels and loops subjected to tiling."""

    def __init__(self, arg, loop_position, gtl_maps=None):
        """Initialize a :class:`TilingArg`.

        :arg arg: a supertype of :class:`TilingArg`, from which this Arg is derived.
        :arg loop_position: the position of the loop in the loop chain that this
            object belongs to.
        :arg gtl_maps: a dict associating global map names to local map names.
        """
        super(TilingArg, self).__init__(arg)
        self.position = arg.position
        self.indirect_position = arg.indirect_position
        self.loop_position = loop_position

        c_local_maps = None
        maps = as_tuple(arg.map, base.Map)
        if gtl_maps:
            c_local_maps = [None]*len(maps)
            for i, map in enumerate(maps):
                c_local_maps[i] = [None]*len(map)
                for j, m in enumerate(map):
                    c_local_maps[i][j] = gtl_maps["%s%d_%d" % (m.name, i, j)]
        self._c_local_maps = c_local_maps

    def c_arg_bindto(self):
        """Assign this Arg's c_pointer to ``arg``."""
        return "%s* %s = %s" % (self.ctype, self.c_arg_name(), self.ref_arg.c_arg_name())

    def c_ind_data(self, idx, i, j=0, is_top=False, offset=None, var=None):
        if not var:
            var = 'i' if not self._c_local_maps else 'n'
        return super(TilingArg, self).c_ind_data(idx, i, j, is_top, offset, var)

    def c_map_name(self, i, j, fromvector=False):
        if not self._c_local_maps:
            map_name = host.Arg.c_map_name(self.ref_arg, i, j)
        else:
            map_name = self._c_local_maps[i][j]
        return map_name if not fromvector else "&%s[0]" % map_name

    def c_map_entry(self, var):
        maps = []
        for idx in range(self.map.arity):
            maps.append("%(map_name)s[%(var)s * %(arity)d + %(idx)d]" % {
                'map_name': self.c_map_name(0, 0),
                'var': var,
                'arity': self.map.arity,
                'idx': idx
            })
        return maps

    def c_vec_entry(self, var, only_base=False):
        vecs = []
        for idx in range(self.map.arity):
            for k in range(self.data.cdim):
                vecs.append(self.c_ind_data(idx, 0, k, var=var))
                if only_base:
                    break
        return vecs

    def c_global_reduction_name(self, count=None):
        return "%(name)s_l%(count)d[0]" % {
            'name': self.c_arg_name(),
            'count': count}

    def c_def_index(self):
        return 'i' if not self._c_local_maps else 'n'

    def c_map_is_vector(self):
        return False if not self._c_local_maps else True

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
        key = str(loop_chain_index)
        key += "".join([k.cache_key for k in kernels])
        key += str(hash(str(fused_ast)))
        return md5(key).hexdigest()

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
            if main._ast:
                main_ast = dcopy(main._ast)
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

        # Tiled kernels are C++, and C++ compilers don't recognize /restrict/
        code = """
#define restrict __restrict

%s
""" % code

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
        self._include_dirs = list(set(flatten([k._include_dirs for k in kernels])))
        self._ldargs = list(set(flatten([k._ldargs for k in kernels])))
        self._headers = list(set(flatten([k._headers for k in kernels])))
        self._user_code = "\n".join(list(set([k._user_code for k in kernels])))
        self._attached_info = {'fundecl': None, 'attached': False}

        # What sort of Kernel do I have?
        if fused_ast:
            # A single AST (as a result of soft or hard fusion)
            self._ast = fused_ast
            self._code = self._ast_to_c(fused_ast)
        else:
            # Multiple functions (AST or strings, as a result of tiling)
            self._ast = None
            self._code = self._multiple_ast_to_c(kernels)
        self._kernels = kernels

        self._initialized = True

    def __iter__(self):
        for k in self._kernels:
            yield k

    def __str__(self):
        return "OP2 FusionKernel: %s" % self._name


# API for fused parallel loops

class ParLoop(sequential.ParLoop):

    """The root class of non-sequential parallel loops."""

    pass


class FusionParLoop(ParLoop):

    def __init__(self, kernel, iterset, *args, **kwargs):
        self._it_space = kwargs['it_space']
        super(FusionParLoop, self).__init__(kernel, iterset, *args, **kwargs)

    def _build_itspace(self, iterset):
        """
        Bypass the construction of a new iteration space.

        This avoids type checking in base.ParLoop._build_itspace, which would
        return an error when the fused loop accesses arguments that are not
        accessed by the base loop.
        """
        return self._it_space


# API for tiled parallel loops

class TilingIterationSpace(base.IterationSpace):

    """A simple bag of :class:`IterationSpace` objects for a sequence of tiled
    parallel loops."""

    def __init__(self, all_itspaces):
        self._iterset = [i._iterset for i in all_itspaces]
        self._extents = [i._extents for i in all_itspaces]
        self._block_shape = [i._block_shape for i in all_itspaces]
        assert all(all_itspaces[0].comm == i.comm for i in all_itspaces)
        self.comm = all_itspaces[0].comm

    def __str__(self):
        output = "OP2 Fused Iteration Space:"
        output += "\n  ".join(["%s with extents %s" % (i._iterset, i._extents)
                               for i in self.iterset])
        return output

    def __repr__(self):
        return "\n".join(["IterationSpace(%r, %r)" % (i._iterset, i._extents)
                          for i in self.iterset])


class TilingJITModule(sequential.JITModule):

    """A special :class:`JITModule` for a sequence of tiled kernels."""

    _cppargs = ['-fpermissive']
    _libraries = []
    _extension = 'cpp'

    _wrapper = """
extern "C" void %(wrapper_name)s(%(executor_arg)s,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(rank)s
                      %(region_flag)s);
void %(wrapper_name)s(%(executor_arg)s,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(rank)s
                      %(region_flag)s) {
  %(user_code)s
  %(wrapper_decs)s;

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
  %(prefetch_maps)s;
  %(vec_inits)s;
  %(prefetch_vecs)s;
  %(buffer_decl)s;
  %(buffer_gather)s
  %(kernel_name)s(%(kernel_args)s);
  i = %(index_expr)s;
  %(itset_loop_body)s;
}
%(tile_finish)s;
%(interm_globals_writeback)s;
"""

    @classmethod
    def _cache_key(cls, kernel, itspace, *args, **kwargs):
        insp_name = kwargs['insp_name']
        key = (insp_name, kwargs['use_glb_maps'], kwargs['use_prefetch'])
        if insp_name != lazy_trace_name:
            return key
        all_kernels = kwargs['all_kernels']
        all_itspaces = kwargs['all_itspaces']
        all_args = kwargs['all_args']
        for kernel, itspace, args in zip(all_kernels, all_itspaces, all_args):
            key += super(TilingJITModule, cls)._cache_key(kernel, itspace, *args)
        return key

    def __init__(self, kernel, itspace, *args, **kwargs):
        if self._initialized:
            return
        self._all_kernels = kwargs.pop('all_kernels')
        self._all_itspaces = kwargs.pop('all_itspaces')
        self._all_args = kwargs.pop('all_args')
        self._executor = kwargs.pop('executor')
        self._use_glb_maps = kwargs.pop('use_glb_maps')
        self._use_prefetch = kwargs.pop('use_prefetch')
        super(TilingJITModule, self).__init__(kernel, itspace, *args, **kwargs)

    def set_argtypes(self, iterset, *args):
        argtypes = [slope.Executor.meta['py_ctype_exec']]
        for itspace in self._all_itspaces:
            if isinstance(itspace.iterset, base.Subset):
                argtypes.append(itspace.iterset._argtype)
        for arg in args:
            if arg._is_mat:
                argtypes.append(arg.data._argtype)
            else:
                for d in arg.data:
                    argtypes.append(d._argtype)
            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, base.Map)
                for map in maps:
                    for m in map:
                        argtypes.append(m._argtype)

        # MPI related stuff (rank, region)
        argtypes.append(ctypes.c_int)
        argtypes.append(ctypes.c_int)

        self._argtypes = argtypes

    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule not in cache, but has no args associated")

        # Set compiler and linker options
        self._kernel._name = 'executor'
        self._kernel._headers.extend(slope.Executor.meta['headers'])
        if self._use_prefetch:
            self._kernel._headers.extend(['#include "xmmintrin.h"'])
        self._kernel._include_dirs.extend(['%s/include/SLOPE' % sys.prefix])
        self._libraries += ['-L%s/lib' % sys.prefix, '-l%s' % slope.get_lib_name()]
        compiler = coffee.system.compiler.get('name')
        self._cppargs += slope.get_compile_opts(compiler)
        fun = super(TilingJITModule, self).compile()

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

        # 1) Construct the wrapper arguments
        code_dict = {}
        code_dict['wrapper_name'] = 'wrap_executor'
        code_dict['executor_arg'] = "%s %s" % (slope.Executor.meta['ctype_exec'],
                                               slope.Executor.meta['name_param_exec'])
        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self._args])
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in self._args])
        code_dict['wrapper_args'] = _wrapper_args
        code_dict['wrapper_decs'] = indent(_wrapper_decs, 1)
        code_dict['rank'] = ", %s %s" % (slope.Executor.meta['ctype_rank'],
                                         slope.Executor.meta['rank'])
        code_dict['region_flag'] = ", %s %s" % (slope.Executor.meta['ctype_region_flag'],
                                                slope.Executor.meta['region_flag'])

        # 2) Construct the kernel invocations
        _loop_body, _user_code, _ssinds_arg = [], [], []
        # For each kernel ...
        for i, (kernel, it_space, args) in enumerate(zip(self._all_kernels,
                                                         self._all_itspaces,
                                                         self._all_args)):
            # ... bind the Executor's arguments to this kernel's arguments
            binding = []
            for a1 in args:
                for a2 in self._args:
                    if a1.data is a2.data and a1.map is a2.map:
                        a1.ref_arg = a2
                        break
                binding.append(a1.c_arg_bindto())
            binding = ";\n".join(binding)

            # ... obtain the /code_dict/ as if it were not part of an Executor,
            # since bits of code generation can be reused
            loop_code_dict = sequential.JITModule(kernel, it_space, *args, delay=True)
            loop_code_dict = loop_code_dict.generate_code()

            # ... does the scatter use global or local maps ?
            if self._use_glb_maps:
                loop_code_dict['index_expr'] = '%s[n]' % self._executor.gtl_maps[i]['DIRECT']
                prefetch_var = 'int p = %s[n + %d]' % (self._executor.gtl_maps[i]['DIRECT'],
                                                       self._use_prefetch)
            else:
                prefetch_var = 'int p = n + %d' % self._use_prefetch

            # ... add prefetch intrinsics, if requested
            prefetch_maps, prefetch_vecs = '', ''
            if self._use_prefetch:
                prefetch = lambda addr: '_mm_prefetch ((char*)(%s), _MM_HINT_T0)' % addr
                prefetch_maps = [a.c_map_entry('p') for a in args if a._is_indirect]
                # can save some instructions since prefetching targets chunks of 32 bytes
                prefetch_maps = flatten([j for j in pm if pm.index(j) % 2 == 0]
                                        for pm in prefetch_maps)
                prefetch_maps = list(OrderedDict.fromkeys(prefetch_maps))
                prefetch_maps = ';\n'.join([prefetch_var] +
                                           [prefetch('&(%s)' % pm) for pm in prefetch_maps])
                prefetch_vecs = flatten(a.c_vec_entry('p', True) for a in args
                                        if a._is_indirect)
                prefetch_vecs = ';\n'.join([prefetch(pv) for pv in prefetch_vecs])
            loop_code_dict['prefetch_maps'] = prefetch_maps
            loop_code_dict['prefetch_vecs'] = prefetch_vecs

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
            loop_code_dict['tile_finish'] = self._executor.c_loop_end[i]
            loop_code_dict['tile_start'] = slope.Executor.meta['tile_start']
            loop_code_dict['tile_end'] = slope.Executor.meta['tile_end']
            loop_code_dict['tile_iter'] = '%s[n]' % self._executor.gtl_maps[i]['DIRECT']
            if _ssind_arg:
                loop_code_dict['tile_iter'] = '%s[%s]' % (_ssind_arg, loop_code_dict['tile_iter'])

            # ... concatenate the rest, i.e., body, user code, ...
            _loop_body.append(strip(TilingJITModule._kernel_wrapper % loop_code_dict))
            _user_code.append(kernel._user_code)
            _ssinds_arg.append(_ssind_decl)

        _loop_chain_body = indent("\n\n".join(_loop_body), 2)
        code_dict['user_code'] = indent("\n".join(_user_code), 1)
        code_dict['ssinds_arg'] = "".join(["%s," % s for s in _ssinds_arg if s])
        code_dict['executor_code'] = indent(self._executor.c_code(_loop_chain_body), 1)

        return code_dict


class TilingParLoop(ParLoop):

    """A special :class:`ParLoop` for a sequence of tiled kernels."""

    def __init__(self, kernel, it_space, *args, **kwargs):
        base.LazyComputation.__init__(self,
                                      kwargs['read_args'],
                                      kwargs['written_args'],
                                      kwargs['inc_args'])

        # Inspector related stuff
        self._all_kernels = kwargs.get('all_kernels', [kernel])
        self._all_itspaces = kwargs.get('all_itspaces', [kernel])
        self._all_args = kwargs.get('all_args', [args])
        self._insp_name = kwargs.get('insp_name')
        self._inspection = kwargs.get('inspection')
        # Executor related stuff
        self._executor = kwargs.get('executor')
        self._use_glb_maps = kwargs.get('use_glb_maps')
        self._use_prefetch = kwargs.get('use_prefetch')

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
            if isinstance(itspace._iterset, base.Subset):
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
                maps = as_tuple(arg.map, base.Map)
                for map in maps:
                    for m in map:
                        arglist.append(m._values.ctypes.data)

        arglist.append(self.it_space.comm.rank)

        return arglist

    @collective
    def compute(self):
        """Execute the kernel over all members of the iteration space."""
        with timed_region("ParLoopChain: executor (%s)" % self._insp_name):
            self.halo_exchange_begin()
            kwargs = {
                'all_kernels': self._all_kernels,
                'all_itspaces': self._all_itspaces,
                'all_args': self._all_args,
                'executor': self._executor,
                'insp_name': self._insp_name,
                'use_glb_maps': self._use_glb_maps,
                'use_prefetch': self._use_prefetch
            }
            fun = TilingJITModule(self.kernel, self.it_space, *self.args, **kwargs)
            arglist = self.prepare_arglist(None, *self.args)
            self._compute(0, fun, *arglist)
            self.halo_exchange_end()
            self._compute(1, fun, *arglist)
            # Only meaningful if the user is enforcing tiling in presence of
            # global reductions
            self.reduction_begin()
            self.reduction_end()
            self.update_arg_data_state()

    @collective
    def _compute(self, part, fun, *arglist):
        with timed_region("ParLoopCKernel"):
            fun(*(arglist + (part,)))

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

"""OP2 sequential backend."""

import numpy as np
import petsc

from exceptions import *
from utils import *
import op_lib_core as core
from pyop2.utils import OP2_INC, OP2_LIB
import runtime_base as rt
from runtime_base import *

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""
    ParLoop(kernel, it_space, *args).compute()

class Dat(Dat):

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        self._data += as_type(other.data, self.dtype)
        return self

    def __isub__(self, other):
        """Pointwise subtraction of fields."""
        self._data -= as_type(other.data, self.dtype)
        return self

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        if np.isscalar(other):
            self._data *= as_type(other, self.dtype)
        else:
            self._data *= as_type(other.data, self.dtype)
        return self

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        if np.isscalar(other):
            self._data /= as_type(other, self.dtype)
        else:
            self._data /= as_type(other.data, self.dtype)
        return self

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.linalg.norm(self._data)

class ParLoop(rt.ParLoop):
    def compute(self):
        _fun = self.generate_code()
        _args = [self._it_space.size]
        for arg in self.args:
            if arg._is_mat:
                _args.append(arg.data.handle.handle)
            else:
                _args.append(arg.data.data)

            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

            if arg._is_indirect or arg._is_mat:
                maps = as_tuple(arg.map, Map)
                for map in maps:
                    _args.append(map.values)

        for c in Const._definitions():
            _args.append(c.data)

        _fun(*_args)

    def generate_code(self):

        key = self._cache_key
        _fun = rt._parloop_cache.get(key)

        if _fun is not None:
            return _fun

        from instant import inline_with_numpy

        def c_arg_name(arg):
            name = arg.data.name
            if arg._is_indirect and not (arg._is_vec_map or arg._uses_itspace):
                name += str(arg.idx)
            return name

        def c_vec_name(arg):
            return c_arg_name(arg) + "_vec"

        def c_map_name(arg):
            return c_arg_name(arg) + "_map"

        def c_wrapper_arg(arg):
            val = "PyObject *_%(name)s" % {'name' : c_arg_name(arg) }
            if arg._is_indirect or arg._is_mat:
                val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)}
                maps = as_tuple(arg.map, Map)
                if len(maps) is 2:
                    val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)+'2'}
            return val

        def c_wrapper_dec(arg):
            if arg._is_mat:
                val = "Mat %(name)s = (Mat)((uintptr_t)PyLong_AsUnsignedLong(_%(name)s))" % \
                     { "name": c_arg_name(arg) }
            else:
                val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
                  {'name' : c_arg_name(arg), 'type' : arg.ctype}
            if arg._is_indirect or arg._is_mat:
                val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                       {'name' : c_map_name(arg)}
            if arg._is_mat:
                val += ";\nint *%(name)s2 = (int *)(((PyArrayObject *)_%(name)s2)->data)" % \
                           {'name' : c_map_name(arg)}
            if arg._is_vec_map:
                val += ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
                       {'type' : arg.ctype,
                        'vec_name' : c_vec_name(arg),
                        'dim' : arg.map.dim}
            return val

        def c_ind_data(arg, idx):
            return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                    {'name' : c_arg_name(arg),
                     'map_name' : c_map_name(arg),
                     'map_dim' : arg.map.dim,
                     'idx' : idx,
                     'dim' : arg.data.cdim}

        def c_kernel_arg(arg):
            if arg._uses_itspace:
                if arg._is_mat:
                    name = "p_%s" % c_arg_name(arg)
                    if arg.data._is_vector_field:
                        return name
                    elif arg.data._is_scalar_field:
                        idx = ''.join(["[i_%d]" % i for i, _ in enumerate(arg.data.dims)])
                        return "(%(t)s (*)[1])&%(name)s%(idx)s" % \
                            {'t' : arg.ctype,
                             'name' : name,
                             'idx' : idx}
                    else:
                        raise RuntimeError("Don't know how to pass kernel arg %s" % arg)
                else:
                    return c_ind_data(arg, "i_%d" % arg.idx.index)
            elif arg._is_indirect:
                if arg._is_vec_map:
                    return c_vec_name(arg)
                return c_ind_data(arg, arg.idx)
            elif isinstance(arg.data, Global):
                return c_arg_name(arg)
            else:
                return "%(name)s + i * %(dim)s" % \
                    {'name' : c_arg_name(arg),
                     'dim' : arg.data.cdim}

        def c_vec_init(arg):
            val = []
            for i in range(arg.map._dim):
                val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                           {'vec_name' : c_vec_name(arg),
                            'idx' : i,
                            'data' : c_ind_data(arg, i)} )
            return ";\n".join(val)

        def c_addto_scalar_field(arg):
            name = c_arg_name(arg)
            p_data = 'p_%s' % name
            maps = as_tuple(arg.map, Map)
            nrows = maps[0].dim
            ncols = maps[1].dim

            return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s)' % \
                {'mat' : name,
                 'vals' : p_data,
                 'nrows' : nrows,
                 'ncols' : ncols,
                 'rows' : "%s + i * %s" % (c_map_name(arg), nrows),
                 'cols' : "%s2 + i * %s" % (c_map_name(arg), ncols)}

        def c_addto_vector_field(arg):
            name = c_arg_name(arg)
            p_data = 'p_%s' % name
            maps = as_tuple(arg.map, Map)
            nrows = maps[0].dim
            ncols = maps[1].dim
            dims = arg.data.sparsity.dims
            rmult = dims[0]
            cmult = dims[1]
            s = []
            for i in xrange(rmult):
                for j in xrange(cmult):
                    idx = '[%d][%d]' % (i, j)
                    val = "&%s%s" % (p_data, idx)
                    row = "%(m)s * %(map)s[i * %(dim)s + i_0] + %(i)s" % \
                          {'m' : rmult,
                           'map' : c_map_name(arg),
                           'dim' : nrows,
                           'i' : i }
                    col = "%(m)s * %(map)s2[i * %(dim)s + i_1] + %(j)s" % \
                          {'m' : cmult,
                           'map' : c_map_name(arg),
                           'dim' : ncols,
                           'j' : j }

                    s.append('addto_scalar(%s, %s, %s, %s)' % (name, val, row, col))
            return ';\n'.join(s)

        def c_assemble(arg):
            name = c_arg_name(arg)
            return "assemble_mat(%s)" % name

        def itspace_loop(i, d):
            return "for (int i_%d=0; i_%d<%d; ++i_%d){" % (i, i, d, i)

        def tmp_decl(arg, extents):
            t = arg.data.ctype
            if arg.data._is_scalar_field:
                dims = ''.join(["[%d]" % d for d in extents])
            elif arg.data._is_vector_field:
                dims = ''.join(["[%d]" % d for d in arg.data.dims])
            else:
                raise RuntimeError("Don't know how to declare temp array for %s" % arg)
            return "%s p_%s%s" % (t, c_arg_name(arg), dims)

        def c_zero_tmp(arg):
            name = "p_" + c_arg_name(arg)
            t = arg.ctype
            if arg.data._is_scalar_field:
                idx = ''.join(["[i_%d]" % i for i,_ in enumerate(arg.data.dims)])
                return "%(name)s%(idx)s = (%(t)s)0" % \
                    {'name' : name, 't' : t, 'idx' : idx}
            elif arg.data._is_vector_field:
                size = np.prod(arg.data.dims)
                return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                    {'name' : name, 't' : t, 'size' : size}
            else:
                raise RuntimeError("Don't know how to zero temp array for %s" % arg)

        def c_const_arg(c):
            return 'PyObject *_%s' % c.name

        def c_const_init(c):
            d = {'name' : c.name,
                 'type' : c.ctype}
            if c.cdim == 1:
                return '%(name)s = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[0]' % d
            tmp = '%(name)s[%%(i)s] = ((%(type)s *)(((PyArrayObject *)_%(name)s)->data))[%%(i)s]' % d
            return ';\n'.join([tmp % {'i' : i} for i in range(c.cdim)])

        args = self.args
        _wrapper_args = ', '.join([c_wrapper_arg(arg) for arg in args])

        _tmp_decs = ';\n'.join([tmp_decl(arg, self._it_space.extents) for arg in args if arg._is_mat])
        _wrapper_decs = ';\n'.join([c_wrapper_dec(arg) for arg in args])

        _const_decs = '\n'.join([const._format_declaration() for const in Const._definitions()]) + '\n'

        _kernel_user_args = [c_kernel_arg(arg) for arg in args]
        _kernel_it_args   = ["i_%d" % d for d in range(len(self._it_space.extents))]
        _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)
        _vec_inits = ';\n'.join([c_vec_init(arg) for arg in args \
                                 if not arg._is_mat and arg._is_vec_map])

        _itspace_loops = '\n'.join([itspace_loop(i,e) for i, e in zip(range(len(self._it_space.extents)), self._it_space.extents)])
        _itspace_loop_close = '}'*len(self._it_space.extents)

        _addtos_vector_field = ';\n'.join([c_addto_vector_field(arg) for arg in args \
                                           if arg._is_mat and arg.data._is_vector_field])
        _addtos_scalar_field = ';\n'.join([c_addto_scalar_field(arg) for arg in args \
                                           if arg._is_mat and arg.data._is_scalar_field])

        _assembles = ';\n'.join([c_assemble(arg) for arg in args if arg._is_mat])

        _zero_tmps = ';\n'.join([c_zero_tmp(arg) for arg in args if arg._is_mat])

        _set_size_wrapper = 'PyObject *_%(set)s_size' % {'set' : self._it_space.name}
        _set_size_dec = 'int %(set)s_size = (int)PyInt_AsLong(_%(set)s_size);' % {'set' : self._it_space.name}
        _set_size = '%(set)s_size' % {'set' : self._it_space.name}

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])
        wrapper = """
            void wrap_%(kernel_name)s__(%(set_size_wrapper)s, %(wrapper_args)s %(const_args)s) {
            %(set_size_dec)s;
            %(wrapper_decs)s;
            %(tmp_decs)s;
            %(const_inits)s;
            for ( int i = 0; i < %(set_size)s; i++ ) {
            %(vec_inits)s;
            %(itspace_loops)s
            %(zero_tmps)s;
            %(kernel_name)s(%(kernel_args)s);
            %(addtos_vector_field)s;
            %(itspace_loop_close)s
            %(addtos_scalar_field)s;
            }
        %(assembles)s;
            }"""

        if any(arg._is_soa for arg in args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            inline %(code)s
            #undef OP2_STRIDE
            """ % {'code' : self._kernel.code}
        else:
            kernel_code = """
            inline %(code)s
            """ % {'code' : self._kernel.code }
        code_to_compile =  wrapper % { 'kernel_name' : self._kernel.name,
                                       'wrapper_args' : _wrapper_args,
                                       'wrapper_decs' : _wrapper_decs,
                                       'const_args' : _const_args,
                                       'const_inits' : _const_inits,
                                       'tmp_decs' : _tmp_decs,
                                       'set_size' : _set_size,
                                       'set_size_dec' : _set_size_dec,
                                       'set_size_wrapper' : _set_size_wrapper,
                                       'itspace_loops' : _itspace_loops,
                                       'itspace_loop_close' : _itspace_loop_close,
                                       'vec_inits' : _vec_inits,
                                       'zero_tmps' : _zero_tmps,
                                       'kernel_args' : _kernel_args,
                                       'addtos_vector_field' : _addtos_vector_field,
                                       'addtos_scalar_field' : _addtos_scalar_field,
                                       'assembles' : _assembles}

        _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                                 additional_definitions = _const_decs + kernel_code,
                                 include_dirs=[OP2_INC, petsc.get_petsc_dir()+'/include'],
                                 source_directory=os.path.dirname(os.path.abspath(__file__)),
                                 wrap_headers=["mat_utils.h"],
                                 library_dirs=[OP2_LIB, petsc.get_petsc_dir()+'/lib'],
                                 libraries=['op2_seq', 'petsc'],
                                 sources=["mat_utils.cxx"])

        rt._parloop_cache[key] = _fun
        return _fun


@validate_type(('mat', Mat, MatTypeError),
               ('x', Dat, DatTypeError),
               ('b', Dat, DatTypeError))
def solve(M, b, x):
    core.solve(M, b, x)

def _setup():
    pass

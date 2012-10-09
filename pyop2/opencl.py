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

"""OP2 OpenCL backend."""

import runtime_base as op2
from utils import verify_reshape, uniquify, maybe_setflags
from runtime_base import IdentityMap, READ, WRITE, RW, INC, MIN, MAX, Set
from runtime_base import Sparsity, IterationSpace
import configuration as cfg
import op_lib_core as core
import pyopencl as cl
from pyopencl import array
import pkg_resources
import pycparser
import numpy as np
from numbers import Number
import collections
import warnings
import math
from jinja2 import Environment, PackageLoader
from pycparser import c_parser, c_ast, c_generator
import os
import re
import time
import md5

class Kernel(op2.Kernel):
    """OP2 OpenCL kernel type."""

    def __init__(self, code, name):
        op2.Kernel.__init__(self, code, name)

    class Instrument(c_ast.NodeVisitor):
        """C AST visitor for instrumenting user kernels.
             - adds memory space attribute to user kernel declaration
             - appends constant declaration to user kernel param list
             - adds a separate function declaration for user kernel
        """
        def instrument(self, ast, kernel_name, instrument, constants):
            self._kernel_name = kernel_name
            self._instrument = instrument
            self._ast = ast
            self._constants = constants
            self.generic_visit(ast)
            idx = ast.ext.index(self._func_node)
            ast.ext.insert(0, self._func_node.decl)

        def visit_FuncDef(self, node):
            if node.decl.name == self._kernel_name:
                self._func_node = node
                self.visit(node.decl)

        def visit_ParamList(self, node):
            for i, p in enumerate(node.params):
                if self._instrument[i][0]:
                    p.storage.append(self._instrument[i][0])
                if self._instrument[i][1]:
                    p.type.quals.append(self._instrument[i][1])

            for cst in self._constants:
                if cst._is_scalar:
                    t = c_ast.TypeDecl(cst._name, [], c_ast.IdentifierType([cst._cl_type]))
                else:
                    t = c_ast.PtrDecl([], c_ast.TypeDecl(cst._name, ["__constant"], c_ast.IdentifierType([cst._cl_type])))
                decl = c_ast.Decl(cst._name, [], [], [], t, None, 0)
                node.params.append(decl)

    def instrument(self, instrument, constants):
        def comment_remover(text):
            """Remove all C- and C++-style comments from a string."""
            # Reference: http://stackoverflow.com/questions/241327/python-snippet-to-remove-c-and-c-comments
            def replacer(match):
                s = match.group(0)
                if s.startswith('/'):
                    return ""
                else:
                    return s
            pattern = re.compile(r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                                 re.DOTALL | re.MULTILINE)
            return re.sub(pattern, replacer, text)

        ast = c_parser.CParser().parse(comment_remover(self._code).replace("\\\n", "\n"))
        Kernel.Instrument().instrument(ast, self._name, instrument, constants)
        return c_generator.CGenerator().visit(ast)

class Arg(op2.Arg):
    """OP2 OpenCL argument type."""

    # Codegen specific
    @property
    def _d_is_staged(self):
        return self._is_direct and not self.data._is_scalar

    @property
    def _i_gen_vec(self):
        assert self._is_vec_map or self._uses_itspace
        return map(lambda i: Arg(self.data, self.map, i, self.access), range(self.map.dim))

class DeviceDataMixin(object):
    """Codegen mixin for datatype and literal translation."""

    ClTypeInfo = collections.namedtuple('ClTypeInfo', ['clstring', 'zero', 'min', 'max'])
    CL_TYPES = {np.dtype('uint8'): ClTypeInfo('uchar', '0', '0', '255'),
                np.dtype('int8'): ClTypeInfo('char', '0', '-127', '127'),
                np.dtype('uint16'): ClTypeInfo('ushort', '0', '0', '65535'),
                np.dtype('int16'): ClTypeInfo('short', '0', '-32767', '32767'),
                np.dtype('uint32'): ClTypeInfo('uint', '0u', '0u', '4294967295u'),
                np.dtype('int32'): ClTypeInfo('int', '0', '-2147483647', '2147483647'),
                np.dtype('uint64'): ClTypeInfo('ulong', '0ul', '0ul', '18446744073709551615ul'),
                np.dtype('int64'): ClTypeInfo('long', '0l', '-9223372036854775807l', '9223372036854775807l'),
                np.dtype('float32'): ClTypeInfo('float', '0.0f', '-3.4028235e+38f', '3.4028235e+38f'),
                np.dtype('float64'): ClTypeInfo('double', '0.0', '-1.7976931348623157e+308', '1.7976931348623157e+308')}

    @property
    def bytes_per_elem(self):
        return self.dtype.itemsize * self.cdim

    @property
    def _is_scalar(self):
        return self.cdim == 1

    @property
    def _cl_type(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].clstring

    @property
    def _cl_type_zero(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].zero

    @property
    def _cl_type_min(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].min

    @property
    def _cl_type_max(self):
        return DeviceDataMixin.CL_TYPES[self.dtype].max

    @property
    def _dirty(self):
        if not hasattr(self, '_ddm_dirty'):
            self._ddm_dirty = False
        return self._ddm_dirty

    @_dirty.setter
    def _dirty(self, value):
        self._ddm_dirty = value


def one_time(func):
    # decorator, memoize and return method first call result
    def wrap(self):
        try:
            value = self._memoize[func.__name__]
        except (KeyError, AttributeError):
            value = func(self)
            try:
                cache = self._memoize
            except AttributeError:
                cache = self._memoize = dict()
            cache[func.__name__] = value
        return value

    wrap.__name__ = func.__name__
    wrap.__doc__ = func.__doc__
    return wrap

class Dat(op2.Dat, DeviceDataMixin):
    """OP2 OpenCL vector data type."""

    _arg_type = Arg
    _array = None

    @property
    def array(self):
        """Return the OpenCL device array or None if not yet initialised."""
        if self._array is None and len(self._data) is not 0:
            self._array =  array.to_device(_queue, self._data)
        return self._array

    @array.setter
    def array(self, ary):
        assert self._array is None or self._array.shape == ary.shape
        self._array = ary
        self._dirty = True

    @property
    def data(self):
        if len(self._data) is 0:
            raise RuntimeError("Temporary dat has no data on the host")

        maybe_setflags(self._data, write=True)
        if self._dirty:
            self.array.get(queue=_queue, ary=self._data)
            if self.soa:
                np.transpose(self._data)
            self._dirty = False
        return self._data

    @property
    def data_ro(self):
        if len(self._data) is 0:
            raise RuntimeError("Temporary dat has no data on the host")
        maybe_setflags(self._data, write=True)
        if self._dirty:
            self.array.get(queue=_queue, ary=self._data)
            if self.soa:
                np.transpose(self._data)
            self._dirty = False
        maybe_setflags(self._data, write=False)
        return self._data

    def _upload_from_c_layer(self):
        self.array.set(self._data, queue=_queue)

    def __iadd__(self, other):
        """Pointwise addition of fields."""
        self.array += other.array
        return self

    def __isub__(self, other):
        """Pointwise multiplication of fields."""
        self.array -= other.array
        return self

    def __imul__(self, other):
        """Pointwise multiplication or scaling of fields."""
        if isinstance(other, (Number, np.generic)):
            self.array *= other
        else:
            self.array *= other.array
        return self

    def __idiv__(self, other):
        """Pointwise division or scaling of fields."""
        if isinstance(other, (Number, np.generic)):
            self.array /= other
        else:
            self.array /= other.array
        return self

    @property
    def norm(self):
        """The L2-norm on the flattened vector."""
        return np.sqrt(array.dot(self.array, self.array).get())

def solve(M, b, x):
    x.data
    b.data
    core.solve(M, b, x)
    x._upload_from_c_layer()
    b._upload_from_c_layer()

class Mat(op2.Mat, DeviceDataMixin):
    """OP2 OpenCL matrix data type."""

    _arg_type = Arg

    @property
    @one_time
    def _dev_array(self):
        return array.empty(_queue, self._sparsity._c_handle.total_nz, self.dtype)

    @property
    @one_time
    def _dev_colidx(self):
        return array.to_device(_queue, self._sparsity._c_handle.colidx)

    @property
    @one_time
    def _dev_rowptr(self):
        return array.to_device(_queue, self._sparsity._c_handle.rowptr)

    def _upload_array(self):
        self._dev_array.set(self._c_handle.array, queue=_queue)
        self._dirty = False

    def assemble(self):
        if self._dirty:
            self._dev_array.get(queue=_queue, ary=self._c_handle.array)
            self._c_handle.restore_array()
            self._dirty = False
        self._c_handle.assemble()

    @property
    def cdim(self):
        return np.prod(self.dims)


class Const(op2.Const, DeviceDataMixin):
    """OP2 OpenCL data that is constant for any element of any set."""

    @property
    @one_time
    def _array(self):
        return array.to_device(_queue, self._data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self._array.set(self._data, queue=_queue)


class Global(op2.Global, DeviceDataMixin):
    """OP2 OpenCL global value."""

    _arg_type = Arg

    @property
    @one_time
    def _array(self):
        return array.to_device(_queue, self._data)

    def _allocate_reduction_array(self, nelems):
        self._h_reduc_array = np.zeros (nelems * self.cdim, dtype=self.dtype)
        self._d_reduc_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_WRITE, size=self._h_reduc_array.nbytes)
        cl.enqueue_copy(_queue, self._d_reduc_buffer, self._h_reduc_array, is_blocking=True).wait()

    @property
    def data(self):
        if self._dirty:
            self._array.get(queue=_queue, ary=self._data)
            self._dirty = False
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)
        self._array.set(self._data, queue=_queue)
        self._dirty = False

    def _post_kernel_reduction_task(self, nelems, reduction_operator):
        assert reduction_operator in [INC, MIN, MAX]

        def generate_code():
            def headers():
                if self.dtype == np.dtype('float64'):
                    return """
#if defined(cl_khr_fp64)
#if defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

"""
                else:
                    return ""

            def op():
                if reduction_operator is INC:
                    return "INC"
                elif reduction_operator is MIN:
                    return "min"
                elif reduction_operator is MAX:
                        return "max"
                assert False

            return """
%(headers)s
#define INC(a,b) ((a)+(b))
__kernel
void global_%(type)s_%(dim)s_post_reduction (
  __global %(type)s* dat,
  __global %(type)s* tmp,
  __private int count
)
{
  __private %(type)s accumulator[%(dim)d];
  for (int j = 0; j < %(dim)d; ++j)
  {
    accumulator[j] = dat[j];
  }
  for (int i = 0; i < count; ++i)
  {
    for (int j = 0; j < %(dim)d; ++j)
    {
      accumulator[j] = %(op)s(accumulator[j], *(tmp + i * %(dim)d + j));
    }
  }
  for (int j = 0; j < %(dim)d; ++j)
  {
    dat[j] = accumulator[j];
  }
}
""" % {'headers': headers(), 'dim': self.cdim, 'type': self._cl_type, 'op': op()}


        if not _reduction_task_cache.has_key((self.dtype, self.cdim, reduction_operator)):
            _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)] = generate_code()

        src = _reduction_task_cache[(self.dtype, self.cdim, reduction_operator)]
        name = "global_%s_%s_post_reduction" % (self._cl_type, self.cdim)
        prg = cl.Program(_ctx, src).build(options="-Werror")
        kernel = prg.__getattr__(name)
        kernel.append_arg(self._array.data)
        kernel.append_arg(self._d_reduc_buffer)
        kernel.append_arg(np.int32(nelems))
        cl.enqueue_task(_queue, kernel).wait()

        del self._d_reduc_buffer

class Map(op2.Map):
    """OP2 OpenCL map, a relation between two Sets."""

    _arg_type = Arg

    @property
    @one_time
    def _array(self):
        assert self._iterset.size != 0, 'cannot upload IdentityMap'
        return array.to_device(_queue, self._values)

    @property
    @one_time
    def md5(self):
        return md5.new(self._values).digest()

class OpPlanCache():
    """Cache for OpPlan."""

    def __init__(self):
        self._cache = dict()

    def get_plan(self, parloop, **kargs):
        try:
            plan = self._cache[parloop._plan_key]
        except KeyError:
            cp = core.op_plan(parloop._kernel, parloop._it_space.iterset, *parloop._args, **kargs)
            plan = OpPlan(parloop, cp)
            self._cache[parloop._plan_key] = plan

        return plan

    @property
    def nentries(self):
        return len(self._cache)

class OpPlan():
    """ Helper proxy for core.op_plan."""

    def __init__(self, parloop, core_plan):
        self._parloop = parloop
        self._core_plan = core_plan

        self.load()

    def load(self):
        self.nuinds = sum(map(lambda a: a._is_indirect, self._parloop._args))
        _ind_desc = [-1] * len(self._parloop._args)
        _d = {}
        _c = 0
        for i, arg in enumerate(self._parloop._args):
            if arg._is_indirect:
                if _d.has_key((arg.data, arg.map)):
                    _ind_desc[i] = _d[(arg.data, arg.map)]
                else:
                    _ind_desc[i] = _c
                    _d[(arg.data, arg.map)] = _c
                    _c += 1
        del _c
        del _d

        _off = [0] * (self._core_plan.ninds + 1)
        for i in range(self._core_plan.ninds):
            _c = 0
            for idesc in _ind_desc:
                if idesc == i:
                    _c += 1
            _off[i+1] = _off[i] + _c

        self._ind_map_buffers = [None] * self._core_plan.ninds
        for i in range(self._core_plan.ninds):
            self._ind_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int32(0).itemsize * (_off[i+1] - _off[i]) * self._parloop._it_space.size))
            s = self._parloop._it_space.size * _off[i]
            e = s + (_off[i+1] - _off[i]) * self._parloop._it_space.size
            cl.enqueue_copy(_queue, self._ind_map_buffers[i], self._core_plan.ind_map[s:e], is_blocking=True).wait()

        self._loc_map_buffers = [None] * self.nuinds
        for i in range(self.nuinds):
            self._loc_map_buffers[i] = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=int(np.int16(0).itemsize * self._parloop._it_space.size))
            s = i * self._parloop._it_space.size
            e = s + self._parloop._it_space.size
            cl.enqueue_copy(_queue, self._loc_map_buffers[i], self._core_plan.loc_map[s:e], is_blocking=True).wait()

        self._ind_sizes_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.ind_sizes.nbytes)
        cl.enqueue_copy(_queue, self._ind_sizes_buffer, self._core_plan.ind_sizes, is_blocking=True).wait()

        self._ind_offs_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.ind_offs.nbytes)
        cl.enqueue_copy(_queue, self._ind_offs_buffer, self._core_plan.ind_offs, is_blocking=True).wait()

        self._blkmap_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.blkmap.nbytes)
        cl.enqueue_copy(_queue, self._blkmap_buffer, self._core_plan.blkmap, is_blocking=True).wait()

        self._offset_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.offset.nbytes)
        cl.enqueue_copy(_queue, self._offset_buffer, self._core_plan.offset, is_blocking=True).wait()

        self._nelems_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.nelems.nbytes)
        cl.enqueue_copy(_queue, self._nelems_buffer, self._core_plan.nelems, is_blocking=True).wait()

        self._nthrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.nthrcol.nbytes)
        cl.enqueue_copy(_queue, self._nthrcol_buffer, self._core_plan.nthrcol, is_blocking=True).wait()

        self._thrcol_buffer = cl.Buffer(_ctx, cl.mem_flags.READ_ONLY, size=self._core_plan.thrcol.nbytes)
        cl.enqueue_copy(_queue, self._thrcol_buffer, self._core_plan.thrcol, is_blocking=True).wait()

        if _debug:
            print 'plan ind_map ' + str(self._core_plan.ind_map)
            print 'plan loc_map ' + str(self._core_plan.loc_map)
            print '_ind_desc ' + str(_ind_desc)
            print 'nuinds %d' % self.nuinds
            print 'ninds %d' % self.ninds
            print '_off ' + str(_off)
            for i in range(self.ninds):
                print 'ind_map[' + str(i) + '] = ' + str(self.ind_map[s:e])
            for i in range(self.nuinds):
                print 'loc_map[' + str(i) + '] = ' + str(self.loc_map[s:e])
            print 'ind_sizes :' + str(self.ind_sizes)
            print 'ind_offs :' + str(self.ind_offs)
            print 'blk_map :' + str(self.blkmap)
            print 'offset :' + str(self.offset)
            print 'nelems :' + str(self.nelems)
            print 'nthrcol :' + str(self.nthrcol)
            print 'thrcol :' + str(self.thrcol)

    @property
    def nshared(self):
        return self._core_plan.nshared

    @property
    def ninds(self):
        return self._core_plan.ninds

    @property
    def ncolors(self):
        return self._core_plan.ncolors

    @property
    def ncolblk(self):
        return self._core_plan.ncolblk

    @property
    def nblocks(self):
        return self._core_plan.nblocks

class DatMapPair(object):
    """ Dummy class needed for codegen
        (could do without but would obfuscate codegen templates)
    """
    def __init__(self, data, map):
        self.data = data
        self.map = map

    def __hash__(self):
        return hash(self.data) ^ hash(self.map)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

class ParLoop(op2.ParLoop):
    """Invocation of an OP2 OpenCL kernel with an access descriptor"""

    def __init__(self, kernel, it_space, *args):
        op2.ParLoop.__init__(self, kernel, it_space, *args)
        self._args = list()
        for a in self.args:
            if a._is_vec_map:
                for i in range(a.map._dim):
                    self._args.append(Arg(a.data, a.map, i, a.access))
            elif a._is_mat:
                pass
            elif a._uses_itspace:
                for i in range(self._it_space.extents[a.idx.index]):
                    self._args.append(Arg(a.data, a.map, i, a.access))
            else:
                self._args.append(a)

        # sort args - keep actual args unchanged
        # order globals r, globals reduc, direct, indirect
        gbls = self._global_non_reduction_args +\
               sorted(self._global_reduction_args,
                      key=lambda arg: (arg.data.dtype.itemsize,arg.data.cdim))
        directs = self._direct_args
        indirects = sorted(self._indirect_args,
                           key=lambda arg: (arg.map.md5, id(arg.data), arg.idx))

        self._args = gbls + directs + indirects

    @property
    def _plan_key(self):
        """Canonical representation of a parloop wrt plan caching."""

        # Globals: irrelevant, they only possibly effect the partition
        # size for reductions.
        # Direct Dats: irrelevant, no staging
        # iteration size: effect ind/loc maps sizes
        # partition size: effect interpretation of ind/loc maps

        # ind: for each dat map pair, the ind and loc map depend on the dim of
        #   the map, and the actual indices referenced
        inds = list()
        for dm in self._dat_map_pairs:
            d = dm.data
            m = dm.map
            indices = tuple(a.idx for a in self._args if a.data == d and a.map == m)

            inds.append((m.md5, m._dim, indices))

        # coloring part of the key,
        # for each dat, includes (map, (idx, ...)) involved (INC)
        # dats do not matter here, but conflicts should be sorted
        cols = list()
        for i, d in enumerate(sorted((dm.data for dm in self._dat_map_pairs),
                                     key=id)):
            conflicts = list()
            has_conflict = False
            for m in uniquify(a.map for a in self._args if a.data == d and a._is_indirect):
                idx = sorted(arg.idx for arg in self._indirect_reduc_args \
                             if arg.data == d and arg.map == m)
                if len(idx) > 0:
                    has_conflict = True
                    conflicts.append((m.md5, tuple(idx)))
            if has_conflict:
                cols.append(tuple(conflicts))

        return (self._it_space.size,
                self._i_partition_size(),
                tuple(inds),
                tuple(cols))

    def __hash__(self):
        """Canonical representation of a parloop wrt generated code caching."""
        # user kernel: hash of Kernel [code + name] (same code can contain
        #   multiple user kernels)
        # hash iteration space description
        # for each actual arg:
        #   its type (dat | gbl | mat)
        #   dtype (required for casts and opencl extensions)
        #   dat.dim (dloops: if staged or reduc; indloops; if not direct dat)
        #   access  (dloops: if staged or reduc; indloops; if not direct dat)
        #   the ind map index: gbl = -1, direct = -1, indirect = X (first occurence
        #     of the dat/map pair) (will tell which arg use which ind/loc maps)
        #     vecmap = -X (size of the map)
        # for vec map arg we need the dimension of the map
        # hash of consts in alphabetial order: name, dtype (used in user kernel)

        def argdimacc(arg):
            if self.is_direct():
                if arg._is_global or (arg._is_dat and not arg.data._is_scalar):
                    return (arg.data.cdim, arg.access)
                else:
                    return ()
            else:
                if (arg._is_global and arg.access is READ) or arg._is_direct:
                    return ()
                else:
                    return (arg.data.cdim, arg.access)

        argdesc = []
        seen = dict()
        c = 0
        for arg in self.args:
            if arg._is_indirect:
                if not seen.has_key((arg.data,arg.map)):
                    seen[(arg.data,arg.map)] = c
                    idesc = (c, (- arg.map.dim) if arg._is_vec_map else arg.idx)
                    c += 1
                else:
                    idesc = (seen[(arg.data,arg.map)], (- arg.map.dim) if arg._is_vec_map else arg.idx)
            else:
                idesc = ()

            d = (arg.data.__class__,
                 arg.data.dtype) + argdimacc(arg) + idesc

            argdesc.append(d)

        hsh = hash(self._kernel)
        hsh ^= hash(self._it_space)
        hsh ^= hash(tuple(argdesc))
        for c in Const._definitions():
            hsh ^= hash(c)

        return hsh

    # generic
    @property
    def _global_reduction_args(self):
        return uniquify(a for a in self._args if a._is_global_reduction)

    @property
    def _global_non_reduction_args(self):
        return uniquify(a for a in self._args if a._is_global and not a._is_global_reduction)

    @property
    def _unique_dats(self):
        return uniquify(a.data for a in self._args if a._is_dat)

    @property
    def _indirect_reduc_args(self):
        return uniquify(a for a in self._args if a._is_indirect_reduction)

    @property
    def _direct_args(self):
        return uniquify(a for a in self._args if a._is_direct)

    @property
    def _direct_non_scalar_args(self):
        return [a for a in self._direct_args if not a.data._is_scalar]

    @property
    def _direct_non_scalar_read_args(self):
        return [a for a in self._direct_non_scalar_args if a.access in [READ, RW]]

    @property
    def _direct_non_scalar_written_args(self):
        return [a for a in self._direct_non_scalar_args if a.access in [WRITE, RW]]

    @property
    def _has_itspace(self):
        return len(self._it_space.extents) > 0

    @property
    def _matrix_args(self):
        return [a for a in self.args if a._is_mat]

    @property
    def _itspace_args(self):
        return [a for a in self.args if a._uses_itspace and not a._is_mat]

    @property
    def _unique_matrix(self):
        return uniquify(a.data for a in self._matrix_args)

    @property
    def _matrix_entry_maps(self):
        """Set of all mappings used in matrix arguments."""
        return uniquify(m for arg in self.args  if arg._is_mat for m in arg.map)

    @property
    def _indirect_args(self):
        return [a for a in self._args if a._is_indirect]

    @property
    def _vec_map_args(self):
        return [a for a in self.args if a._is_vec_map]

    @property
    def _dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args)

    @property
    def _nonreduc_vec_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._vec_map_args if a.access is not INC)

    @property
    def _reduc_vec_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._vec_map_args if a.access is INC)

    @property
    def _nonreduc_itspace_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._itspace_args if a.access is not INC)

    @property
    def _reduc_itspace_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._itspace_args if a.access is INC)

    @property
    def _read_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args if a.access in [READ, RW])

    @property
    def _written_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._indirect_args if a.access in [WRITE, RW])

    @property
    def _indirect_reduc_dat_map_pairs(self):
        return uniquify(DatMapPair(a.data, a.map) for a in self._args if a._is_indirect_reduction)

    def dump_gen_code(self, src):
        if cfg['dump-gencode']:
            path = cfg['dump-gencode-path'] % {"kernel": self._kernel._name,
                                               "time": time.strftime('%Y-%m-%d@%H:%M:%S')}

            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(src)

    def _d_max_local_memory_required_per_elem(self):
        """Computes the maximum shared memory requirement per iteration set elements."""
        def max_0(iterable):
            return max(iterable) if iterable else 0
        staging = max_0([a.data.bytes_per_elem for a in self._direct_non_scalar_args])
        reduction = max_0([a.data.dtype.itemsize for a in self._global_reduction_args])
        return max(staging, reduction)

    def _i_partition_size(self):
        #TODO FIX: something weird here
        #available_local_memory
        warnings.warn('temporary fix to available local memory computation (-512)')
        available_local_memory = _max_local_memory - 512
        # 16bytes local mem used for global / local indices and sizes
        available_local_memory -= 16
        # (4/8)ptr size per dat passed as argument (dat)
        available_local_memory -= (_address_bits / 8) * (len(self._unique_dats) + len(self._global_non_reduction_args))
        # (4/8)ptr size per dat/map pair passed as argument (ind_map)
        available_local_memory -= (_address_bits / 8) * len(self._dat_map_pairs)
        # (4/8)ptr size per global reduction temp array
        available_local_memory -= (_address_bits / 8) * len(self._global_reduction_args)
        # (4/8)ptr size per indirect arg (loc_map)
        available_local_memory -= (_address_bits / 8) * len(filter(lambda a: not a._is_indirect, self._args))
        # (4/8)ptr size * 7: for plan objects
        available_local_memory -= (_address_bits / 8) * 7
        # 1 uint value for block offset
        available_local_memory -= 4
        # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
        available_local_memory -= 7
        # 12: shared_memory_offset, active_thread_count, active_thread_count_ceiling variables (could be 8 or 12 depending)
        #     and 3 for potential padding after shared mem buffer
        available_local_memory -= 12 + 3
        # 2 * (4/8)ptr size + 1uint32: DAT_via_MAP_indirection(./_size/_map) per dat map pairs
        available_local_memory -= 4 + (_address_bits / 8) * 2 * len(self._dat_map_pairs)
        # inside shared memory padding
        available_local_memory -= 2 * (len(self._dat_map_pairs) - 1)

        max_bytes = sum(map(lambda a: a.data.bytes_per_elem, self._indirect_args))
        return available_local_memory / (2 * _warpsize * max_bytes) * (2 * _warpsize)

    def launch_configuration(self):
        if self.is_direct():
            per_elem_max_local_mem_req = self._d_max_local_memory_required_per_elem()
            shared_memory_offset = per_elem_max_local_mem_req * _warpsize
            if per_elem_max_local_mem_req == 0:
                wgs = _max_work_group_size
            else:
                # 16bytes local mem used for global / local indices and sizes
                # (4/8)ptr bytes for each dat buffer passed to the kernel
                # (4/8)ptr bytes for each temporary global reduction buffer passed to the kernel
                # 7: 7bytes potentialy lost for aligning the shared memory buffer to 'long'
                warnings.warn('temporary fix to available local memory computation (-512)')
                available_local_memory = _max_local_memory - 512
                available_local_memory -= 16
                available_local_memory -= (len(self._unique_dats) + len(self._global_non_reduction_args))\
                                          * (_address_bits / 8)
                available_local_memory -= len(self._global_reduction_args) * (_address_bits / 8)
                available_local_memory -= 7
                ps = available_local_memory / per_elem_max_local_mem_req
                wgs = min(_max_work_group_size, (ps / _warpsize) * _warpsize)
            nwg = min(_pref_work_group_count, int(math.ceil(self._it_space.size / float(wgs))))
            ttc = wgs * nwg

            local_memory_req = per_elem_max_local_mem_req * wgs
            return {'thread_count': ttc,
                    'work_group_size': wgs,
                    'work_group_count': nwg,
                    'local_memory_size': local_memory_req,
                    'local_memory_offset': shared_memory_offset}
        else:
            return {'partition_size': self._i_partition_size()}

    def codegen(self, conf):
        def instrument_user_kernel():
            inst = []

            for arg in self.args:
                i = None
                if self.is_direct():
                    if (arg._is_direct and arg.data._is_scalar) or\
                       (arg._is_global and not arg._is_global_reduction):
                        i = ("__global", None)
                    else:
                        i = ("__private", None)
                else: # indirect loop
                    if arg._is_direct or (arg._is_global and not arg._is_global_reduction):
                        i = ("__global", None)
                    elif (arg._is_indirect or arg._is_vec_map) and not arg._is_indirect_reduction:
                        i = ("__local", None)
                    else:
                        i = ("__private", None)

                inst.append(i)

            for i in self._it_space.extents:
                inst.append(("__private", None))

            return self._kernel.instrument(inst, Const._definitions())

        # check cache
        src = op2._parloop_cache.get(hash(self))
        if src:
            return src

        #do codegen
        user_kernel = instrument_user_kernel()
        template = _jinja2_direct_loop if self.is_direct()\
                                       else _jinja2_indirect_loop

        src = template.render({'parloop': self,
                               'user_kernel': user_kernel,
                               'launch': conf,
                               'codegen': {'amd': _AMD_fixes},
                               'op2const': Const._definitions()
                              }).encode("ascii")
        self.dump_gen_code(src)
        op2._parloop_cache[hash(self)] = src
        return src

    def compute(self):
        def compile_kernel(src, name):
            prg = cl.Program(_ctx, source).build(options="-Werror")
            return prg.__getattr__(name + '_stub')

        conf = self.launch_configuration()

        if not self.is_direct():
            plan = _plan_cache.get_plan(self, partition_size=conf['partition_size'])
            conf['local_memory_size'] = plan.nshared
            conf['ninds'] = plan.ninds
            conf['work_group_size'] = min(_max_work_group_size, conf['partition_size'])
            conf['work_group_count'] = plan.nblocks
        conf['warpsize'] = _warpsize

        source = self.codegen(conf)
        kernel = compile_kernel(source, self._kernel._name)

        for a in self._unique_dats:
            kernel.append_arg(a.array.data)

        for a in self._global_non_reduction_args:
            kernel.append_arg(a.data._array.data)

        for a in self._global_reduction_args:
            a.data._allocate_reduction_array(conf['work_group_count'])
            kernel.append_arg(a.data._d_reduc_buffer)

        for cst in Const._definitions():
            kernel.append_arg(cst._array.data)

        for m in self._unique_matrix:
            kernel.append_arg(m._dev_array.data)
            m._upload_array()
            kernel.append_arg(m._dev_rowptr.data)
            kernel.append_arg(m._dev_colidx.data)

        for m in self._matrix_entry_maps:
            kernel.append_arg(m._array.data)

        if self.is_direct():
            kernel.append_arg(np.int32(self._it_space.size))

            cl.enqueue_nd_range_kernel(_queue, kernel, (conf['thread_count'],), (conf['work_group_size'],), g_times_l=False).wait()
        else:
            for i in range(plan.ninds):
                kernel.append_arg(plan._ind_map_buffers[i])

            for i in range(plan.nuinds):
                kernel.append_arg(plan._loc_map_buffers[i])

            kernel.append_arg(plan._ind_sizes_buffer)
            kernel.append_arg(plan._ind_offs_buffer)
            kernel.append_arg(plan._blkmap_buffer)
            kernel.append_arg(plan._offset_buffer)
            kernel.append_arg(plan._nelems_buffer)
            kernel.append_arg(plan._nthrcol_buffer)
            kernel.append_arg(plan._thrcol_buffer)

            block_offset = 0
            for i in range(plan.ncolors):
                blocks_per_grid = int(plan.ncolblk[i])
                threads_per_block = min(_max_work_group_size, conf['partition_size'])
                thread_count = threads_per_block * blocks_per_grid

                kernel.set_last_arg(np.int32(block_offset))
                cl.enqueue_nd_range_kernel(_queue, kernel, (int(thread_count),), (int(threads_per_block),), g_times_l=False).wait()
                block_offset += blocks_per_grid

        # mark !READ data as dirty
        for arg in self.args:
            if arg.access not in [READ]:
                arg.data._dirty = True
            if arg._is_dat:
                maybe_setflags(arg.data._data, write=False)

        for mat in [arg.data for arg in self._matrix_args]:
            mat.assemble()

        for i, a in enumerate(self._global_reduction_args):
            a.data._post_kernel_reduction_task(conf['work_group_count'], a.access)

    def is_direct(self):
        return all(map(lambda a: a._is_direct or isinstance(a.data, Global) or isinstance(a.data, Mat), self._args))

#Monkey patch pyopencl.Kernel for convenience
_original_clKernel = cl.Kernel

class CLKernel (_original_clKernel):
    def __init__(self, *args, **kargs):
        super(CLKernel, self).__init__(*args, **kargs)
        self._karg = 0

    def reset_args(self):
        self._karg = 0;

    def append_arg(self, arg):
        self.set_arg(self._karg, arg)
        self._karg += 1

    def set_last_arg(self, arg):
        self.set_arg(self._karg, arg)

cl.Kernel = CLKernel

def par_loop(kernel, it_space, *args):
    ParLoop(kernel, it_space, *args).compute()

# backend interface:
def empty_plan_cache():
    global _plan_cache
    _plan_cache = OpPlanCache()

def ncached_plans():
    global _plan_cache
    return _plan_cache.nentries

def _setup():
    global _ctx
    global _queue
    global _pref_work_group_count
    global _max_local_memory
    global _address_bits
    global _max_work_group_size
    global _has_dpfloat
    global _warpsize
    global _AMD_fixes
    global _plan_cache
    global _reduction_task_cache

    _ctx = cl.create_some_context()
    _queue = cl.CommandQueue(_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
    _pref_work_group_count = _queue.device.max_compute_units
    _max_local_memory = _queue.device.local_mem_size
    _address_bits = _queue.device.address_bits
    _max_work_group_size = _queue.device.max_work_group_size
    _has_dpfloat = 'cl_khr_fp64' in _queue.device.extensions or 'cl_amd_fp64' in _queue.device.extensions
    if not _has_dpfloat:
        warnings.warn('device does not support double precision floating point computation, expect undefined behavior for double')

    if _queue.device.type == cl.device_type.CPU:
        _warpsize = 1
    elif _queue.device.type == cl.device_type.GPU:
        # assumes nvidia, will probably fail with AMD gpus
        _warpsize = 32

    _AMD_fixes = _queue.device.platform.vendor in ['Advanced Micro Devices, Inc.']
    _plan_cache = OpPlanCache()
    _reduction_task_cache = dict()

_debug = False
_ctx = None
_queue = None
_pref_work_group_count = 0
_max_local_memory = 0
_address_bits = 32
_max_work_group_size = 0
_has_dpfloat = False
_warpsize = 0
_AMD_fixes = False
_plan_cache = None
_reduction_task_cache = None

_jinja2_env = Environment(loader=PackageLoader("pyop2", "assets"))
_jinja2_direct_loop = _jinja2_env.get_template("opencl_direct_loop.jinja2")
_jinja2_indirect_loop = _jinja2_env.get_template("opencl_indirect_loop.jinja2")

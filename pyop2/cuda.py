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

import jinja2
import numpy as np
import pycuda.driver as driver
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycparser import c_parser, c_ast, c_generator

import base
from configuration import configuration
import device as op2
from device import *
import plan
from profiling import lineprof, Timer
from utils import verify_reshape


class Kernel(op2.Kernel):

    def __init__(self, code, name, opts={}, include_dirs=[]):
        if self._initialized:
            return
        op2.Kernel.__init__(self, code, name, opts, include_dirs)
        self._code = self.instrument()

    def instrument(self):
        class Instrument(c_ast.NodeVisitor):

            """C AST visitor for instrumenting user kernels.
                 - adds __device__ declaration to function definitions
            """

            def visit_FuncDef(self, node):
                node.decl.funcspec.insert(0, '__device__')

        ast = c_parser.CParser().parse(self._code)
        Instrument().generic_visit(ast)
        return c_generator.CGenerator().visit(ast)


class Arg(op2.Arg):

    def _subset_index(self, s, subset):
        return ("_ssinds[%s]" % s) if subset else ("(%s)" % s)

    def _indirect_kernel_arg_name(self, idx, subset):
        if self._is_mat:
            rmap, cmap = self.map
            ridx, cidx = self.idx
            rmult, cmult = self.data.dims
            esize = rmult * cmult
            size = esize * rmap.arity * cmap.arity
            if self._flatten and esize > 1:
                # In the case of rmap and cmap arity 3 and rmult and cmult 2 we
                # need the local block numbering to be:
                #
                #  0  4  8 |  1  5  9   The 3 x 3 blocks have the same
                # 12 16 20 | 13 17 22   numbering with an offset of:
                # 24 28 32 | 25 29 33
                # -------------------     0 1
                #  2  6 10 |  3  7 11     2 3
                # 14 18 22 | 15 19 33
                # 26 30 24 | 27 31 35

                # Numbering of the base block
                block00 = '((i%(i0)s %% %(rarity)d) * %(carity)d + (i%(i1)s %% %(carity)d)) * %(esize)d'
                # Offset along the rows (2 for the lower half)
                roffs = ' + %(rmult)d * (i%(i0)s / %(rarity)d)'
                # Offset along the columns (1 for the right half)
                coffs = ' + i%(i1)s / %(carity)d'
                pos = lambda i0, i1: (block00 + roffs + coffs) % \
                    {'i0': i0, 'i1': i1, 'rarity': rmap.arity,
                     'carity': cmap.arity, 'esize': esize, 'rmult': rmult}
            else:
                pos = lambda i0, i1: 'i%(i0)s * %(rsize)d + i%(i1)s * %(csize)d' % \
                    {'i0': i0, 'i1': i1, 'rsize': cmap.arity * esize, 'csize': esize}
            d = {'n': self.name,
                 'offset': self._lmaoffset_name,
                 'idx': self._subset_index("ele_offset + %s" % idx, subset),
                 't': self.ctype,
                 'size': size,
                 'lcdim': 1 if self._flatten else cmult,
                 'pos': pos(ridx.index, cidx.index)}
            # We walk through the lma-data in order of the
            # alphabet:
            #  A B C
            #  D E F
            #  G H I
            #       J K
            #       L M
            #  where each sub-block is walked in the same order:
            #  A1 A2
            #  A3 A4
            return """(%(t)s (*)[%(lcdim)s])(%(n)s + %(offset)s + %(idx)s * %(size)s + %(pos)s)""" % d
        if self._is_global:
            if self._is_global_reduction:
                return self._reduction_local_name
            else:
                return self.name
        if self._is_direct:
            if self.data.soa:
                return "%s + %s" % (self.name, sub("%s + offset_b_abs" % idx))
            return "%s + %s * %s" % (self.name,
                                     self.data.cdim,
                                     self._subset_index("%s + offset_b_abs" % idx, subset))
        if self._is_indirect:
            if self._is_vec_map:
                return self._vec_name
            if self._uses_itspace:
                if self.access is op2.INC:
                    return "%s[i%s]" % (self._vec_name, self.idx.index)
                return "%s + loc_map[(%s+i%s) * set_size + %s + offset_b]*%s" \
                    % (self._shared_name, self._which_indirect,
                       self.idx.index, idx, self.data.cdim)
            if self.access is op2.INC:
                return self._local_name()
            else:
                return "%s + loc_map[%s * set_size + %s + offset_b]*%s" \
                    % (self._shared_name, self._which_indirect, idx,
                       self.data.cdim)

    def _direct_kernel_arg_name(self, idx=None):
        if self._is_staged_direct:
            return self._local_name()
        elif self._is_global_reduction:
            return self._reduction_local_name
        elif self._is_global:
            return self.name
        else:
            return "%s + %s" % (self.name, idx)


class Subset(op2.Subset):

    def _allocate_device(self):
        if not hasattr(self, '_device_data'):
            self._device_data = gpuarray.to_gpu(self.indices)


class DeviceDataMixin(op2.DeviceDataMixin):

    def _allocate_device(self):
        if self.state is DeviceDataMixin.DEVICE_UNALLOCATED:
            if self.soa:
                shape = tuple(reversed(self.shape))
            else:
                shape = self.shape
            self._device_data = gpuarray.zeros(shape=shape, dtype=self.dtype)
            self.state = DeviceDataMixin.HOST

    def _to_device(self):
        self._allocate_device()
        if self.state is DeviceDataMixin.HOST:
            self._device_data.set(self._maybe_to_soa(self._data))
            self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        if self.state is DeviceDataMixin.DEVICE:
            self._device_data.get(self._data)
            self._data = self._maybe_to_aos(self._data)
            self.state = DeviceDataMixin.BOTH


# Needs to be here to pick up correct mixin
class Dat(DeviceDataMixin, op2.Dat):

    pass


class Sparsity(op2.Sparsity):

    def __init__(self, *args, **kwargs):
        self._block_sparse = False
        super(Sparsity, self).__init__(*args, **kwargs)

    @property
    def rowptr(self):
        if not hasattr(self, '__rowptr'):
            setattr(self, '__rowptr',
                    gpuarray.to_gpu(self._rowptr))
        return getattr(self, '__rowptr')

    @property
    def colidx(self):
        if not hasattr(self, '__colidx'):
            setattr(self, '__colidx',
                    gpuarray.to_gpu(self._colidx))
        return getattr(self, '__colidx')


class Mat(DeviceDataMixin, op2.Mat):
    _lma2csr_cache = dict()

    @property
    def _lmadata(self):
        if not hasattr(self, '__lmadata'):
            nentries = 0
            # dense block of rmap.arity x cmap.arity for each rmap/cmap pair
            for rmap, cmap in self.sparsity.maps:
                nentries += rmap.arity * cmap.arity

            entry_size = 0
            # all pairs of maps in the sparsity must have the same
            # iterset, there are sum(iterset.size) * nentries total
            # entries in the LMA data
            for rmap, cmap in self.sparsity.maps:
                entry_size += rmap.iterset.size
            # each entry in the block is size dims[0] x dims[1]
            entry_size *= np.asscalar(np.prod(self.dims))
            nentries *= entry_size
            setattr(self, '__lmadata',
                    gpuarray.zeros(shape=nentries, dtype=self.dtype))
        return getattr(self, '__lmadata')

    def _lmaoffset(self, iterset):
        offset = 0
        size = self.sparsity.maps[0][0].toset.size
        size *= np.asscalar(np.prod(self.dims))
        for rmap, cmap in self.sparsity.maps:
            if rmap.iterset is iterset:
                break
            offset += rmap.arity * cmap.arity
        return offset * size

    @property
    def _rowptr(self):
        return self._sparsity.rowptr

    @property
    def _colidx(self):
        return self._sparsity.colidx

    @property
    def _csrdata(self):
        if not hasattr(self, '__csrdata'):
            setattr(self, '__csrdata',
                    gpuarray.zeros(shape=self._sparsity.nz,
                                   dtype=self.dtype))
        return getattr(self, '__csrdata')

    def __call__(self, *args, **kwargs):
        self._assembled = False
        return super(Mat, self).__call__(*args, **kwargs)

    def __getitem__(self, idx):
        """Block matrices are not yet supported in CUDA, always yield self."""
        return self

    @timed_function("CUDA assembly")
    def _assemble(self):
        if self._assembled:
            return
        self._assembled = True
        mod, sfun, vfun = Mat._lma2csr_cache.get(self.dtype,
                                                 (None, None, None))
        if mod is None:
            d = {'type': self.ctype}
            src = _matrix_support_template.render(d).encode('ascii')
            compiler_opts = ['-m64', '-Xptxas', '-dlcm=ca',
                             '-Xptxas=-v', '-O3', '-use_fast_math', '-DNVCC']
            mod = SourceModule(src, options=compiler_opts)
            sfun = mod.get_function('__lma_to_csr')
            vfun = mod.get_function('__lma_to_csr_vector')
            sfun.prepare('PPPPPiPii')
            vfun.prepare('PPPPPiiPiii')
            Mat._lma2csr_cache[self.dtype] = mod, sfun, vfun

        for rowmap, colmap in self.sparsity.maps:
            assert rowmap.iterset is colmap.iterset
            nelems = rowmap.iterset.size
            nthread = 128
            nblock = (nelems * rowmap.arity * colmap.arity) / nthread + 1

            rowmap._to_device()
            colmap._to_device()
            offset = self._lmaoffset(rowmap.iterset) * self.dtype.itemsize
            arglist = [np.intp(self._lmadata.gpudata) + offset,
                       self._csrdata.gpudata,
                       self._rowptr.gpudata,
                       self._colidx.gpudata,
                       rowmap._device_values.gpudata,
                       np.int32(rowmap.arity)]
            if self._is_scalar_field:
                arglist.extend([colmap._device_values.gpudata,
                                np.int32(colmap.arity),
                                np.int32(nelems)])
                fun = sfun
            else:
                arglist.extend([np.int32(self.dims[0]),
                                colmap._device_values.gpudata,
                                np.int32(colmap.arity),
                                np.int32(self.dims[1]),
                                np.int32(nelems)])
                fun = vfun
            _stream.synchronize()
            fun.prepared_async_call((int(nblock), 1, 1), (nthread, 1, 1), _stream, *arglist)

    @property
    def values(self):
        base._trace.evaluate(set([self]), set([self]))
        shape = self.sparsity.maps[0][0].toset.size * self.dims[0][0][0]
        shape = (shape, shape)
        ret = np.zeros(shape=shape, dtype=self.dtype)
        csrdata = self._csrdata.get()
        rowptr = self.sparsity._rowptr
        colidx = self.sparsity._colidx
        for r, (rs, re) in enumerate(zip(rowptr[:-1], rowptr[1:])):
            cols = colidx[rs:re]
            ret[r, cols] = csrdata[rs:re]
        return ret

    @property
    def array(self):
        base._trace.evaluate(set([self]), set([self]))
        return self._csrdata.get()

    @modifies
    def zero_rows(self, rows, diag_val=1.0):
        """Zeroes the specified rows of the matrix, with the exception of the
        diagonal entry, which is set to diag_val. May be used for applying
        strong boundary conditions.

        :param rows: a :class:`Subset` or an iterable"""
        base._trace.evaluate(set([self]), set([self]))
        rows = rows.indices if isinstance(rows, Subset) else rows
        for row in rows:
            s = self.sparsity._rowptr[row]
            e = self.sparsity._rowptr[row + 1]
            diag = np.where(self.sparsity._colidx[s:e] == row)[0]
            self._csrdata[s:e].fill(0)
            if len(diag) == 1:
                diag += s       # offset from row start
                self._csrdata[diag:diag + 1].fill(diag_val)

    def zero(self):
        base._trace.evaluate(set([]), set([self]))
        self._csrdata.fill(0)
        self._lmadata.fill(0)
        self._version_set_zero()

    def duplicate(self):
        other = Mat(self.sparsity)
        base._trace.evaluate(set([self]), set([self]))
        setattr(other, '__csrdata', self._csrdata.copy())
        return other


class Const(DeviceDataMixin, op2.Const):

    def _format_declaration(self):
        d = {'dim': self.cdim,
             'type': self.ctype,
             'name': self.name}

        if self.cdim == 1:
            return "__constant__ %(type)s %(name)s;" % d
        return "__constant__ %(type)s %(name)s[%(dim)s];" % d

    def _to_device(self, module):
        ptr, size = module.get_global(self.name)
        if size != self.data.nbytes:
            raise RuntimeError("Const %s needs %d bytes, but only space for %d" %
                               (self, self.data.nbytes, size))
        if self.state is DeviceDataMixin.HOST:
            driver.memcpy_htod(ptr, self._data)
            self.state = DeviceDataMixin.BOTH

    def _from_device(self):
        raise RuntimeError("Copying Const %s from device makes no sense" % self)


class Global(DeviceDataMixin, op2.Global):

    def _allocate_reduction_buffer(self, grid_size, op):
        if not hasattr(self, '_reduction_buffer') or \
           self._reduction_buffer.size != grid_size:
            self._host_reduction_buffer = np.zeros(np.prod(grid_size) * self.cdim,
                                                   dtype=self.dtype).reshape((-1,) + self._dim)
            if op is not op2.INC:
                self._host_reduction_buffer[:] = self._data
            self._reduction_buffer = gpuarray.to_gpu(self._host_reduction_buffer)
        else:
            if op is not op2.INC:
                self._reduction_buffer.fill(self._data)
            else:
                self._reduction_buffer.fill(0)

    @property
    def data(self):
        base._trace.evaluate(set([self]), set())
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST
        return self._data

    @data.setter
    def data(self, value):
        base._trace.evaluate(set(), set([self]))
        self._data = verify_reshape(value, self.dtype, self.dim)
        if self.state is not DeviceDataMixin.DEVICE_UNALLOCATED:
            self.state = DeviceDataMixin.HOST

    def _finalise_reduction_begin(self, grid_size, op):
        # Need to make sure the kernel launch finished
        _stream.synchronize()
        self._reduction_buffer.get(ary=self._host_reduction_buffer)

    def _finalise_reduction_end(self, grid_size, op):
        self.state = DeviceDataMixin.HOST
        tmp = self._host_reduction_buffer
        if op is op2.MIN:
            tmp = np.min(tmp, axis=0)
            fn = min
        elif op is op2.MAX:
            tmp = np.max(tmp, axis=0)
            fn = max
        else:
            tmp = np.sum(tmp, axis=0)
        for i in range(self.cdim):
            if op is op2.INC:
                self._data[i] += tmp[i]
            else:
                self._data[i] = fn(self._data[i], tmp[i])


class Map(op2.Map):

    def _to_device(self):
        if not hasattr(self, '_device_values'):
            self._device_values = gpuarray.to_gpu(self._values)
        elif self._state is not DeviceDataMixin.BOTH:
            self._device_values.set(self._values)
        self._state = DeviceDataMixin.BOTH

    def _from_device(self):
        if not hasattr(self, '_device_values') is None:
            raise RuntimeError("No values for Map %s on device" % self)
        self._state = DeviceDataMixin.HOST
        self._device_values.get(self._values)


class Plan(plan.Plan):

    @property
    def nthrcol(self):
        if not hasattr(self, '_nthrcol_gpuarray'):
            self._nthrcol_gpuarray = gpuarray.to_gpu(super(Plan, self).nthrcol)
        return self._nthrcol_gpuarray

    @property
    def thrcol(self):
        if not hasattr(self, '_thrcol_gpuarray'):
            self._thrcol_gpuarray = gpuarray.to_gpu(super(Plan, self).thrcol)
        return self._thrcol_gpuarray

    @property
    def offset(self):
        if not hasattr(self, '_offset_gpuarray'):
            self._offset_gpuarray = gpuarray.to_gpu(super(Plan, self).offset)
        return self._offset_gpuarray

    @property
    def ind_map(self):
        if not hasattr(self, '_ind_map_gpuarray'):
            self._ind_map_gpuarray = gpuarray.to_gpu(super(Plan, self).ind_map)
        return self._ind_map_gpuarray

    @property
    def ind_offs(self):
        if not hasattr(self, '_ind_offs_gpuarray'):
            self._ind_offs_gpuarray = gpuarray.to_gpu(super(Plan, self).ind_offs)
        return self._ind_offs_gpuarray

    @property
    def ind_sizes(self):
        if not hasattr(self, '_ind_sizes_gpuarray'):
            self._ind_sizes_gpuarray = gpuarray.to_gpu(super(Plan, self).ind_sizes)
        return self._ind_sizes_gpuarray

    @property
    def loc_map(self):
        if not hasattr(self, '_loc_map_gpuarray'):
            self._loc_map_gpuarray = gpuarray.to_gpu(super(Plan, self).loc_map)
        return self._loc_map_gpuarray

    @property
    def nelems(self):
        if not hasattr(self, '_nelems_gpuarray'):
            self._nelems_gpuarray = gpuarray.to_gpu(super(Plan, self).nelems)
        return self._nelems_gpuarray

    @property
    def blkmap(self):
        if not hasattr(self, '_blkmap_gpuarray'):
            self._blkmap_gpuarray = gpuarray.to_gpu(super(Plan, self).blkmap)
        return self._blkmap_gpuarray

_cusp_cache = dict()


def _cusp_solver(M, parameters):
    cache_key = lambda t, p: (t,
                              p['ksp_type'],
                              p['pc_type'],
                              p['ksp_rtol'],
                              p['ksp_atol'],
                              p['ksp_max_it'],
                              p['ksp_gmres_restart'],
                              p['ksp_monitor'])
    module = _cusp_cache.get(cache_key(M.ctype, parameters))
    if module:
        return module

    import codepy.toolchain
    from cgen import FunctionBody, FunctionDeclaration
    from cgen import Block, Statement, Include, Value
    from codepy.bpl import BoostPythonModule
    from codepy.cuda import CudaModule
    gcc_toolchain = codepy.toolchain.guess_toolchain()
    nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()
    if 'CUSP_HOME' in os.environ:
        nvcc_toolchain.add_library('cusp', [os.environ['CUSP_HOME']], [], [])
    host_mod = BoostPythonModule()
    nvcc_mod = CudaModule(host_mod)
    nvcc_includes = ['thrust/device_vector.h',
                     'thrust/fill.h',
                     'cusp/csr_matrix.h',
                     'cusp/krylov/cg.h',
                     'cusp/krylov/bicgstab.h',
                     'cusp/krylov/gmres.h',
                     'cusp/precond/diagonal.h',
                     'cusp/precond/smoothed_aggregation.h',
                     'cusp/precond/ainv.h',
                     'string']
    nvcc_mod.add_to_preamble([Include(s) for s in nvcc_includes])
    nvcc_mod.add_to_preamble([Statement('using namespace std')])

    # We're translating PETSc preconditioner types to CUSP
    diag = Statement('cusp::precond::diagonal< ValueType, cusp::device_memory >M(A)')
    ainv = Statement(
        'cusp::precond::scaled_bridson_ainv< ValueType, cusp::device_memory >M(A)')
    amg = Statement(
        'cusp::precond::smoothed_aggregation< IndexType, ValueType, cusp::device_memory >M(A)')
    none = Statement(
        'cusp::identity_operator< ValueType, cusp::device_memory >M(nrows, ncols)')
    preconditioners = {
        'diagonal': diag,
        'jacobi': diag,
        'ainv': ainv,
        'ainvcusp': ainv,
        'amg': amg,
        'hypre': amg,
        'none': none,
        None: none
    }
    try:
        precond_call = preconditioners[parameters['pc_type']]
    except KeyError:
        raise RuntimeError("Cusp does not support preconditioner type %s" %
                           parameters['pc_type'])
    solvers = {
        'cg': Statement('cusp::krylov::cg(A, x, b, monitor, M)'),
        'bicgstab': Statement('cusp::krylov::bicgstab(A, x, b, monitor, M)'),
        'gmres': Statement('cusp::krylov::gmres(A, x, b, %(ksp_gmres_restart)d, monitor, M)' % parameters)
    }
    try:
        solve_call = solvers[parameters['ksp_type']]
    except KeyError:
        raise RuntimeError("Cusp does not support solver type %s" %
                           parameters['ksp_type'])
    monitor = 'monitor(b, %(ksp_max_it)d, %(ksp_rtol)g, %(ksp_atol)g)' % parameters

    nvcc_function = FunctionBody(
        FunctionDeclaration(Value('void', '__cusp_solve'),
                            [Value('CUdeviceptr', '_rowptr'),
                             Value('CUdeviceptr', '_colidx'),
                             Value('CUdeviceptr', '_csrdata'),
                             Value('CUdeviceptr', '_b'),
                             Value('CUdeviceptr', '_x'),
                             Value('int', 'nrows'),
                             Value('int', 'ncols'),
                             Value('int', 'nnz')]),
        Block([
            Statement('typedef int IndexType'),
            Statement('typedef %s ValueType' % M.ctype),
            Statement(
                'typedef typename cusp::array1d_view< thrust::device_ptr<IndexType> > indices'),
            Statement(
                'typedef typename cusp::array1d_view< thrust::device_ptr<ValueType> > values'),
            Statement(
                'typedef cusp::csr_matrix_view< indices, indices, values, IndexType, ValueType, cusp::device_memory > matrix'),
            Statement('thrust::device_ptr< IndexType > rowptr((IndexType *)_rowptr)'),
            Statement('thrust::device_ptr< IndexType > colidx((IndexType *)_colidx)'),
            Statement('thrust::device_ptr< ValueType > csrdata((ValueType *)_csrdata)'),
            Statement('thrust::device_ptr< ValueType > d_b((ValueType *)_b)'),
            Statement('thrust::device_ptr< ValueType > d_x((ValueType *)_x)'),
            Statement('indices row_offsets(rowptr, rowptr + nrows + 1)'),
            Statement('indices column_indices(colidx, colidx + nnz)'),
            Statement('values matrix_values(csrdata, csrdata + nnz)'),
            Statement('values b(d_b, d_b + nrows)'),
            Statement('values x(d_x, d_x + ncols)'),
            Statement('thrust::fill(x.begin(), x.end(), (ValueType)0)'),
            Statement(
                'matrix A(nrows, ncols, nnz, row_offsets, column_indices, matrix_values)'),
            Statement('cusp::%s_monitor< ValueType > %s' %
                      ('verbose' if parameters['ksp_monitor'] else 'default',
                       monitor)),
            precond_call,
            solve_call
        ]))

    host_mod.add_to_preamble([Include('boost/python/extract.hpp'), Include('string')])
    host_mod.add_to_preamble([Statement('using namespace boost::python')])
    host_mod.add_to_preamble([Statement('using namespace std')])

    nvcc_mod.add_function(nvcc_function)

    host_mod.add_function(
        FunctionBody(
            FunctionDeclaration(Value('void', 'solve'),
                                [Value('object', '_rowptr'),
                                 Value('object', '_colidx'),
                                 Value('object', '_csrdata'),
                                 Value('object', '_b'),
                                 Value('object', '_x'),
                                 Value('object', '_nrows'),
                                 Value('object', '_ncols'),
                                 Value('object', '_nnz')]),
            Block([
                Statement(
                    'CUdeviceptr rowptr = extract<CUdeviceptr>(_rowptr.attr("gpudata"))'),
                Statement(
                    'CUdeviceptr colidx = extract<CUdeviceptr>(_colidx.attr("gpudata"))'),
                Statement(
                    'CUdeviceptr csrdata = extract<CUdeviceptr>(_csrdata.attr("gpudata"))'),
                Statement('CUdeviceptr b = extract<CUdeviceptr>(_b.attr("gpudata"))'),
                Statement('CUdeviceptr x = extract<CUdeviceptr>(_x.attr("gpudata"))'),
                Statement('int nrows = extract<int>(_nrows)'),
                Statement('int ncols = extract<int>(_ncols)'),
                Statement('int nnz = extract<int>(_nnz)'),
                Statement('__cusp_solve(rowptr, colidx, csrdata, b, x, nrows, ncols, nnz)')
            ])))

    nvcc_toolchain.cflags.append('-arch')
    nvcc_toolchain.cflags.append('sm_20')
    nvcc_toolchain.cflags.append('-O3')
    module = nvcc_mod.compile(gcc_toolchain, nvcc_toolchain, debug=configuration["debug"])

    _cusp_cache[cache_key(M.ctype, parameters)] = module
    return module

# FIXME: inherit from base while device gives us the PETSc solver


class Solver(base.Solver):

    def _solve(self, M, x, b):
        b._to_device()
        x._to_device()
        module = _cusp_solver(M, self.parameters)
        module.solve(M._rowptr,
                     M._colidx,
                     M._csrdata,
                     b._device_data,
                     x._device_data,
                     int(b.dataset.size * b.cdim),
                     int(x.dataset.size * x.cdim),
                     M._csrdata.size)
        x.state = DeviceDataMixin.DEVICE


class JITModule(base.JITModule):

    def __init__(self, kernel, itspace_extents, *args, **kwargs):
        """
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementors.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        if self._initialized:
            return
        self._parloop = kwargs.get('parloop')
        self._kernel = self._parloop._kernel
        self._config = kwargs.get('config')
        self._initialized = True

    def compile(self):
        if hasattr(self, '_fun'):
            # It should not be possible to pull a jit module out of
            # the cache referencing its par_loop
            if hasattr(self, '_parloop'):
                raise RuntimeError("JITModule is holding onto parloop, causing a memory leak (should never happen)")
            return self._fun
        # If we weren't in the cache we /must/ have a par_loop
        if not hasattr(self, '_parloop'):
            raise RuntimeError("JITModule has no parloop associated with it, should never happen")

        compiler_opts = ['-m64', '-Xptxas', '-dlcm=ca',
                         '-Xptxas=-v', '-O3', '-use_fast_math', '-DNVCC']
        inttype = np.dtype('int32').char
        argtypes = inttype      # set size
        argtypes += inttype  # offset
        if self._config['subset']:
            argtypes += "P"  # subset's indices

        d = {'parloop': self._parloop,
             'launch': self._config,
             'constants': Const._definitions()}

        if self._parloop._is_direct:
            src = _direct_loop_template.render(d).encode('ascii')
            for arg in self._parloop.args:
                argtypes += "P"  # pointer to each Dat's data
        else:
            src = _indirect_loop_template.render(d).encode('ascii')
            for arg in self._parloop._unique_args:
                if arg._is_mat:
                    # pointer to lma data, offset into lma data
                    # for case of multiple map pairs.
                    argtypes += "P"
                    argtypes += inttype
                else:
                    # pointer to each unique Dat's data
                    argtypes += "P"
            argtypes += "PPPP"  # ind_map, loc_map, ind_sizes, ind_offs
            argtypes += inttype  # block offset
            argtypes += "PPPPP"  # blkmap, offset, nelems, nthrcol, thrcol
            argtypes += inttype  # number of colours in the block

        self._module = SourceModule(src, options=compiler_opts)
        self._dump_generated_code(src, ext="cu")

        # Upload Const data.
        for c in Const._definitions():
            c._to_device(self._module)

        self._fun = self._module.get_function(self._parloop._stub_name)
        self._fun.prepare(argtypes)
        # Blow away everything we don't need any more
        del self._parloop
        del self._kernel
        del self._config
        return self._fun

    @timed_function("ParLoop kernel")
    def __call__(self, grid, block, stream, *args, **kwargs):
        if configuration["profiling"]:
            t_ = self.compile().prepared_timed_call(grid, block, *args, **kwargs)()
            Timer("CUDA kernel").add(t_)
        else:
            self.compile().prepared_async_call(grid, block, stream, *args, **kwargs)


class ParLoop(op2.ParLoop):

    def launch_configuration(self, part):
        if self._is_direct:
            max_smem = self._max_shared_memory_needed_per_set_element
            smem_offset = max_smem * _WARPSIZE
            max_block = _device.get_attribute(driver.device_attribute.MAX_BLOCK_DIM_X)
            if max_smem == 0:
                block_size = max_block
            else:
                threads_per_sm = _AVAILABLE_SHARED_MEMORY / max_smem
                block_size = min(max_block, (threads_per_sm / _WARPSIZE) * _WARPSIZE)
            max_grid = _device.get_attribute(driver.device_attribute.MAX_GRID_DIM_X)
            grid_size = min(max_grid, (block_size + part.size) / block_size)

            grid_size = np.asscalar(np.int64(grid_size))
            block_size = (block_size, 1, 1)
            grid_size = (grid_size, 1, 1)

            required_smem = np.asscalar(max_smem * np.prod(block_size))
            return {'op2stride': self._it_space.size,
                    'smem_offset': smem_offset,
                    'WARPSIZE': _WARPSIZE,
                    'required_smem': required_smem,
                    'block_size': block_size,
                    'grid_size': grid_size}
        else:
            return {'op2stride': self._it_space.size,
                    'WARPSIZE': 32}

    @collective
    @lineprof
    def _compute(self, part, fun, *arglist):
        if part.size == 0:
            # Return before plan call if no computation should occur
            return
        arglist = [np.int32(part.size), np.int32(part.offset)]
        config = self.launch_configuration(part)
        config['subset'] = False
        if isinstance(part.set, Subset):
            config['subset'] = True
            part.set._allocate_device()
            arglist.append(np.intp(part.set._device_data.gpudata))

        fun = JITModule(self.kernel, self.it_space, *self.args, parloop=self, config=config)

        if self._is_direct:
            _args = self.args
            block_size = config['block_size']
            max_grid_size = config['grid_size']
            shared_size = config['required_smem']
        else:
            _args = self._unique_args
            maxbytes = sum([a.dtype.itemsize * a.data.cdim
                            for a in self._unwound_args if a._is_indirect])
            # shared memory as reported by the device, divided by some
            # factor.  This is the same calculation as done inside
            # op_plan_core, but without assuming 48K shared memory.
            # It would be much nicer if we could tell op_plan_core "I
            # have X bytes shared memory"
            part_size = (_AVAILABLE_SHARED_MEMORY / (64 * maxbytes)) * 64
            _plan = Plan(part,
                         *self._unwound_args,
                         partition_size=part_size)
            max_grid_size = _plan.ncolblk.max()

        for arg in _args:
            if arg._is_mat:
                d = arg.data._lmadata.gpudata
                itset = self._it_space.iterset
                if isinstance(itset, Subset):
                    itset = itset.superset
                offset = arg.data._lmaoffset(itset)
                arglist.append(np.intp(d))
                arglist.append(np.int32(offset))
            else:
                arg.data._allocate_device()
                if arg.access is not op2.WRITE:
                    arg.data._to_device()
                karg = arg.data._device_data
                if arg._is_global_reduction:
                    arg.data._allocate_reduction_buffer(max_grid_size,
                                                        arg.access)
                    karg = arg.data._reduction_buffer
                arglist.append(np.intp(karg.gpudata))

        if self._is_direct:
            _stream.synchronize()
            fun(max_grid_size, block_size, _stream, *arglist,
                shared_size=shared_size)
        else:
            arglist.append(_plan.ind_map.gpudata)
            arglist.append(_plan.loc_map.gpudata)
            arglist.append(_plan.ind_sizes.gpudata)
            arglist.append(_plan.ind_offs.gpudata)
            arglist.append(None)  # Block offset
            arglist.append(_plan.blkmap.gpudata)
            arglist.append(_plan.offset.gpudata)
            arglist.append(_plan.nelems.gpudata)
            arglist.append(_plan.nthrcol.gpudata)
            arglist.append(_plan.thrcol.gpudata)
            arglist.append(None)  # Number of colours in this block
            block_offset = 0

            for col in xrange(_plan.ncolors):
                blocks = _plan.ncolblk[col]
                if blocks > 0:
                    arglist[-1] = np.int32(blocks)
                    arglist[-7] = np.int32(block_offset)
                    blocks = np.asscalar(blocks)
                    # Compute capability < 3 can handle at most 2**16  - 1
                    # blocks in any one dimension of the grid.
                    if blocks >= 2 ** 16:
                        grid_size = (2 ** 16 - 1, (blocks - 1) / (2 ** 16 - 1) + 1, 1)
                    else:
                        grid_size = (blocks, 1, 1)

                    block_size = (128, 1, 1)
                    shared_size = np.asscalar(_plan.nsharedCol[col])
                    # Global reductions require shared memory of at least block
                    # size * sizeof(double) for the reduction buffer
                    if any(arg._is_global_reduction for arg in self.args):
                        shared_size = max(128 * 8, shared_size)

                    _stream.synchronize()
                    fun(grid_size, block_size, _stream, *arglist,
                        shared_size=shared_size)

                block_offset += blocks

        _stream.synchronize()
        for arg in self.args:
            if arg._is_global_reduction:
                arg.data._finalise_reduction_begin(max_grid_size, arg.access)
                arg.data._finalise_reduction_end(max_grid_size, arg.access)
            elif not arg._is_mat:
                # Data state is updated in finalise_reduction for Global
                if arg.access is not op2.READ:
                    arg.data.state = DeviceDataMixin.DEVICE


_device = None
_context = None
_WARPSIZE = 32
_AVAILABLE_SHARED_MEMORY = 0
_direct_loop_template = None
_indirect_loop_template = None
_matrix_support_template = None
_stream = None


def _setup():
    global _device
    global _context
    global _WARPSIZE
    global _AVAILABLE_SHARED_MEMORY
    global _stream
    if _device is None or _context is None:
        import pycuda.autoinit
        _device = pycuda.autoinit.device
        _context = pycuda.autoinit.context
        _WARPSIZE = _device.get_attribute(driver.device_attribute.WARP_SIZE)
        _AVAILABLE_SHARED_MEMORY = _device.get_attribute(
            driver.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
        _stream = driver.Stream()
    global _direct_loop_template
    global _indirect_loop_template
    global _matrix_support_template
    env = jinja2.Environment(loader=jinja2.PackageLoader('pyop2', 'assets'))
    if _direct_loop_template is None:
        _direct_loop_template = env.get_template('cuda_direct_loop.jinja2')

    if _indirect_loop_template is None:
        _indirect_loop_template = env.get_template('cuda_indirect_loop.jinja2')
    if _matrix_support_template is None:
        _matrix_support_template = env.get_template('cuda_matrix_support.jinja2')

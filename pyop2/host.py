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

"""Base classes extending those from the :mod:`base` module with functionality
common to backends executing on the host."""

from textwrap import dedent
from copy import deepcopy as dcopy

import base
import compilation
from base import *
# Override base ParLoop with flop-logging version in petsc_base
from petsc_base import ParLoop  # noqa: pass-through
from mpi import collective
from configuration import configuration
from utils import as_tuple, strip

import coffee.plan
from coffee.plan import ASTKernel


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to CPU execution."""
        self._original_ast = dcopy(ast)
        ast_handler = ASTKernel(ast, self._include_dirs)
        ast_handler.plan_cpu(self._opts)
        self._applied_blas = ast_handler.blas
        return ast_handler.gencode()


class Arg(base.Arg):

    def c_arg_name(self, i=0, j=None):
        name = self.name
        if self._is_indirect and not (self._is_vec_map or self._uses_itspace):
            name = "%s_%d" % (name, self.idx)
        if i is not None:
            # For a mixed ParLoop we can't necessarily assume all arguments are
            # also mixed. If that's not the case we want index 0.
            if not self._is_mat and len(self.data) == 1:
                i = 0
            name += "_%d" % i
        if j is not None:
            name += "_%d" % j
        return name

    def c_vec_name(self):
        return self.c_arg_name() + "_vec"

    def c_map_name(self, i, j):
        return self.c_arg_name() + "_map%d_%d" % (i, j)

    def c_offset_name(self, i, j):
        return self.c_arg_name() + "_off%d_%d" % (i, j)

    def c_wrapper_arg(self):
        if self._is_mat:
            val = "Mat %s_" % self.c_arg_name()
        else:
            val = ', '.join(["%s *%s" % (self.ctype, self.c_arg_name(i))
                             for i in range(len(self.data))])
        if self._is_indirect or self._is_mat:
            for i, map in enumerate(as_tuple(self.map, Map)):
                for j, m in enumerate(map):
                    val += ", int *%s" % self.c_map_name(i, j)
        return val

    def c_vec_dec(self, is_facet=False):
        facet_mult = 2 if is_facet else 1
        cdim = self.data.cdim if self._flatten else 1
        return "%(type)s *%(vec_name)s[%(arity)s];\n" % \
            {'type': self.ctype,
             'vec_name': self.c_vec_name(),
             'arity': self.map.arity * cdim * facet_mult}

    def c_wrapper_dec(self):
        val = ""
        if self._is_mixed_mat:
            rows, cols = self.data.sparsity.shape
            for i in range(rows):
                for j in range(cols):
                    val += "Mat %(iname)s; MatNestGetSubMat(%(name)s_, %(i)d, %(j)d, &%(iname)s);\n" \
                        % {'name': self.c_arg_name(),
                           'iname': self.c_arg_name(i, j),
                           'i': i,
                           'j': j}
        elif self._is_mat:
            val += "Mat %(iname)s = %(name)s_;\n" % {'name': self.c_arg_name(),
                                                     'iname': self.c_arg_name(0, 0)}
        return val

    def c_ind_data(self, idx, i, j=0, is_top=False, offset=None):
        return "%(name)s + (%(map_name)s[i * %(arity)s + %(idx)s]%(top)s%(off_mul)s%(off_add)s)* %(dim)s%(off)s" % \
            {'name': self.c_arg_name(i),
             'map_name': self.c_map_name(i, 0),
             'arity': self.map.split[i].arity,
             'idx': idx,
             'top': ' + start_layer' if is_top else '',
             'dim': self.data[i].cdim,
             'off': ' + %d' % j if j else '',
             'off_mul': ' * %d' % offset if is_top and offset is not None else '',
             'off_add': ' + %d' % offset if not is_top and offset is not None else ''}

    def c_ind_data_xtr(self, idx, i, j=0):
        return "%(name)s + (xtr_%(map_name)s[%(idx)s])*%(dim)s%(off)s" % \
            {'name': self.c_arg_name(i),
             'map_name': self.c_map_name(i, 0),
             'idx': idx,
             'dim': 1 if self._flatten else str(self.data[i].cdim),
             'off': ' + %d' % j if j else ''}

    def c_kernel_arg_name(self, i, j):
        return "p_%s" % self.c_arg_name(i, j)

    def c_global_reduction_name(self, count=None):
        return self.c_arg_name()

    def c_local_tensor_name(self, i, j):
        return self.c_kernel_arg_name(i, j)

    def c_kernel_arg(self, count, i=0, j=0, shape=(0,), layers=1):
        if self._is_dat_view and not self._is_direct:
            raise NotImplementedError("Indirect DatView not implemented")
        if self._uses_itspace:
            if self._is_mat:
                if self.data[i, j]._is_vector_field:
                    return self.c_kernel_arg_name(i, j)
                elif self.data[i, j]._is_scalar_field:
                    return "(%(t)s (*)[%(dim)d])&%(name)s" % \
                        {'t': self.ctype,
                         'dim': shape[0],
                         'name': self.c_kernel_arg_name(i, j)}
                else:
                    raise RuntimeError("Don't know how to pass kernel arg %s" % self)
            else:
                if self.data is not None and self.data.dataset._extruded:
                    return self.c_ind_data_xtr("i_%d" % self.idx.index, i)
                elif self._flatten:
                    return "%(name)s + %(map_name)s[i * %(arity)s + i_0 %% %(arity)d] * %(dim)s + (i_0 / %(arity)d)" % \
                        {'name': self.c_arg_name(),
                         'map_name': self.c_map_name(0, i),
                         'arity': self.map.arity,
                         'dim': self.data[i].cdim}
                else:
                    return self.c_ind_data("i_%d" % self.idx.index, i)
        elif self._is_indirect:
            if self._is_vec_map:
                return self.c_vec_name()
            return self.c_ind_data(self.idx, i)
        elif self._is_global_reduction:
            return self.c_global_reduction_name(count)
        elif isinstance(self.data, Global):
            return self.c_arg_name(i)
        else:
            if self._is_dat_view:
                idx = "(%(idx)s + i * %(dim)s)" % {'idx': self.data[i].index,
                                                   'dim': super(DatView, self.data[i]).cdim}
            else:
                idx = "(i * %(dim)s)" % {'dim': self.data[i].cdim}
            return "%(name)s + %(idx)s" % {'name': self.c_arg_name(i),
                                           'idx': idx}

    def c_vec_init(self, is_top, is_facet=False):
        is_top_init = is_top
        val = []
        vec_idx = 0
        for i, (m, d) in enumerate(zip(self.map, self.data)):
            is_top = is_top_init and m.iterset._extruded
            if self._flatten:
                for k in range(d.cdim):
                    for idx in range(m.arity):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                   {'vec_name': self.c_vec_name(),
                                    'idx': vec_idx,
                                    'data': self.c_ind_data(idx, i, k, is_top=is_top,
                                                            offset=m.offset[idx] if is_top else None)})
                        vec_idx += 1
                    # In the case of interior horizontal facets the map for the
                    # vertical does not exist so it has to be dynamically
                    # created by adding the offset to the map of the current
                    # cell. In this way the only map required is the one for
                    # the bottom layer of cells and the wrapper will make sure
                    # to stage in the data for the entire map spanning the facet.
                    if is_facet:
                        for idx in range(m.arity):
                            val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                       {'vec_name': self.c_vec_name(),
                                        'idx': vec_idx,
                                        'data': self.c_ind_data(idx, i, k, is_top=is_top,
                                                                offset=m.offset[idx])})
                            vec_idx += 1
            else:
                for idx in range(m.arity):
                    val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                               {'vec_name': self.c_vec_name(),
                                'idx': vec_idx,
                                'data': self.c_ind_data(idx, i, is_top=is_top,
                                                        offset=m.offset[idx] if is_top else None)})
                    vec_idx += 1
                if is_facet:
                    for idx in range(m.arity):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                   {'vec_name': self.c_vec_name(),
                                    'idx': vec_idx,
                                    'data': self.c_ind_data(idx, i, is_top=is_top,
                                                            offset=m.offset[idx])})
                        vec_idx += 1
        return ";\n".join(val)

    def c_addto(self, i, j, buf_name, tmp_name, tmp_decl,
                extruded=None, is_facet=False, applied_blas=False):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rows_str = "%s + i * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + i * %s" % (self.c_map_name(1, j), ncols)

        if extruded is not None:
            rows_str = extruded + self.c_map_name(0, i)
            cols_str = extruded + self.c_map_name(1, j)

        if is_facet:
            nrows *= 2
            ncols *= 2

        ret = []
        rbs, cbs = self.data.sparsity[i, j].dims[0][0]
        rdim = rbs * nrows
        addto_name = buf_name
        addto = 'MatSetValuesLocal'
        if self.data._is_vector_field:
            addto = 'MatSetValuesBlockedLocal'
            if self._flatten:
                if applied_blas:
                    idx = "[(%%(ridx)s)*%d + (%%(cidx)s)]" % rdim
                else:
                    idx = "[%(ridx)s][%(cidx)s]"
                ret = []
                idx_l = idx % {'ridx': "%d*j + k" % rbs,
                               'cidx': "%d*l + m" % cbs}
                idx_r = idx % {'ridx': "j + %d*k" % nrows,
                               'cidx': "l + %d*m" % ncols}
                # Shuffle xxx yyy zzz into xyz xyz xyz
                ret = ["""
                %(tmp_decl)s;
                for ( int j = 0; j < %(nrows)d; j++ ) {
                   for ( int k = 0; k < %(rbs)d; k++ ) {
                      for ( int l = 0; l < %(ncols)d; l++ ) {
                         for ( int m = 0; m < %(cbs)d; m++ ) {
                            %(tmp_name)s%(idx_l)s = %(buf_name)s%(idx_r)s;
                         }
                      }
                   }
                }""" % {'nrows': nrows,
                        'ncols': ncols,
                        'rbs': rbs,
                        'cbs': cbs,
                        'idx_l': idx_l,
                        'idx_r': idx_r,
                        'buf_name': buf_name,
                        'tmp_decl': tmp_decl,
                        'tmp_name': tmp_name}]
                addto_name = tmp_name

            rmap, cmap = maps
            rdim, cdim = self.data.dims[i][j]
            if rmap.vector_index is not None or cmap.vector_index is not None:
                rows_str = "rowmap"
                cols_str = "colmap"
                addto = "MatSetValuesLocal"
                fdict = {'nrows': nrows,
                         'ncols': ncols,
                         'rdim': rdim,
                         'cdim': cdim,
                         'rowmap': self.c_map_name(0, i),
                         'colmap': self.c_map_name(1, j),
                         'drop_full_row': 0 if rmap.vector_index is not None else 1,
                         'drop_full_col': 0 if cmap.vector_index is not None else 1}
                # Horrible hack alert
                # To apply BCs to a component of a Dat with cdim > 1
                # we encode which components to apply things to in the
                # high bits of the map value
                # The value that comes in is:
                # -(row + 1 + sum_i 2 ** (30 - i))
                # where i are the components to zero
                #
                # So, the actual row (if it's negative) is:
                # (~input) & ~0x70000000
                # And we can determine which components to zero by
                # inspecting the high bits (1 << 30 - i)
                ret.append("""
                PetscInt rowmap[%(nrows)d*%(rdim)d];
                PetscInt colmap[%(ncols)d*%(cdim)d];
                int discard, tmp, block_row, block_col;
                for ( int j = 0; j < %(nrows)d; j++ ) {
                    block_row = %(rowmap)s[i*%(nrows)d + j];
                    discard = 0;
                    if ( block_row < 0 ) {
                        tmp = -(block_row + 1);
                        discard = 1;
                        block_row = tmp & ~0x70000000;
                    }
                    for ( int k = 0; k < %(rdim)d; k++ ) {
                        if ( discard && (%(drop_full_row)d || ((tmp & (1 << (30 - k))) != 0)) ) {
                            rowmap[j*%(rdim)d + k] = -1;
                        } else {
                            rowmap[j*%(rdim)d + k] = (block_row)*%(rdim)d + k;
                        }
                    }
                }
                for ( int j = 0; j < %(ncols)d; j++ ) {
                    discard = 0;
                    block_col = %(colmap)s[i*%(ncols)d + j];
                    if ( block_col < 0 ) {
                        tmp = -(block_col + 1);
                        discard = 1;
                        block_col = tmp & ~0x70000000;
                    }
                    for ( int k = 0; k < %(cdim)d; k++ ) {
                        if ( discard && (%(drop_full_col)d || ((tmp & (1 << (30 - k))) != 0)) ) {
                            colmap[j*%(rdim)d + k] = -1;
                        } else {
                            colmap[j*%(cdim)d + k] = (block_col)*%(cdim)d + k;
                        }
                    }
                }
                """ % fdict)
                nrows *= rdim
                ncols *= cdim
        ret.append("""%(addto)s(%(mat)s, %(nrows)s, %(rows)s,
                                         %(ncols)s, %(cols)s,
                                         (const PetscScalar *)%(vals)s,
                                         %(insert)s);""" %
                   {'mat': self.c_arg_name(i, j),
                    'vals': addto_name,
                    'addto': addto,
                    'nrows': nrows,
                    'ncols': ncols,
                    'rows': rows_str,
                    'cols': cols_str,
                    'insert': "INSERT_VALUES" if self.access == WRITE else "ADD_VALUES"})
        return "\n".join(ret)

    def c_local_tensor_dec(self, extents, i, j):
        if self._is_mat:
            size = 1
        else:
            size = self.data.split[i].cdim
        return tuple([d * size for d in extents])

    def c_zero_tmp(self, i, j):
        t = self.ctype
        if self.data[i, j]._is_scalar_field:
            idx = ''.join(["[i_%d]" % ix for ix in range(len(self.data.dims))])
            return "%(name)s%(idx)s = (%(t)s)0" % \
                {'name': self.c_kernel_arg_name(i, j), 't': t, 'idx': idx}
        elif self.data[i, j]._is_vector_field:
            if self._flatten:
                return "%(name)s[0][0] = (%(t)s)0" % \
                    {'name': self.c_kernel_arg_name(i, j), 't': t}
            size = np.prod(self.data[i, j].dims)
            return "memset(%(name)s, 0, sizeof(%(t)s) * %(size)s)" % \
                {'name': self.c_kernel_arg_name(i, j), 't': t, 'size': size}
        else:
            raise RuntimeError("Don't know how to zero temp array for %s" % self)

    def c_add_offset(self, is_facet=False):
        if not self.map.iterset._extruded:
            return ""
        val = []
        vec_idx = 0
        for i, (m, d) in enumerate(zip(self.map, self.data)):
            for k in range(d.cdim if self._flatten else 1):
                for idx in range(m.arity):
                    val.append("%(name)s[%(j)d] += %(offset)d * %(dim)s;" %
                               {'name': self.c_vec_name(),
                                'j': vec_idx,
                                'offset': m.offset[idx],
                                'dim': d.cdim})
                    vec_idx += 1
                if is_facet:
                    for idx in range(m.arity):
                        val.append("%(name)s[%(j)d] += %(offset)d * %(dim)s;" %
                                   {'name': self.c_vec_name(),
                                    'j': vec_idx,
                                    'offset': m.offset[idx],
                                    'dim': d.cdim})
                        vec_idx += 1
        return '\n'.join(val)+'\n'

    # New globals generation which avoids false sharing.
    def c_intermediate_globals_decl(self, count):
        return "%(type)s %(name)s_l%(count)s[1][%(dim)s]" % \
            {'type': self.ctype,
             'name': self.c_arg_name(),
             'count': str(count),
             'dim': self.data.cdim}

    def c_intermediate_globals_init(self, count):
        if self.access == INC:
            init = "(%(type)s)0" % {'type': self.ctype}
        else:
            init = "%(name)s[i]" % {'name': self.c_arg_name()}
        return "for ( int i = 0; i < %(dim)s; i++ ) %(name)s_l%(count)s[0][i] = %(init)s" % \
            {'dim': self.data.cdim,
             'name': self.c_arg_name(),
             'count': str(count),
             'init': init}

    def c_intermediate_globals_writeback(self, count):
        d = {'gbl': self.c_arg_name(),
             'local': "%(name)s_l%(count)s[0][i]" %
             {'name': self.c_arg_name(), 'count': str(count)}}
        if self.access == INC:
            combine = "%(gbl)s[i] += %(local)s" % d
        elif self.access == MIN:
            combine = "%(gbl)s[i] = %(gbl)s[i] < %(local)s ? %(gbl)s[i] : %(local)s" % d
        elif self.access == MAX:
            combine = "%(gbl)s[i] = %(gbl)s[i] > %(local)s ? %(gbl)s[i] : %(local)s" % d
        return """
#pragma omp critical
for ( int i = 0; i < %(dim)s; i++ ) %(combine)s;
""" % {'combine': combine, 'dim': self.data.cdim}

    def c_map_decl(self, is_facet=False):
        if self._is_mat:
            dsets = self.data.sparsity.dsets
        else:
            dsets = (self.data.dataset,)
        val = []
        for i, (map, dset) in enumerate(zip(as_tuple(self.map, Map), dsets)):
            for j, (m, d) in enumerate(zip(map, dset)):
                dim = m.arity
                if self._is_dat and self._flatten:
                    dim *= d.cdim
                if is_facet:
                    dim *= 2
                val.append("int xtr_%(name)s[%(dim)s];" %
                           {'name': self.c_map_name(i, j), 'dim': dim})
        return '\n'.join(val)+'\n'

    def c_map_init(self, is_top=False, is_facet=False):
        if self._is_mat:
            dsets = self.data.sparsity.dsets
        else:
            dsets = (self.data.dataset,)
        val = []
        for i, (map, dset) in enumerate(zip(as_tuple(self.map, Map), dsets)):
            for j, (m, d) in enumerate(zip(map, dset)):
                for idx in range(m.arity):
                    if self._is_dat and self._flatten and d.cdim > 1:
                        for k in range(d.cdim):
                            val.append("xtr_%(name)s[%(ind_flat)s] = %(dat_dim)s * (*(%(name)s + i * %(dim)s + %(ind)s)%(off_top)s)%(offset)s;" %
                                       {'name': self.c_map_name(i, j),
                                        'dim': m.arity,
                                        'ind': idx,
                                        'dat_dim': d.cdim,
                                        'ind_flat': (2 if is_facet else 1) * m.arity * k + idx,
                                        'offset': ' + '+str(k) if k > 0 else '',
                                        'off_top': ' + start_layer * '+str(m.offset[idx]) if is_top else ''})
                    else:
                        val.append("xtr_%(name)s[%(ind)s] = *(%(name)s + i * %(dim)s + %(ind)s)%(off_top)s;" %
                                   {'name': self.c_map_name(i, j),
                                    'dim': m.arity,
                                    'ind': idx,
                                    'off_top': ' + start_layer * '+str(m.offset[idx]) if is_top else ''})
                if is_facet:
                    for idx in range(m.arity):
                        if self._is_dat and self._flatten and d.cdim > 1:
                            for k in range(d.cdim):
                                val.append("xtr_%(name)s[%(ind_flat)s] = %(dat_dim)s * (*(%(name)s + i * %(dim)s + %(ind)s)%(off)s)%(offset)s;" %
                                           {'name': self.c_map_name(i, j),
                                            'dim': m.arity,
                                            'ind': idx,
                                            'dat_dim': d.cdim,
                                            'ind_flat': m.arity * (k * 2 + 1) + idx,
                                            'offset': ' + '+str(k) if k > 0 else '',
                                            'off': ' + ' + str(m.offset[idx])})
                        else:
                            val.append("xtr_%(name)s[%(ind)s] = *(%(name)s + i * %(dim)s + %(ind_zero)s)%(off_top)s%(off)s;" %
                                       {'name': self.c_map_name(i, j),
                                        'dim': m.arity,
                                        'ind': idx + m.arity,
                                        'ind_zero': idx,
                                        'off_top': ' + start_layer' if is_top else '',
                                        'off': ' + ' + str(m.offset[idx])})
        return '\n'.join(val)+'\n'

    def c_map_bcs(self, sign, is_facet):
        maps = as_tuple(self.map, Map)
        val = []
        # To throw away boundary condition values, we subtract a large
        # value from the map to make it negative then add it on later to
        # get back to the original
        max_int = 10000000

        need_bottom = False
        # Apply any bcs on the first (bottom) layer
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                bottom_masks = None
                for location, name in m.implicit_bcs:
                    if location == "bottom":
                        if bottom_masks is None:
                            bottom_masks = m.bottom_mask[name].copy()
                        else:
                            bottom_masks += m.bottom_mask[name]
                        need_bottom = True
                if bottom_masks is not None:
                    for idx in range(m.arity):
                        if bottom_masks[idx] < 0:
                            val.append("xtr_%(name)s[%(ind)s] %(sign)s= %(val)s;" %
                                       {'name': self.c_map_name(i, j),
                                        'val': max_int,
                                        'ind': idx,
                                        'sign': sign})
        if need_bottom:
            val.insert(0, "if (j_0 == 0) {")
            val.append("}")

        need_top = False
        pos = len(val)
        # Apply any bcs on last (top) layer
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                top_masks = None
                for location, name in m.implicit_bcs:
                    if location == "top":
                        if top_masks is None:
                            top_masks = m.top_mask[name].copy()
                        else:
                            top_masks += m.top_mask[name]
                        need_top = True
                if top_masks is not None:
                    facet_offset = m.arity if is_facet else 0
                    for idx in range(m.arity):
                        if top_masks[idx] < 0:
                            val.append("xtr_%(name)s[%(ind)s] %(sign)s= %(val)s;" %
                                       {'name': self.c_map_name(i, j),
                                        'val': max_int,
                                        'ind': idx + facet_offset,
                                        'sign': sign})
        if need_top:
            val.insert(pos, "if (j_0 == top_layer - 1) {")
            val.append("}")
        return '\n'.join(val)+'\n'

    def c_add_offset_map(self, is_facet=False):
        if self._is_mat:
            dsets = self.data.sparsity.dsets
        else:
            dsets = (self.data.dataset,)
        val = []
        for i, (map, dset) in enumerate(zip(as_tuple(self.map, Map), dsets)):
            if not map.iterset._extruded:
                continue
            for j, (m, d) in enumerate(zip(map, dset)):
                for idx in range(m.arity):
                    if self._is_dat and self._flatten and d.cdim > 1:
                        for k in range(d.cdim):
                            val.append("xtr_%(name)s[%(ind_flat)s] += %(off)d * %(dim)s;" %
                                       {'name': self.c_map_name(i, j),
                                        'off': m.offset[idx],
                                        'ind_flat': m.arity * k + idx,
                                        'dim': d.cdim})
                    else:
                        val.append("xtr_%(name)s[%(ind)s] += %(off)d;" %
                                   {'name': self.c_map_name(i, j),
                                    'off': m.offset[idx],
                                    'ind': idx})
                if is_facet:
                    for idx in range(m.arity):
                        if self._is_dat and self._flatten and d.cdim > 1:
                            for k in range(d.cdim):
                                val.append("xtr_%(name)s[%(ind_flat)s] += %(off)d * %(dim)s;" %
                                           {'name': self.c_map_name(i, j),
                                            'off': m.offset[idx],
                                            'ind_flat': m.arity * (k + d.cdim) + idx,
                                            'dim': d.cdim})
                        else:
                            val.append("xtr_%(name)s[%(ind)s] += %(off)d;" %
                                       {'name': self.c_map_name(i, j),
                                        'off': m.offset[idx],
                                        'ind': m.arity + idx})
        return '\n'.join(val)+'\n'

    def c_buffer_decl(self, size, idx, buf_name, is_facet=False, init=True):
        buf_type = self.data.ctype
        dim = len(size)
        compiler = coffee.plan.compiler
        isa = coffee.plan.isa
        align = compiler['align'](isa["alignment"]) if compiler and size[-1] % isa["dp_reg"] == 0 else ""
        init_expr = " = " + "{" * dim + "0.0" + "}" * dim if self.access in [WRITE, INC] else ""
        if not init:
            init_expr = ""

        return "%(typ)s %(name)s%(dim)s%(align)s%(init)s" % \
            {"typ": buf_type,
             "name": buf_name,
             "dim": "".join(["[%d]" % (d * (2 if is_facet else 1)) for d in size]),
             "align": " " + align,
             "init": init_expr}

    def c_buffer_gather(self, size, idx, buf_name):
        dim = 1 if self._flatten else self.data.cdim
        return ";\n".join(["%(name)s[i_0*%(dim)d%(ofs)s] = *(%(ind)s%(ofs)s);\n" %
                           {"name": buf_name,
                            "dim": dim,
                            "ind": self.c_kernel_arg(idx),
                            "ofs": " + %s" % j if j else ""} for j in range(dim)])

    def c_buffer_scatter_vec(self, count, i, j, mxofs, buf_name):
        dim = self.data.split[i].cdim
        return ";\n".join(["*(%(ind)s%(nfofs)s) %(op)s %(name)s[i_0*%(dim)d%(nfofs)s%(mxofs)s]" %
                           {"ind": self.c_kernel_arg(count, i, j),
                            "op": "=" if self.access == WRITE else "+=",
                            "name": buf_name,
                            "dim": dim,
                            "nfofs": " + %d" % o if o else "",
                            "mxofs": " + %d" % (mxofs[0] * dim) if mxofs else ""}
                           for o in range(dim)])

    def c_buffer_scatter_offset(self, count, i, j, ofs_name):
        if self.data.dataset._extruded:
            return '%(ofs_name)s = %(map_name)s[i_0]' % {
                'ofs_name': ofs_name,
                'map_name': 'xtr_%s' % self.c_map_name(0, i),
            }
        else:
            return '%(ofs_name)s = %(map_name)s[i * %(arity)d + i_0] * %(dim)s' % {
                'ofs_name': ofs_name,
                'map_name': self.c_map_name(0, i),
                'arity': self.map.arity,
                'dim': self.data.split[i].cdim
            }

    def c_buffer_scatter_vec_flatten(self, count, i, j, mxofs, buf_name, ofs_name, loop_size):
        dim = self.data.split[i].cdim
        return ";\n".join(["%(name)s[%(ofs_name)s%(nfofs)s] %(op)s %(buf_name)s[i_0%(buf_ofs)s%(mxofs)s]" %
                           {"name": self.c_arg_name(),
                            "op": "=" if self.access == WRITE else "+=",
                            "buf_name": buf_name,
                            "ofs_name": ofs_name,
                            "nfofs": " + %d" % o,
                            "buf_ofs": " + %d" % (o*loop_size,),
                            "mxofs": " + %d" % (mxofs[0] * dim) if mxofs else ""}
                           for o in range(dim)])


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []
    _system_headers = []
    _extension = 'c'

    def __init__(self, kernel, itspace, *args, **kwargs):
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
        # Return early if we were in the cache.
        if self._initialized:
            return
        self._kernel = kernel
        self._fun = None
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._iteration_region = kwargs.get('iterate', ALL)
        self._initialized = True
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        self.set_argtypes(itspace.iterset, *args)
        self.compile()

    @collective
    def __call__(self, *args):
        return self._fun(*args)

    @property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @collective
    def compile(self):
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = coffee.plan.compiler
        blas = coffee.plan.blas
        blas_header, blas_namespace, externc_open, externc_close = ("", "", "", "")
        if self._kernel._applied_blas:
            blas_header = blas.get('header')
            blas_namespace = blas.get('namespace', '')
            if blas['name'] == 'eigen':
                externc_open = 'extern "C" {'
                externc_close = '}'
        if self._kernel._cpp:
            externc_open = 'extern "C" {'
            externc_close = '}'
        headers = "\n".join([compiler.get('vect_header', ""), blas_header])
        if any(arg._is_soa for arg in self._args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            %(header)s
            %(namespace)s
            %(code)s
            #undef OP2_STRIDE
            """ % {'code': self._kernel.code(),
                   'namespace': blas_namespace,
                   'header': headers}
        else:
            kernel_code = """
            %(header)s
            %(namespace)s
            %(code)s
            """ % {'code': self._kernel.code(),
                   'namespace': blas_namespace,
                   'header': headers}
        code_to_compile = strip(dedent(self._wrapper) % self.generate_code())

        _const_decs = '\n'.join([const._format_declaration()
                                for const in Const._definitions()]) + '\n'

        code_to_compile = """
        #include <petsc.h>
        #include <stdbool.h>
        #include <math.h>
        %(sys_headers)s
        %(consts)s

        %(kernel)s

        %(externc_open)s
        %(wrapper)s
        %(externc_close)s
        """ % {'consts': _const_decs, 'kernel': kernel_code,
               'wrapper': code_to_compile,
               'externc_open': externc_open,
               'externc_close': externc_close,
               'sys_headers': '\n'.join(self._kernel._headers + self._system_headers)}

        self._dump_generated_code(code_to_compile)
        if configuration["debug"]:
            self._wrapper_code = code_to_compile

        extension = self._extension
        cppargs = self._cppargs
        cppargs += ["-I%s/include" % d for d in get_petsc_dir()] + \
                   ["-I%s" % d for d in self._kernel._include_dirs] + \
                   ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        if compiler:
            cppargs += [compiler[coffee.plan.isa['inst_set']]]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        if self._kernel._applied_blas:
            blas_dir = blas['dir']
            if blas_dir:
                cppargs += ["-I%s/include" % blas_dir]
                ldargs += ["-L%s/lib" % blas_dir]
            ldargs += blas['link']
            if blas['name'] == 'eigen':
                extension = "cpp"
        ldargs += self._kernel._ldargs

        if self._kernel._cpp:
            extension = "cpp"
        self._fun = compilation.load(code_to_compile,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     argtypes=self._argtypes,
                                     restype=None,
                                     compiler=compiler.get('name'))
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._itspace
        del self._direct
        return self._fun

    def generate_code(self):
        snippets = wrapper_snippets(self._itspace, self._args,
                                    kernel_name=self._kernel._name,
                                    user_code=self._kernel._user_code,
                                    wrapper_name=self._wrapper_name,
                                    iteration_region=self._iteration_region,
                                    applied_blas=self._kernel._applied_blas)
        return snippets


def wrapper_snippets(itspace, args,
                     kernel_name=None, wrapper_name=None, user_code=None,
                     iteration_region=ALL, applied_blas=False):
    """Generates code snippets for the wrapper,
    ready to be into a template.

    :param itspace: :class:`IterationSpace` object of the :class:`ParLoop`,
                    This is built from the iteration :class:`Set`.
    :param args: :class:`Arg`s of the :class:`ParLoop`
    :param kernel_name: Kernel function name (forwarded)
    :param user_code: Code to insert into the wrapper (forwarded)
    :param wrapper_name: Wrapper function name (forwarded)
    :param iteration_region: Iteration region, this is specified when
                             creating a :class:`ParLoop`.
    :param applied_blas: COFFEE sometimes sets this true.

    :return: dict containing the code snippets
    """

    assert kernel_name is not None
    if wrapper_name is None:
        wrapper_name = "wrap_" + kernel_name
    if user_code is None:
        user_code = ""

    direct = all(a.map is None for a in args)

    def itspace_loop(i, d):
        return "for (int i_%d=0; i_%d<%d; ++i_%d) {" % (i, i, d, i)

    def c_const_arg(c):
        return '%s *%s_' % (c.ctype, c.name)

    def c_const_init(c):
        d = {'name': c.name,
             'type': c.ctype}
        if c.cdim == 1:
            return '%(name)s = *%(name)s_' % d
        tmp = '%(name)s[%%(i)s] = %(name)s_[%%(i)s]' % d
        return ';\n'.join([tmp % {'i': i} for i in range(c.cdim)])

    def extrusion_loop():
        if direct:
            return "{"
        return "for (int j_0 = start_layer; j_0 < end_layer; ++j_0){"

    _ssinds_arg = ""
    _index_expr = "n"
    is_top = (iteration_region == ON_TOP)
    is_facet = (iteration_region == ON_INTERIOR_FACETS)

    if isinstance(itspace._iterset, Subset):
        _ssinds_arg = "int* ssinds,"
        _index_expr = "ssinds[n]"

    _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in args])

    # Pass in the is_facet flag to mark the case when it's an interior horizontal facet in
    # an extruded mesh.
    _wrapper_decs = ';\n'.join([arg.c_wrapper_dec() for arg in args])

    _vec_decs = ';\n'.join([arg.c_vec_dec(is_facet=is_facet) for arg in args if arg._is_vec_map])

    if len(Const._defs) > 0:
        _const_args = ', '
        _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
    else:
        _const_args = ''
    _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

    _intermediate_globals_decl = ';\n'.join(
        [arg.c_intermediate_globals_decl(count)
         for count, arg in enumerate(args)
         if arg._is_global_reduction])
    _intermediate_globals_init = ';\n'.join(
        [arg.c_intermediate_globals_init(count)
         for count, arg in enumerate(args)
         if arg._is_global_reduction])
    _intermediate_globals_writeback = ';\n'.join(
        [arg.c_intermediate_globals_writeback(count)
         for count, arg in enumerate(args)
         if arg._is_global_reduction])

    _vec_inits = ';\n'.join([arg.c_vec_init(is_top, is_facet=is_facet) for arg in args
                             if not arg._is_mat and arg._is_vec_map])

    indent = lambda t, i: ('\n' + '  ' * i).join(t.split('\n'))

    _map_decl = ""
    _apply_offset = ""
    _map_init = ""
    _extr_loop = ""
    _extr_loop_close = ""
    _map_bcs_m = ""
    _map_bcs_p = ""
    _layer_arg = ""
    if itspace._extruded:
        _layer_arg = ", int start_layer, int end_layer, int top_layer"
        _map_decl += ';\n'.join([arg.c_map_decl(is_facet=is_facet)
                                 for arg in args if arg._uses_itspace])
        _map_init += ';\n'.join([arg.c_map_init(is_top=is_top, is_facet=is_facet)
                                 for arg in args if arg._uses_itspace])
        _map_bcs_m += ';\n'.join([arg.c_map_bcs("-", is_facet) for arg in args if arg._is_mat])
        _map_bcs_p += ';\n'.join([arg.c_map_bcs("+", is_facet) for arg in args if arg._is_mat])
        _apply_offset += ';\n'.join([arg.c_add_offset_map(is_facet=is_facet)
                                     for arg in args if arg._uses_itspace])
        _apply_offset += ';\n'.join([arg.c_add_offset(is_facet=is_facet)
                                     for arg in args if arg._is_vec_map])
        _extr_loop = '\n' + extrusion_loop()
        _extr_loop_close = '}\n'

    # Build kernel invocation. Let X be a parameter of the kernel representing a
    # tensor accessed in an iteration space. Let BUFFER be an array of the same
    # size as X.  BUFFER is declared and intialized in the wrapper function.
    # In particular, if:
    # - X is written or incremented, then BUFFER is initialized to 0
    # - X is read, then BUFFER gathers data expected by X
    _buf_name, _buf_decl, _buf_gather, _tmp_decl, _tmp_name = {}, {}, {}, {}, {}
    for count, arg in enumerate(args):
        if not arg._uses_itspace:
            continue
        _buf_name[arg] = "buffer_%s" % arg.c_arg_name(count)
        _tmp_name[arg] = "tmp_%s" % _buf_name[arg]
        _buf_size = list(itspace._extents)
        if not arg._is_mat:
            # Readjust size to take into account the size of a vector space
            _dat_size = (arg.data.cdim, )
            # Only adjust size if not flattening (in which case the buffer is extents*dat.dim)
            if not arg._flatten:
                _buf_size = [sum([e*d for e, d in zip(_buf_size, _dat_size)])]
                _loop_size = [_buf_size[i]/_dat_size[i] for i in range(len(_buf_size))]
            else:
                _buf_size = [sum(_buf_size)]
                _loop_size = _buf_size
        else:
            if applied_blas:
                _buf_size = [reduce(lambda x, y: x*y, _buf_size)]
        _buf_decl[arg] = arg.c_buffer_decl(_buf_size, count, _buf_name[arg], is_facet=is_facet)
        _tmp_decl[arg] = arg.c_buffer_decl(_buf_size, count, _tmp_name[arg], is_facet=is_facet,
                                           init=False)
        if arg.access not in [WRITE, INC]:
            _itspace_loops = '\n'.join(['  ' * n + itspace_loop(n, e) for n, e in enumerate(_loop_size)])
            _buf_gather[arg] = arg.c_buffer_gather(_buf_size, count, _buf_name[arg])
            _itspace_loop_close = '\n'.join('  ' * n + '}' for n in range(len(_loop_size) - 1, -1, -1))
            _buf_gather[arg] = "\n".join([_itspace_loops, _buf_gather[arg], _itspace_loop_close])
    _kernel_args = ', '.join([arg.c_kernel_arg(count) if not arg._uses_itspace else _buf_name[arg]
                              for count, arg in enumerate(args)])
    _buf_gather = ";\n".join(_buf_gather.values())
    _buf_decl = ";\n".join(_buf_decl.values())

    def itset_loop_body(i, j, shape, offsets, is_facet=False):
        template_scatter = """
    %(offset_decl)s;
    %(ofs_itspace_loops)s
    %(ind)s%(offset)s
    %(ofs_itspace_loop_close)s
    %(itspace_loops)s
    %(ind)s%(buffer_scatter)s;
    %(itspace_loop_close)s
"""
        nloops = len(shape)
        mult = 1 if not is_facet else 2
        _buf_scatter = {}
        for count, arg in enumerate(args):
            if not (arg._uses_itspace and arg.access in [WRITE, INC]):
                continue
            elif (arg._is_mat and arg._is_mixed) or (arg._is_dat and nloops > 1):
                raise NotImplementedError
            elif arg._is_mat:
                continue
            elif arg._is_dat and not arg._flatten:
                shape = shape[0]
                loop_size = shape*mult
                _itspace_loops, _itspace_loop_close = itspace_loop(0, loop_size), '}'
                _scatter_stmts = arg.c_buffer_scatter_vec(count, i, j, offsets, _buf_name[arg])
                _buf_offset, _buf_offset_decl = '', ''
            elif arg._is_dat:
                dim, shape = arg.data.split[i].cdim, shape[0]
                loop_size = shape*mult/dim
                _itspace_loops, _itspace_loop_close = itspace_loop(0, loop_size), '}'
                _buf_offset_name = 'offset_%d[%s]' % (count, '%s')
                _buf_offset_decl = 'int %s' % _buf_offset_name % loop_size
                _buf_offset_array = _buf_offset_name % 'i_0'
                _buf_offset = '%s;' % arg.c_buffer_scatter_offset(count, i, j, _buf_offset_array)
                _scatter_stmts = arg.c_buffer_scatter_vec_flatten(count, i, j, offsets, _buf_name[arg],
                                                                  _buf_offset_array, loop_size)
            else:
                raise NotImplementedError
            _buf_scatter[arg] = template_scatter % {
                'ind': '  ' * nloops,
                'offset_decl': _buf_offset_decl,
                'offset': _buf_offset,
                'buffer_scatter': _scatter_stmts,
                'itspace_loops': indent(_itspace_loops, 2),
                'itspace_loop_close': indent(_itspace_loop_close, 2),
                'ofs_itspace_loops': indent(_itspace_loops, 2) if _buf_offset else '',
                'ofs_itspace_loop_close': indent(_itspace_loop_close, 2) if _buf_offset else ''
            }
        scatter = ";\n".join(_buf_scatter.values())

        if itspace._extruded:
            _addtos_extruded = ';\n'.join([arg.c_addto(i, j, _buf_name[arg],
                                                       _tmp_name[arg],
                                                       _tmp_decl[arg],
                                                       "xtr_", is_facet=is_facet,
                                                       applied_blas=applied_blas)
                                           for arg in args if arg._is_mat])
            _addtos = ""
        else:
            _addtos_extruded = ""
            _addtos = ';\n'.join([arg.c_addto(i, j, _buf_name[arg],
                                              _tmp_name[arg],
                                              _tmp_decl[arg],
                                              applied_blas=applied_blas)
                                  for count, arg in enumerate(args) if arg._is_mat])

        if not _buf_scatter:
            _itspace_loops = ''
            _itspace_loop_close = ''

        template = """
    %(scatter)s
    %(ind)s%(addtos_extruded)s;
    %(addtos)s;
"""
        return template % {
            'ind': '  ' * nloops,
            'scatter': scatter,
            'addtos_extruded': indent(_addtos_extruded, 2 + nloops),
            'addtos': indent(_addtos, 2),
        }

    return {'kernel_name': kernel_name,
            'wrapper_name': wrapper_name,
            'ssinds_arg': _ssinds_arg,
            'index_expr': _index_expr,
            'wrapper_args': _wrapper_args,
            'user_code': user_code,
            'wrapper_decs': indent(_wrapper_decs, 1),
            'const_args': _const_args,
            'const_inits': indent(_const_inits, 1),
            'vec_inits': indent(_vec_inits, 2),
            'layer_arg': _layer_arg,
            'map_decl': indent(_map_decl, 2),
            'vec_decs': indent(_vec_decs, 2),
            'map_init': indent(_map_init, 5),
            'apply_offset': indent(_apply_offset, 3),
            'extr_loop': indent(_extr_loop, 5),
            'map_bcs_m': indent(_map_bcs_m, 5),
            'map_bcs_p': indent(_map_bcs_p, 5),
            'extr_loop_close': indent(_extr_loop_close, 2),
            'interm_globals_decl': indent(_intermediate_globals_decl, 3),
            'interm_globals_init': indent(_intermediate_globals_init, 3),
            'interm_globals_writeback': indent(_intermediate_globals_writeback, 3),
            'buffer_decl': _buf_decl,
            'buffer_gather': _buf_gather,
            'kernel_args': _kernel_args,
            'itset_loop_body': '\n'.join([itset_loop_body(i, j, shape, offsets, is_facet=(iteration_region == ON_INTERIOR_FACETS))
                                          for i, j, shape, offsets in itspace])}

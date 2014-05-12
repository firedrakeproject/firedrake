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

import base
import compilation
from base import *
from mpi import collective
from configuration import configuration
from utils import as_tuple

from coffee.ast_base import Node
from coffee.ast_plan import ASTKernel
import coffee.ast_plan
from coffee.ast_vectorizer import vect_roundup


class Kernel(base.Kernel):

    def _ast_to_c(self, ast, opts={}):
        """Transform an Abstract Syntax Tree representing the kernel into a
        string of code (C syntax) suitable to CPU execution."""
        if not isinstance(ast, Node):
            return ast
        self._ast = ast
        ast_handler = ASTKernel(ast)
        ast_handler.plan_cpu(opts)
        return ast.gencode()


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
        cdim = self.data.dataset.cdim if self._flatten else 1
        return "%(type)s *%(vec_name)s[%(arity)s];\n" % \
            {'type': self.ctype,
             'vec_name': self.c_vec_name(),
             'arity': self.map.arity * cdim * (2 if is_facet else 1)}

    def c_wrapper_dec(self, is_facet=False):
        val = ""
        if self._is_mixed_mat:
            rows, cols = self._dat.sparsity.shape
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

    def c_ind_data(self, idx, i, j=0, is_top=False, layers=1, offset=None):
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

    def c_ind_data_xtr(self, idx, i, j=0, is_top=False, layers=1):
        return "%(name)s + (xtr_%(map_name)s[%(idx)s]%(top)s%(offset)s)*%(dim)s%(off)s" % \
            {'name': self.c_arg_name(i),
             'map_name': self.c_map_name(i, 0),
             'idx': idx,
             'top': ' + start_layer' if is_top else '',
             'dim': 1 if self._flatten else str(self.data[i].cdim),
             'off': ' + %d' % j if j else '',
             'offset': ' * _'+self.c_offset_name(i, 0)+'['+idx+']' if is_top else ''}

    def c_kernel_arg_name(self, i, j):
        return "p_%s" % self.c_arg_name(i, j)

    def c_global_reduction_name(self, count=None):
        return self.c_arg_name()

    def c_local_tensor_name(self, i, j):
        return self.c_kernel_arg_name(i, j)

    def c_kernel_arg(self, count, i=0, j=0, shape=(0,), is_top=False, layers=1):
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
                    return self.c_ind_data_xtr("i_%d" % self.idx.index, i, is_top=is_top, layers=layers)
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
            return "%(name)s + i * %(dim)s" % {'name': self.c_arg_name(i),
                                               'dim': self.data[i].cdim}

    def c_vec_init(self, is_top, layers, is_facet=False):
        val = []
        vec_idx = 0
        for i, (m, d) in enumerate(zip(self.map, self.data)):
            if self._flatten:
                for k in range(d.dataset.cdim):
                    for idx in range(m.arity):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                   {'vec_name': self.c_vec_name(),
                                    'idx': vec_idx,
                                    'data': self.c_ind_data(idx, i, k, is_top=is_top, layers=layers,
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
                                        'data': self.c_ind_data(idx, i, k, is_top=is_top, layers=layers,
                                                                offset=m.offset[idx])})
                            vec_idx += 1
            else:
                for idx in range(m.arity):
                    val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                               {'vec_name': self.c_vec_name(),
                                'idx': vec_idx,
                                'data': self.c_ind_data(idx, i, is_top=is_top, layers=layers,
                                                        offset=m.offset[idx] if is_top else None)})
                    vec_idx += 1
                if is_facet:
                    for idx in range(m.arity):
                        val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                                   {'vec_name': self.c_vec_name(),
                                    'idx': vec_idx,
                                    'data': self.c_ind_data(idx, i, is_top=is_top, layers=layers,
                                                            offset=m.offset[idx])})
                        vec_idx += 1
        return ";\n".join(val)

    def c_addto_scalar_field(self, i, j, buf_name, extruded=None, is_facet=False):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rows_str = "%s + i * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + i * %s" % (self.c_map_name(1, j), ncols)

        if extruded is not None:
            rows_str = extruded + self.c_map_name(0, i)
            cols_str = extruded + self.c_map_name(1, j)

        return 'addto_vector(%(mat)s, %(vals)s, %(nrows)s, %(rows)s, %(ncols)s, %(cols)s, %(insert)d)' % \
            {'mat': self.c_arg_name(i, j),
             'vals': buf_name,
             'nrows': nrows * (2 if is_facet else 1),
             'ncols': ncols * (2 if is_facet else 1),
             'rows': rows_str,
             'cols': cols_str,
             'insert': self.access == WRITE}

    def c_addto_vector_field(self, i, j, xtr="", is_facet=False):
        maps = as_tuple(self.map, Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rmult, cmult = self.data.sparsity[i, j].dims
        s = []
        if self._flatten:
            idx = '[i_0][i_1]'
            val = "&%s%s" % ("buffer_" + self.c_arg_name(), idx)
            row = "%(m)s * %(xtr)s%(map)s[%(elem_idx)si_0 %% %(dim)s] + (i_0 / %(dim)s)" % \
                  {'m': rmult,
                   'map': self.c_map_name(0, i),
                   'dim': nrows,
                   'elem_idx': "i * %d +" % (nrows) if xtr == "" else "",
                   'xtr': xtr}
            col = "%(m)s * %(xtr)s%(map)s[%(elem_idx)si_1 %% %(dim)s] + (i_1 / %(dim)s)" % \
                  {'m': cmult,
                   'map': self.c_map_name(1, j),
                   'dim': ncols,
                   'elem_idx': "i * %d +" % (ncols) if xtr == "" else "",
                   'xtr': xtr}
            return 'addto_scalar(%s, %s, %s, %s, %d)' \
                % (self.c_arg_name(i, j), val, row, col, self.access == WRITE)
        for r in xrange(rmult):
            for c in xrange(cmult):
                idx = '[i_0 + %d][i_1 + %d]' % (r, c)
                val = "&%s%s" % ("buffer_" + self.c_arg_name(), idx)
                row = "%(m)s * %(xtr)s%(map)s[%(elem_idx)si_0] + %(r)s" % \
                      {'m': rmult,
                       'map': self.c_map_name(0, i),
                       'dim': nrows,
                       'r': r,
                       'elem_idx': "i * %d +" % (nrows) if xtr == "" else "",
                       'xtr': xtr}
                col = "%(m)s * %(xtr)s%(map)s[%(elem_idx)si_1] + %(c)s" % \
                      {'m': cmult,
                       'map': self.c_map_name(1, j),
                       'dim': ncols,
                       'c': c,
                       'elem_idx': "i * %d +" % (ncols) if xtr == "" else "",
                       'xtr': xtr}

                s.append('addto_scalar(%s, %s, %s, %s, %d)'
                         % (self.c_arg_name(i, j), val, row, col, self.access == WRITE))
        return ';\n'.join(s)

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
            for k in range(d.dataset.cdim if self._flatten else 1):
                for idx in range(m.arity):
                    val.append("%(name)s[%(j)d] += %(offset)s[%(i)d] * %(dim)s;" %
                               {'name': self.c_vec_name(),
                                'i': idx,
                                'j': vec_idx,
                                'offset': self.c_offset_name(i, 0),
                                'dim': d.dataset.cdim})
                    vec_idx += 1
                if is_facet:
                    for idx in range(m.arity):
                        val.append("%(name)s[%(j)d] += %(offset)s[%(i)d] * %(dim)s;" %
                                   {'name': self.c_vec_name(),
                                    'i': idx,
                                    'j': vec_idx,
                                    'offset': self.c_offset_name(i, 0),
                                    'dim': d.dataset.cdim})
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

    def c_map_init(self, is_top=False, layers=1, is_facet=False):
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
                                        'ind_flat': m.arity * k + idx,
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
                                            'ind_flat': m.arity * (k + d.cdim) + idx,
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

    def c_map_bcs(self, top_bottom, layers, sign):
        maps = as_tuple(self.map, Map)
        val = []
        if top_bottom is None:
            return ""

        # To throw away boundary condition values, we subtract a large
        # value from the map to make it negative then add it on later to
        # get back to the original
        max_int = 10000000
        if top_bottom[0]:
            # We need to apply the bottom bcs
            val.append("if (j_0 == 0){")
            for i, map in enumerate(maps):
                if not map.iterset._extruded:
                    continue
                for j, m in enumerate(map):
                    for idx in range(m.arity):
                        val.append("xtr_%(name)s[%(ind)s] %(sign)s= %(val)s;" %
                                   {'name': self.c_map_name(i, j),
                                    'val': max_int if m.bottom_mask[idx] < 0 else 0,
                                    'ind': idx,
                                    'sign': sign})
            val.append("}")

        if top_bottom[1]:
            # We need to apply the top bcs
            val.append("if (j_0 == end_layer - 1){")
            for i, map in enumerate(maps):
                if not map.iterset._extruded:
                    continue
                for j, m in enumerate(map):
                    for idx in range(m.arity):
                        val.append("xtr_%(name)s[%(ind)s] %(sign)s= %(val)s;" %
                                   {'name': self.c_map_name(i, j),
                                    'val': max_int if m.top_mask[idx] < 0 else 0,
                                    'ind': idx,
                                    'sign': sign})
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
                            val.append("xtr_%(name)s[%(ind_flat)s] += %(off)s[%(ind)s] * %(dim)s;" %
                                       {'name': self.c_map_name(i, j),
                                        'off': self.c_offset_name(i, j),
                                        'ind': idx,
                                        'ind_flat': m.arity * k + idx,
                                        'dim': d.cdim})
                    else:
                        val.append("xtr_%(name)s[%(ind)s] += %(off)s[%(ind)s];" %
                                   {'name': self.c_map_name(i, j),
                                    'off': self.c_offset_name(i, j),
                                    'ind': idx})
                if is_facet:
                    for idx in range(m.arity):
                        if self._is_dat and self._flatten and d.cdim > 1:
                            for k in range(d.cdim):
                                val.append("xtr_%(name)s[%(ind_flat)s] += %(off)s[%(ind)s] * %(dim)s;" %
                                           {'name': self.c_map_name(i, j),
                                            'off': self.c_offset_name(i, j),
                                            'ind': idx,
                                            'ind_flat': m.arity * (k + d.cdim) + idx,
                                            'dim': d.cdim})
                        else:
                            val.append("xtr_%(name)s[%(ind)s] += %(off)s[%(ind_zero)s];" %
                                       {'name': self.c_map_name(i, j),
                                        'off': self.c_offset_name(i, j),
                                        'ind': m.arity + idx,
                                        'ind_zero': idx})
        return '\n'.join(val)+'\n'

    def c_offset_init(self):
        maps = as_tuple(self.map, Map)
        val = []
        for i, map in enumerate(maps):
            if not map.iterset._extruded:
                continue
            for j, m in enumerate(map):
                val.append("int *%s" % self.c_offset_name(i, j))
        if len(val) == 0:
            return ""
        return ", " + ", ".join(val)

    def c_buffer_decl(self, size, idx, buf_name, is_facet=False):
        buf_type = self.data.ctype
        dim = len(size)
        compiler = coffee.ast_plan.compiler
        isa = coffee.ast_plan.intrinsics
        return (buf_name, "%(typ)s %(name)s%(dim)s%(align)s%(init)s" %
                {"typ": buf_type,
                 "name": buf_name,
                 "dim": "".join(["[%d]" % (d * (2 if is_facet else 1)) for d in size]),
                 "align": " " + compiler.get("align")(isa["alignment"]) if compiler else "",
                 "init": " = " + "{" * dim + "0.0" + "}" * dim if self.access._mode in ['WRITE', 'INC'] else ""})

    def c_buffer_gather(self, size, idx, buf_name):
        dim = 1 if self._flatten else self.data.cdim
        return ";\n".join(["%(name)s[i_0*%(dim)d%(ofs)s] = *(%(ind)s%(ofs)s);\n" %
                           {"name": buf_name,
                            "dim": dim,
                            "ind": self.c_kernel_arg(idx),
                            "ofs": " + %s" % j if j else ""} for j in range(dim)])

    def c_buffer_scatter_mm(self, i, j, mxofs, buf_name, buf_scat_name):
        return "%(name_scat)s[i_0][i_1] = %(buf_name)s[%(row)d + i_0][%(col)d + i_1];" % \
            {"name_scat": buf_scat_name,
             "buf_name": buf_name,
             "row": mxofs[0],
             "col": mxofs[1]}

    def c_buffer_scatter_vec(self, count, i, j, mxofs, buf_name):
        dim = 1 if self._flatten else self.data.split[i].cdim
        return ";\n".join(["*(%(ind)s%(nfofs)s) %(op)s %(name)s[i_0*%(dim)d%(nfofs)s%(mxofs)s]" %
                           {"ind": self.c_kernel_arg(count, i, j),
                            "op": "=" if self._access._mode == "WRITE" else "+=",
                            "name": buf_name,
                            "dim": dim,
                            "nfofs": " + %d" % o if o else "",
                            "mxofs": " + %d" % (mxofs[0] * dim) if mxofs else ""}
                           for o in range(dim)])


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []

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
        self._itspace = itspace
        self._args = args
        self._direct = kwargs.get('direct', False)
        self._iteration_region = kwargs.get('iterate', ALL)
        self._initialized = True

    @collective
    def __call__(self, *args, **kwargs):
        argtypes = kwargs.get('argtypes', None)
        restype = kwargs.get('restype', None)
        return self.compile(argtypes, restype)(*args)

    @property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @collective
    def compile(self, argtypes=None, restype=None):
        if hasattr(self, '_fun'):
            # It should not be possible to pull a jit module out of
            # the cache /with/ arguments
            if hasattr(self, '_args'):
                raise RuntimeError("JITModule is holding onto args, causing a memory leak (should never happen)")
            self._fun.argtypes = argtypes
            self._fun.restype = restype
            return self._fun
        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")
        strip = lambda code: '\n'.join([l for l in code.splitlines()
                                        if l.strip() and l.strip() != ';'])

        compiler = coffee.ast_plan.compiler
        if any(arg._is_soa for arg in self._args):
            kernel_code = """
            #define OP2_STRIDE(a, idx) a[idx]
            %(header)s
            %(code)s
            #undef OP2_STRIDE
            """ % {'code': self._kernel.code,
                   'header': compiler.get('vect_header', '')}
        else:
            kernel_code = """
            %(header)s
            %(code)s
            """ % {'code': self._kernel.code,
                   'header': compiler.get('vect_header', '')}
        code_to_compile = strip(dedent(self._wrapper) % self.generate_code())

        _const_decs = '\n'.join([const._format_declaration()
                                for const in Const._definitions()]) + '\n'

        code_to_compile = """
        #include <mat_utils.h>
        #include <stdbool.h>
        #include <math.h>
        %(sys_headers)s
        %(consts)s

        %(kernel)s

        %(wrapper)s
        """ % {'consts': _const_decs, 'kernel': kernel_code,
               'wrapper': code_to_compile,
               'sys_headers': '\n'.join(self._kernel._headers)}

        self._dump_generated_code(code_to_compile)
        if configuration["debug"]:
            self._wrapper_code = code_to_compile

        cppargs = ["-I%s/include" % d for d in get_petsc_dir()] + \
                  ["-I%s" % d for d in self._kernel._include_dirs] + \
                  ["-I%s" % os.path.abspath(os.path.dirname(__file__))]
        if compiler:
            cppargs += [compiler[coffee.ast_plan.intrinsics['inst_set']]]
        ldargs = ["-L%s/lib" % d for d in get_petsc_dir()] + \
                 ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()] + \
                 ["-lpetsc", "-lm"] + self._libraries
        self._fun = compilation.load(code_to_compile,
                                     self._wrapper_name,
                                     cppargs=cppargs,
                                     ldargs=ldargs,
                                     argtypes=argtypes,
                                     restype=restype,
                                     compiler=compiler.get('name'))
        # Blow away everything we don't need any more
        del self._args
        del self._kernel
        del self._itspace
        del self._direct
        del self._iteration_region
        return self._fun

    def generate_code(self):

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
            if self._direct:
                return "{"
            return "for (int j_0 = start_layer; j_0 < end_layer; ++j_0){"

        _ssinds_arg = ""
        _index_expr = "n"
        is_top = (self._iteration_region == ON_TOP)
        is_facet = (self._iteration_region == ON_INTERIOR_FACETS)

        if isinstance(self._itspace._iterset, Subset):
            _ssinds_arg = "int* ssinds,"
            _index_expr = "ssinds[n]"

        _wrapper_args = ', '.join([arg.c_wrapper_arg() for arg in self._args])

        # Pass in the is_facet flag to mark the case when it's an interior horizontal facet in
        # an extruded mesh.
        _wrapper_decs = ';\n'.join([arg.c_wrapper_dec(is_facet=is_facet) for arg in self._args])

        _vec_decs = ';\n'.join([arg.c_vec_dec(is_facet=is_facet) for arg in self._args if arg._is_vec_map])

        if len(Const._defs) > 0:
            _const_args = ', '
            _const_args += ', '.join([c_const_arg(c) for c in Const._definitions()])
        else:
            _const_args = ''
        _const_inits = ';\n'.join([c_const_init(c) for c in Const._definitions()])

        _intermediate_globals_decl = ';\n'.join(
            [arg.c_intermediate_globals_decl(count)
             for count, arg in enumerate(self._args)
             if arg._is_global_reduction])
        _intermediate_globals_init = ';\n'.join(
            [arg.c_intermediate_globals_init(count)
             for count, arg in enumerate(self._args)
             if arg._is_global_reduction])
        _intermediate_globals_writeback = ';\n'.join(
            [arg.c_intermediate_globals_writeback(count)
             for count, arg in enumerate(self._args)
             if arg._is_global_reduction])

        _vec_inits = ';\n'.join([arg.c_vec_init(is_top, self._itspace.layers, is_facet) for arg in self._args
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
        if self._itspace._extruded:
            a_bcs = self._itspace.iterset._extruded_bcs
            _layer_arg = ", int start_layer, int end_layer"
            _off_args = ''.join([arg.c_offset_init() for arg in self._args
                                 if arg._uses_itspace or arg._is_vec_map])
            _map_decl += ';\n'.join([arg.c_map_decl(is_facet=is_facet)
                                     for arg in self._args if arg._uses_itspace])
            _map_init += ';\n'.join([arg.c_map_init(is_top=is_top, layers=self._itspace.layers, is_facet=is_facet)
                                     for arg in self._args if arg._uses_itspace])
            _map_bcs_m += ';\n'.join([arg.c_map_bcs(a_bcs, self._itspace.layers, "-")
                                      for arg in self._args if arg._is_mat])
            _map_bcs_p += ';\n'.join([arg.c_map_bcs(a_bcs, self._itspace.layers, "+")
                                      for arg in self._args if arg._is_mat])
            _apply_offset += ';\n'.join([arg.c_add_offset_map(is_facet=is_facet)
                                         for arg in self._args if arg._uses_itspace])
            _apply_offset += ';\n'.join([arg.c_add_offset(is_facet=is_facet)
                                         for arg in self._args if arg._is_vec_map])
            _extr_loop = '\n' + extrusion_loop()
            _extr_loop_close = '}\n'
        else:
            _off_args = ""

        # Build kernel invocation. Let X be a parameter of the kernel representing a tensor
        # accessed in an iteration space. Let BUFFER be an array of the same size as X.
        # BUFFER is declared and intialized in the wrapper function.
        # * if X is written or incremented in the kernel, then BUFFER is initialized to 0
        # * if X in read in the kernel, then BUFFER gathers data expected by X
        _itspace_args = [(count, arg) for count, arg in enumerate(self._args) if arg._uses_itspace]
        _buf_gather = ""
        _layout_decl = ""
        _layout_loops = ""
        _layout_loops_close = ""
        _layout_assign = ""
        _buf_decl = {}
        _buf_name = ""
        for count, arg in _itspace_args:
            _buf_name = "buffer_" + arg.c_arg_name(count)
            _layout_name = None
            _buf_size = list(self._itspace._extents)
            if not arg._is_mat:
                # Readjust size to take into account the size of a vector space
                dim = arg.data.dim
                _dat_size = [s[0] for s in dim] if len(arg.data.dim) > 1 else dim
                _buf_size = [sum([e*d for e, d in zip(_buf_size, _dat_size)])]
                _loop_size = [_buf_size[i]/_dat_size[i] for i in range(len(_buf_size))]
            if self._kernel._opts.get('ap'):
                if arg._is_mat:
                    # Layout of matrices must be restored prior to the invokation of addto_vector
                    # if padding was used
                    _layout_name = "buffer_layout_" + arg.c_arg_name(count)
                    _layout_decl = arg.c_buffer_decl(_buf_size, count, _layout_name, is_facet=is_facet)[1]
                    _layout_loops = '\n'.join(['  ' * n + itspace_loop(n, e) for n, e in enumerate(_buf_size)])
                    _layout_assign = _layout_name + "[i_0][i_1]" + " = " + _buf_name + "[i_0][i_1]"
                    _layout_loops_close = '\n'.join('  ' * n + '}' for n in range(len(_buf_size) - 1, -1, -1))
                _buf_size = [vect_roundup(s) for s in _buf_size]
            _buf_decl[arg] = arg.c_buffer_decl(_buf_size, count, _buf_name, is_facet=is_facet)
            _buf_name = _layout_name or _buf_name
            if arg.access._mode not in ['WRITE', 'INC']:
                _itspace_loops = '\n'.join(['  ' * n + itspace_loop(n, e) for n, e in enumerate(_loop_size)])
                _buf_gather = arg.c_buffer_gather(_buf_size, count, _buf_name)
                _itspace_loop_close = '\n'.join('  ' * n + '}' for n in range(len(_loop_size) - 1, -1, -1))
                _buf_gather = "\n".join([_itspace_loops, _buf_gather, _itspace_loop_close])
        _kernel_args = ', '.join([arg.c_kernel_arg(count) if not arg._uses_itspace else _buf_decl[arg][0]
                                  for count, arg in enumerate(self._args)])
        _buf_decl = ";\n".join([decl for name, decl in _buf_decl.values()])

        def itset_loop_body(i, j, shape, offsets, is_facet=False):
            nloops = len(shape)
            _itspace_loops = '\n'.join(['  ' * n + itspace_loop(n, e) for n, e in enumerate(shape)])
            _itspace_args = [(count, arg) for count, arg in enumerate(self._args)
                             if arg.access._mode in ['WRITE', 'INC'] and arg._uses_itspace]
            _buf_scatter = ""
            _buf_decl_scatter = ""
            _buf_scatter_name = None
            for count, arg in _itspace_args:
                if arg._is_mat and arg._is_mixed:
                    _buf_scatter_name = "scatter_buffer_" + arg.c_arg_name(i, j)
                    _buf_decl_scatter = arg.data.ctype + " " + _buf_scatter_name + "".join("[%d]" % d for d in shape)
                    _buf_scatter = arg.c_buffer_scatter_mm(i, j, offsets, _buf_name, _buf_scatter_name)
                elif not arg._is_mat:
                    _buf_scatter = arg.c_buffer_scatter_vec(count, i, j, offsets, _buf_name)
                else:
                    _buf_scatter = ""
            _itspace_loop_close = '\n'.join('  ' * n + '}' for n in range(nloops - 1, -1, -1))
            _addto_buf_name = _buf_scatter_name or _buf_name
            if self._itspace._extruded:
                _addtos_scalar_field_extruded = ';\n'.join([arg.c_addto_scalar_field(i, j, _addto_buf_name, "xtr_", is_facet=is_facet) for arg in self._args
                                                            if arg._is_mat and arg.data[i, j]._is_scalar_field])
                _addtos_vector_field = ';\n'.join([arg.c_addto_vector_field(i, j, "xtr_", is_facet=is_facet) for arg in self._args
                                                  if arg._is_mat and arg.data[i, j]._is_vector_field])
                _addtos_scalar_field = ""
            else:
                _addtos_scalar_field_extruded = ""
                _addtos_scalar_field = ';\n'.join([arg.c_addto_scalar_field(i, j, _addto_buf_name) for count, arg in enumerate(self._args)
                                                   if arg._is_mat and arg.data[i, j]._is_scalar_field])
                _addtos_vector_field = ';\n'.join([arg.c_addto_vector_field(i, j) for arg in self._args
                                                  if arg._is_mat and arg.data[i, j]._is_vector_field])

            if not _addtos_vector_field and not _buf_scatter:
                _itspace_loops = ''
                _itspace_loop_close = ''

            template = """
    %(buffer_decl_scatter)s;
    %(itspace_loops)s
    %(ind)s%(buffer_scatter)s;
    %(ind)s%(addtos_vector_field)s;
    %(itspace_loop_close)s
    %(ind)s%(addtos_scalar_field_extruded)s;
    %(addtos_scalar_field)s;
"""

            return template % {
                'ind': '  ' * nloops,
                'itspace_loops': indent(_itspace_loops, 2),
                'buffer_decl_scatter': _buf_decl_scatter,
                'buffer_scatter': _buf_scatter,
                'addtos_vector_field': indent(_addtos_vector_field, 2 + nloops),
                'itspace_loop_close': indent(_itspace_loop_close, 2),
                'addtos_scalar_field_extruded': indent(_addtos_scalar_field_extruded, 2 + nloops),
                'apply_offset': indent(_apply_offset, 3),
                'extr_loop_close': indent(_extr_loop_close, 2),
                'addtos_scalar_field': indent(_addtos_scalar_field, 2)
            }

        return {'kernel_name': self._kernel.name,
                'wrapper_name': self._wrapper_name,
                'ssinds_arg': _ssinds_arg,
                'index_expr': _index_expr,
                'wrapper_args': _wrapper_args,
                'user_code': self._kernel._user_code,
                'wrapper_decs': indent(_wrapper_decs, 1),
                'const_args': _const_args,
                'const_inits': indent(_const_inits, 1),
                'vec_inits': indent(_vec_inits, 2),
                'off_args': _off_args,
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
                'layout_decl': _layout_decl,
                'layout_loop': _layout_loops,
                'layout_assign': _layout_assign,
                'layout_loop_close': _layout_loops_close,
                'kernel_args': _kernel_args,
                'itset_loop_body': '\n'.join([itset_loop_body(i, j, shape, offsets, is_facet=(self._iteration_region == ON_INTERIOR_FACETS))
                                              for i, j, shape, offsets in self._itspace])}

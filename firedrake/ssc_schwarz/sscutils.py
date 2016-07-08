from __future__ import absolute_import
from firedrake.petsc import PETSc
from pyop2 import sequential as seq
from pyop2 import base as pyop2
from pyop2 import op2
import numpy


# Fake up some PyOP2 objects so we can abuse the PyOP2 code
# compilation pipeline to get a callable function pointer for
# assembling into a dense matrix.
# FIXME: Not correct for MixedElement yet.
class DenseSparsity(object):
    def __init__(self, rset, cset):
        if isinstance(rset, op2.MixedDataSet) or \
           isinstance(cset, op2.MixedDataSet):
            raise NotImplementedError
        self.shape = (1, 1)
        self._nrows = rset.size
        self._ncols = cset.size
        self._dims = (((rset.cdim, cset.cdim), ), )
        self.dims = self._dims
        self.dsets = rset, cset

    def __getitem__(self, *args):
        return self


class MatArg(seq.Arg):
    def __init__(self, data, map, idx, access, flatten):
        self.data = data
        self._map = map
        self._idx = idx
        self._access = access
        self._flatten = flatten
        rdims = tuple(d.cdim for d in data.sparsity.dsets[0])
        cdims = tuple(d.cdim for d in data.sparsity.dsets[1])
        self._block_shape = tuple(tuple((mr.arity * dr, mc.arity * dc)
                                        for mc, dc in zip(map[1], cdims))
                                  for mr, dr in zip(map[0], rdims))
        self.cache_key = None

    def c_addto(self, i, j, buf_name, tmp_name, tmp_decl,
                extruded=None, is_facet=False, applied_blas=False):
        # Override global c_addto to index the map locally rather than globally.
        from pyop2.utils import as_tuple
        maps = as_tuple(self.map, op2.Map)
        nrows = maps[0].split[i].arity
        ncols = maps[1].split[j].arity
        rows_str = "%s + n * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + n * %s" % (self.c_map_name(1, j), ncols)

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
        addto = 'MatSetValues'
        if self.data._is_vector_field:
            addto = 'MatSetValuesBlocked'
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
                raise NotImplementedError
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
                    'insert': "INSERT_VALUES" if self.access == op2.WRITE else "ADD_VALUES"})
        return "\n".join(ret)


class DenseMat(pyop2.Mat):
    def __init__(self, rset, cset):
        self._sparsity = DenseSparsity(rset, cset)
        self.dtype = numpy.dtype(PETSc.ScalarType)

    def __call__(self, access, path, flatten=False):
        path_maps = [arg.map for arg in path]
        path_idxs = [arg.idx for arg in path]
        return MatArg(self, path_maps, path_idxs, access, flatten)


class JITModule(seq.JITModule):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        # Don't want to cache these anywhere I think.
        return None


def matrix_callable(kernels, V, coordinates, *coefficients):
    kinfo = kernels[0]
    args = []
    matarg = DenseMat(V.dof_dset, V.dof_dset)(op2.INC,
                                              (V.cell_node_map()[op2.i[0]],
                                               V.cell_node_map()[op2.i[1]]),
                                              flatten=True)
    matarg.position = len(args)
    args.append(matarg)
    carg = coordinates.dat(op2.READ, coordinates.cell_node_map(), flatten=True)
    carg.position = len(args)
    args.append(carg)
    for n in kinfo.coefficient_map:
        c = coefficients[n]
        arg = c.dat(op2.READ, c.cell_node_map(), flatten=True)
        arg.position = len(args)
        args.append(arg)
    itspace = pyop2.build_itspace(args, op2.Subset(coordinates.cell_set, [0]))
    mod = JITModule(kinfo.kernel, itspace, *args)
    return mod._fun

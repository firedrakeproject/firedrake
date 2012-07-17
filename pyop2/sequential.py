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

from exceptions import *
from utils import *
import op_lib_core as core

# Data API

class Access(object):
    """OP2 access type."""

    _modes = ["READ", "WRITE", "RW", "INC", "MIN", "MAX"]

    @validate_in(('mode', _modes, ModeValueError))
    def __init__(self, mode):
        self._mode = mode

    def __str__(self):
        return "OP2 Access: %s" % self._mode

    def __repr__(self):
        return "Access('%s')" % self._mode

READ  = Access("READ")
WRITE = Access("WRITE")
RW    = Access("RW")
INC   = Access("INC")
MIN   = Access("MIN")
MAX   = Access("MAX")

# Data API

class Arg(object):
    def __init__(self, data=None, map=None, idx=None, access=None):
        self._dat = data
        self._map = map
        self._idx = idx
        self._access = access
        self._lib_handle = None

    @property
    def c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_arg(self, dat=isinstance(self._dat, Dat),
                                         gbl=isinstance(self._dat, Global))
        return self._lib_handle

    @property
    def data(self):
        """Data carrier: Dat, Mat, Const or Global."""
        return self._dat

    @property
    def map(self):
        """Mapping."""
        return self._map

    @property
    def idx(self):
        """Index into the mapping."""
        return self._idx

    @property
    def access(self):
        """Access descriptor."""
        return self._access

    @property
    def _is_soa(self):
        return isinstance(self._dat, Dat) and self._dat.soa

    @property
    def _is_vec_map(self):
        return self._is_indirect and self._idx is None

    @property
    def _is_global(self):
        return isinstance(self._dat, Global)

    @property
    def _is_global_reduction(self):
        return self._is_global and self._access in [INC, MIN, MAX]

    @property
    def _is_dat(self):
        return isinstance(self._dat, Dat)

    @property
    def _is_INC(self):
        return self._access == INC

    @property
    def _is_MIN(self):
        return self._access == MIN

    @property
    def _is_MAX(self):
        return self._access == MAX

    @property
    def _is_direct(self):
        return isinstance(self._dat, Dat) and self._map is IdentityMap

    @property
    def _is_indirect(self):
        return isinstance(self._dat, Dat) and self._map not in [None, IdentityMap]

    @property
    def _is_indirect_and_not_read(self):
        return self._is_indirect and self._access is not READ


    @property
    def _is_indirect_reduction(self):
        return self._is_indirect and self._access is INC

    @property
    def _is_global(self):
        return isinstance(self._dat, Global)

    @property
    def _is_mat(self):
        return isinstance(self._dat, Mat)

class Set(object):
    """OP2 set."""

    _globalcount = 0

    @validate_type(('size', int, SizeTypeError), ('name', str, NameTypeError))
    def __init__(self, size, name=None):
        self._size = size
        self._name = name or "set_%d" % Set._globalcount
        self._lib_handle = None
        Set._globalcount += 1

    @classmethod
    def fromhdf5(cls, f, name):
        slot = f[name]
        size = slot.value.astype(np.int)
        shape = slot.shape
        if shape != (1,):
            raise SizeTypeError("Shape of %s is incorrect" % name)
        return cls(size[0], name)

    def __call__(self, *dims):
        if len(dims) is 0:
            return self
        return IterationSpace(self, dims)

    @property
    def c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_set(self)
        return self._lib_handle

    @property
    def size(self):
        """Set size"""
        return self._size

    @property
    def name(self):
        """User-defined label"""
        return self._name

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%s, '%s')" % (self._size, self._name)

class DataCarrier(object):
    """Abstract base class for OP2 data."""

    @property
    def dtype(self):
        """Data type."""
        return self._data.dtype

    @property
    def name(self):
        """User-defined label."""
        return self._name

    @property
    def dim(self):
        """Dimension/shape of a single data item."""
        return self._dim

class Dat(DataCarrier):
    """OP2 vector data. A Dat holds a value for every member of a set."""

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]
    _arg_type = Arg

    @validate_type(('dataset', Set, SetTypeError), ('name', str, NameTypeError))
    def __init__(self, dataset, dim, data=None, dtype=None, name=None, soa=None):
        self._dataset = dataset
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, (dataset.size,)+self._dim, allow_none=True)
        # Are these data in SoA format, rather than standard AoS?
        self._soa = bool(soa)
        # Make data "look" right
        if self._soa:
            self._data = self._data.T
        self._name = name or "dat_%d" % Dat._globalcount
        self._lib_handle = None
        Dat._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, path, access):
        if isinstance(path, Map):
            return self._arg_type(data=self, map=path, access=access)
        else:
            path._dat = self
            path._access = access
            return path

    @classmethod
    def fromhdf5(cls, dataset, f, name):
        slot = f[name]
        data = slot.value
        dim = slot.shape[1:]
        soa = slot.attrs['type'].find(':soa') > 0
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        # We don't pass soa to the constructor, because that
        # transposes the data, but we've got them from the hdf5 file
        # which has them in the right shape already.
        ret = cls(dataset, dim[0], data, name=name)
        ret._soa = soa
        return ret

    @property
    def c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_dat(self)
        return self._lib_handle

    @property
    def dataset(self):
        """Set on which the Dat is defined."""
        return self._dataset

    @property
    def soa(self):
        """Are the data in SoA format?"""
        return self._soa

    @property
    def data(self):
        """Data array."""
        if len(self._data) is 0:
            raise RuntimeError("Illegal access: No data associated with this Dat!")
        return self._data

    def __str__(self):
        return "OP2 Dat: %s on (%s) with dim %s and datatype %s" \
               % (self._name, self._dataset, self._dim, self._data.dtype.name)

    def __repr__(self):
        return "Dat(%r, %s, '%s', None, '%s')" \
               % (self._dataset, self._dim, self._data.dtype, self._name)

class Sparsity(object):
    """OP2 Sparsity, a matrix structure derived from the cross product of
    two sets of maps"""

    _globalcount = 0

    def __init__(self, rmap, cmap, dims, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        # FIXME: Should take a tupe of rmaps and cmaps
        self._rmap = rmap
        self._cmap = cmap
        self._dims = as_tuple(dims, int)
        self._name = name or "global_%d" % Sparsity._globalcount
        self._lib_handle = core.op_sparsity(self)
        Sparsity._globalcount += 1

class Mat(DataCarrier):
    """OP2 matrix data. A Mat is defined on a sparsity pattern and holds a value
    for each element in the sparsity."""

    _globalcount = 0
    _modes = [WRITE, INC]
    _arg_type = Arg

    @validate_type(('sparsity', Sparsity, SparsityTypeError), \
                   ('name', str, NameTypeError))
    def __init__(self, sparsity, dim, dtype=None, name=None):
        self._sparsity = sparsity
        # FIXME: Eventually we want to take a tuple of dims
        self._dim = dim
        self._datatype = np.dtype(dtype)
        self._name = name or "mat_%d" % Mat._globalcount
        self._lib_handle = core.op_mat(self)
        Mat._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, maps, access):
        maps = as_tuple(maps, Map, 2)
        for map, dataset in zip(maps, (self._sparsity._rmap, self._sparsity._cmap)):
            if map._dataset != dataset:
                raise SetValueError("Invalid data set for map %s (is %s, should be %s)" \
                        % (map._name, map._dataset._name, dataset._name))
        return self._arg_type(data=self, map=maps, access=access)

    @property
    def sparsity(self):
        """Sparsity on which the Mat is defined."""
        return self._sparsity

    @property
    def dtype(self):
        """Data type."""
        return self._datatype

    def __str__(self):
        return "OP2 Mat: %s, sparsity (%s), dimension %s, datatype %s" \
               % (self._name, self._sparsity, self._dim, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, %s, '%s', '%s')" \
               % (self._sparsity, self._dim, self._datatype, self._name)

    @property
    def dtype(self):
        """Datatype of this matrix"""
        return self._datatype

class Const(DataCarrier):
    """Data that is constant for any element of any set."""

    class NonUniqueNameError(ValueError):
        """Name already in use."""

    _globalcount = 0
    _modes = [READ]

    _defs = set()

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data, name, dtype=None):
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, self._dim)
        self._name = name or "const_%d" % Const._globalcount
        if any(self._name is const._name for const in Const._defs):
            raise Const.NonUniqueNameError(
                "OP2 Constants are globally scoped, %s is already in use" % self._name)
        self._access = READ
        Const._globalcount += 1
        Const._defs.add(self)

    @classmethod
    def fromhdf5(cls, f, name):
        slot = f[name]
        dim = slot.shape
        data = slot.value
        if len(dim) < 1:
            raise DimTypeError("Invalid dimension value %s" % dim)
        return cls(dim, data, name)

    @property
    def data(self):
        """Data array."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

    def __str__(self):
        return "OP2 Const: %s of dim %s and type %s with value %s" \
               % (self._name, self._dim, self._data.dtype.name, self._data)

    def __repr__(self):
        return "Const(%s, %s, '%s')" \
               % (self._dim, self._data, self._name)

    def remove_from_namespace(self):
        if self in Const._defs:
            Const._defs.remove(self)

    def format_for_c(self, typemap):
        dec = 'static const ' + typemap[self._data.dtype.name] + ' ' + self._name
        if self._dim[0] > 1:
            dec += '[' + str(self._dim[0]) + ']'
        dec += ' = '
        if self._dim[0] > 1:
            dec += '{'
        dec += ', '.join(str(datum) for datum in self._data)
        if self._dim[0] > 1:
            dec += '}'

        dec += ';'
        return dec

class Global(DataCarrier):
    """OP2 global value."""

    _globalcount = 0
    _modes = [READ, INC, MIN, MAX]
    _arg_type = Arg

    @validate_type(('name', str, NameTypeError))
    def __init__(self, dim, data, dtype=None, name=None):
        self._dim = as_tuple(dim, int)
        self._data = verify_reshape(data, dtype, self._dim)
        self._name = name or "global_%d" % Global._globalcount
        Global._globalcount += 1

    @validate_in(('access', _modes, ModeValueError))
    def __call__(self, access):
        return self._arg_type(data=self, access=access)

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
                % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global('%s', %r, %r)" % (self._name, self._dim, self._data)

    @property
    def data(self):
        """Data array."""
        return self._data

    @data.setter
    def data(self, value):
        self._data = verify_reshape(value, self.dtype, self.dim)

#FIXME: Part of kernel API, but must be declared before Map for the validation.

class IterationIndex(object):
    """OP2 iteration space index"""

    def __init__(self, index):
        assert isinstance(index, int), "i must be an int"
        self._index = index

    def __str__(self):
        return "OP2 IterationIndex: %d" % self._index

    def __repr__(self):
        return "IterationIndex(%d)" % self._index

    @property
    def index(self):
        return self._index

def i(index):
    """Shorthand for constructing IterationIndex objects"""
    return IterationIndex(index)

class Map(object):
    """OP2 map, a relation between two Sets."""

    _globalcount = 0
    _arg_type = Arg

    @validate_type(('iterset', Set, SetTypeError), ('dataset', Set, SetTypeError), \
            ('dim', int, DimTypeError), ('name', str, NameTypeError))
    def __init__(self, iterset, dataset, dim, values, name=None):
        self._iterset = iterset
        self._dataset = dataset
        self._dim = dim
        self._values = verify_reshape(values, np.int32, (iterset.size, dim))
        self._name = name or "map_%d" % Map._globalcount
        self._lib_handle = None
        Map._globalcount += 1

    @validate_type(('index', (int, IterationIndex), IndexTypeError))
    def __call__(self, index):
        if isinstance(index, int) and not (0 <= index < self._dim):
            raise IndexValueError("Index must be in interval [0,%d]" % (self._dim-1))
        if isinstance(index, IterationIndex) and index.index not in [0, 1]:
            raise IndexValueError("IterationIndex must be in interval [0,1]")
        return self._arg_type(map=self, idx=index)

    @property
    def c_handle(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_map(self)
        return self._lib_handle

    @classmethod
    def fromhdf5(cls, iterset, dataset, f, name):
        slot = f[name]
        values = slot.value
        dim = slot.shape[1:]
        if len(dim) != 1:
            raise DimTypeError("Unrecognised dimension value %s" % dim)
        return cls(iterset, dataset, dim[0], values, name)

    @property
    def iterset(self):
        """Set mapped from."""
        return self._iterset

    @property
    def dataset(self):
        """Set mapped to."""
        return self._dataset

    @property
    def dim(self):
        """Dimension of the mapping: number of dataset elements mapped to per
        iterset element."""
        return self._dim

    @property
    def dtype(self):
        """Data type."""
        return self._values.dtype

    @property
    def values(self):
        """Mapping array."""
        return self._values

    @property
    def name(self):
        """User-defined label"""
        return self._name

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with dim %s" \
               % (self._name, self._iterset, self._dataset, self._dim)

    def __repr__(self):
        return "Map(%r, %r, %s, None, '%s')" \
               % (self._iterset, self._dataset, self._dim, self._name)

IdentityMap = Map(Set(0), Set(0), 1, [], 'identity')

# Kernel API

class IterationSpace(object):
    """OP2 iteration space type."""

    @validate_type(('iterset', Set, SetTypeError))
    def __init__(self, iterset, extents):
        self._iterset = iterset
        self._extents = as_tuple(extents, int)

    @property
    def iterset(self):
        """Set this IterationSpace is defined on."""
        return self._iterset

    @property
    def extents(self):
        """Extents of the IterationSpace."""
        return self._extents

    @property
    def size(self):
        return self._iterset.size

    def __str__(self):
        return "OP2 Iteration Space: %s with extents %s" % self._extents

    def __repr__(self):
        return "IterationSpace(%r, %r)" % (self._iterset, self._extents)

class Kernel(object):
    """OP2 kernel type."""

    _globalcount = 0

    @validate_type(('name', str, NameTypeError))
    def __init__(self, code, name):
        self._name = name or "kernel_%d" % Kernel._globalcount
        self._code = code
        Kernel._globalcount += 1

    @property
    def name(self):
        """Kernel name, must match the kernel function name in the code."""
        return self._name

    def compile(self):
        pass

    def handle(self):
        pass

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", "%s")' % (self._code, self._name)

# Parallel loop API

def par_loop(kernel, it_space, *args):
    """Invocation of an OP2 kernel with an access descriptor"""

    from instant import inline_with_numpy

    # FIXME: Complex and float16 not supported
    typemap = { "bool":    "unsigned char",
                "int":     "int",
                "int8":    "char",
                "int16":   "short",
                "int32":   "int",
                "int64":   "long long",
                "uint8":   "unsigned char",
                "uint16":  "unsigned short",
                "uint32":  "unsigned int",
                "uint64":  "unsigned long long",
                "float":   "double",
                "float32": "float",
                "float64": "double" }

    def c_arg_name(arg):
        name = arg._dat._name
        if arg._is_indirect and not arg._is_vec_map:
            name += str(arg.idx)
        return name

    def c_vec_name(arg):
        return c_arg_name(arg) + "_vec"

    def c_map_name(arg):
        return c_arg_name(arg) + "_map"

    def c_type(arg):
        return typemap[arg._dat.dtype.name]

    def c_wrapper_arg(arg):
        val = "PyObject *_%(name)s" % {'name' : c_arg_name(arg) }
        if arg._is_indirect:
            val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)}
        return val

    def c_wrapper_dec(arg):
        val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
              {'name' : c_arg_name(arg), 'type' : c_type(arg)}
        if arg._is_indirect:
            val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                   {'name' : c_map_name(arg)}
            if not arg._is_mat and arg._is_vec_map:
                val += ";\n%(type)s *%(vec_name)s[%(dim)s]" % \
                       {'type' : c_type(arg),
                        'vec_name' : c_vec_name(arg),
                        'dim' : arg.map._dim}
        return val

    def c_ind_data(arg, idx):
        return "%(name)s + %(map_name)s[i * %(map_dim)s + %(idx)s] * %(dim)s" % \
                {'name' : c_arg_name(arg),
                 'map_name' : c_map_name(arg),
                 'map_dim' : arg.map._dim,
                 'idx' : idx,
                 'dim' : arg.data._dim[0]}

    def c_kernel_arg(arg):
        if arg._is_mat:
            return c_arg_name(arg)
        elif arg._is_indirect:
            if arg._is_vec_map:
                return c_vec_name(arg)
            return c_ind_data(arg, arg.idx)
        elif isinstance(arg.data, Global):
            return c_arg_name(arg)
        else:
            return "%(name)s + i * %(dim)s" % \
                {'name' : c_arg_name(arg),
                 'dim' : arg.data._dim[0]}

    def c_vec_init(arg):
        val = []
        for i in range(arg.map._dim):
            val.append("%(vec_name)s[%(idx)s] = %(data)s" %
                       {'vec_name' : c_vec_name(arg),
                        'idx' : i,
                        'data' : c_ind_data(arg, i)} )
        return ";\n".join(val)

    _wrapper_args = ', '.join([c_wrapper_arg(arg) for arg in args])

    _wrapper_decs = ';\n'.join([c_wrapper_dec(arg) for arg in args])

    _const_decs = '\n'.join([const.format_for_c(typemap) for const in sorted(Const._defs)]) + '\n'

    _kernel_user_args = [c_kernel_arg(arg) for arg in args]
    _kernel_it_args   = ["i_%d" % d for d in range(len(it_space.dims))]
    _kernel_args = ', '.join(_kernel_user_args + _kernel_it_args)

    _vec_inits = ';\n'.join([c_vec_init(arg) for arg in args \
                             if not arg._is_mat and arg._is_vec_map])

    # FIXME: i_0 and i_1 should be loops created depending on the existence of
    # iteration space arguments
    wrapper = """
    void wrap_%(kernel_name)s__(%(wrapper_args)s) {
        %(wrapper_decs)s;
        int i_0 = 0, i_1 = 0;
        for ( int i = 0; i < %(size)s; i++ ) {
            %(vec_inits)s;
            %(kernel_name)s(%(kernel_args)s);
        }
    }"""

    if any(arg._is_soa for arg in args):
        kernel_code = """
        #define OP2_STRIDE(a, idx) a[idx]
        %(code)s
        #undef OP2_STRIDE
        """ % {'code' : kernel._code}
    else:
        kernel_code = """
        %(code)s
        """ % {'code' : kernel._code }

    code_to_compile =  wrapper % { 'kernel_name' : kernel._name,
                      'wrapper_args' : _wrapper_args,
                      'wrapper_decs' : _wrapper_decs,
                      'size' : it_space.size,
                      'vec_inits' : _vec_inits,
                      'kernel_args' : _kernel_args }

    _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel_code,
                             additional_definitions = _const_decs + kernel_code)

    _args = []
    for arg in args:
        _args.append(arg.data.data)
        if arg._is_indirect:
            _args.append(arg.map.values)

    _fun(*_args)

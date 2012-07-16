# This file is part of PyOP2.
#
# PyOP2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyOP2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyOP2.  If not, see <http://www.gnu.org/licenses>
#
# Copyright (c) 2011, Graham Markall <grm08@doc.ic.ac.uk> and others. Please see
# the AUTHORS file in the main source directory for a full list of copyright
# holders.

"""OP2 sequential backend."""

import numpy as np
import op_lib_core as core

def as_tuple(item, type=None, length=None):
    # Empty list if we get passed None
    if item is None:
        t = []
    else:
        # Convert iterable to list...
        try:
            t = tuple(item)
        # ... or create a list of a single item
        except TypeError:
            t = (item,)*(length or 1)
    if length:
        assert len(t) == length, "Tuple needs to be of length %d" % length
    if type:
        assert all(isinstance(i, type) for i in t), \
                "Items need to be of %s" % type
    return t

# Kernel API

class Access(object):
    """OP2 access type."""

    _modes = ["READ", "WRITE", "RW", "INC", "MIN", "MAX"]

    def __init__(self, mode):
        assert mode in self._modes, "Mode needs to be one of %s" % self._modes
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

class IterationSpace(object):
    """OP2 iteration space type."""

    def __init__(self, iterset, dims):
        assert isinstance(iterset, Set), "Iteration set needs to be of type Set"
        self._iterset = iterset
        self._dims = as_tuple(dims, int)

    def __str__(self):
        return "OP2 Iteration Space: %s and extra dimensions %s" % self._dims

    def __repr__(self):
        return "IterationSpace(%r, %r)" % (self._iterset, self._dims)

class Kernel(object):
    """OP2 kernel type."""

    _globalcount = 0

    def __init__(self, code, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._name = name or "kernel_%d" % Kernel._globalcount
        self._code = code
        Kernel._globalcount += 1

    def compile(self):
        pass

    def handle(self):
        pass

    def __str__(self):
        return "OP2 Kernel: %s" % self._name

    def __repr__(self):
        return 'Kernel("""%s""", "%s")' % (self._code, self._name)

# Data API

class Arg(object):
    def __init__(self, data=None, map=None, idx=None, access=None):
        self._dat = data
        self._map = map
        self._idx = idx
        self._access = access
        self._lib_handle = None

    def build_core_arg(self):
        if self._lib_handle is None:
            self._lib_handle = core.op_arg(self, dat=isinstance(self._dat, Dat),
                                           gbl=isinstance(self._dat, Global))

    @property
    def data(self):
        return self._dat
    @property
    def map(self):
        return self._map
    @property
    def idx(self):
        return self._idx
    @property
    def access(self):
        return self._access

    def is_indirect(self):
        return self._map is not None and self._map is not IdentityMap and not isinstance(self._dat, Global)

    def is_indirect_and_not_read(self):
        return self.is_indirect() and self._access is not READ

class Set(object):
    """OP2 set."""

    _globalcount = 0

    def __init__(self, size, name=None):
        assert isinstance(size, int), "Size must be of type int"
        assert not name or isinstance(name, str), "Name must be of type str"
        self._size = size
        self._name = name or "set_%d" % Set._globalcount
        self._lib_handle = core.op_set(self)
        Set._globalcount += 1

    @property
    def size(self):
        """Set size"""
        return self._size

    def __str__(self):
        return "OP2 Set: %s with size %s" % (self._name, self._size)

    def __repr__(self):
        return "Set(%s, '%s')" % (self._size, self._name)

class DataCarrier(object):
    """Abstract base class for OP2 data."""

    @property
    def dtype(self):
        """Datatype of this data carrying object"""
        return self._data.dtype

    def _verify_reshape(self, data, dtype, shape):
        """Verify data is of type dtype and try to reshaped to shape."""

        t = np.dtype(dtype) if dtype is not None else None
        try:
            return np.asarray(data, dtype=t).reshape(shape)
        except ValueError:
            raise ValueError("Invalid data: expected %d values, got %d" % \
                    (np.prod(shape), np.asarray(data).size))

class Dat(DataCarrier):
    """OP2 vector data. A Dat holds a value for every member of a set."""

    _globalcount = 0
    _modes = [READ, WRITE, RW, INC]
    _arg_type = Arg

    def __init__(self, dataset, dim, data=None, dtype=None, name=None):
        assert isinstance(dataset, Set), "Data set must be of type Set"
        assert not name or isinstance(name, str), "Name must be of type str"

        self._dataset = dataset
        self._dim = as_tuple(dim, int)
        self._data = self._verify_reshape(data, dtype, (dataset.size,)+self._dim)
        self._name = name or "dat_%d" % Dat._globalcount
        self._lib_handle = core.op_dat(self)
        Dat._globalcount += 1

    def __call__(self, path, access):
        assert access in self._modes, \
                "Acess descriptor must be one of %s" % self._modes
        if isinstance(path, Map):
            return self._arg_type(data=self, map=path, access=access)
        else:
            path._dat = self
            path._access = access
            return path

    def __str__(self):
        return "OP2 Dat: %s on (%s) with dim %s and datatype %s" \
               % (self._name, self._dataset, self._dim, self._data.dtype.name)

    def __repr__(self):
        return "Dat(%r, %s, '%s', None, '%s')" \
               % (self._dataset, self._dim, self._data.dtype, self._name)

    @property
    def data(self):
        return self._data

class Mat(DataCarrier):
    """OP2 matrix data. A Mat is defined on the cartesian product of two Sets
    and holds a value for each element in the product."""

    _globalcount = 0
    _modes = [WRITE, INC]
    _arg_type = Arg

    def __init__(self, datasets, dim, dtype=None, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._datasets = as_tuple(datasets, Set, 2)
        self._dim = as_tuple(dim, int)
        self._datatype = np.dtype(dtype)
        self._name = name or "mat_%d" % Mat._globalcount
        Mat._globalcount += 1

    def __call__(self, maps, access):
        assert access in self._modes, \
                "Acess descriptor must be one of %s" % self._modes
        for map, dataset in zip(maps, self._datasets):
            assert map._dataset == dataset, \
                    "Invalid data set for map %s (is %s, should be %s)" \
                    % (map._name, map._dataset._name, dataset._name)
        return self._arg_type(data=self, map=maps, access=access)

    def __str__(self):
        return "OP2 Mat: %s, row set (%s), col set (%s), dimension %s, datatype %s" \
               % (self._name, self._datasets[0], self._datasets[1], self._dim, self._datatype.name)

    def __repr__(self):
        return "Mat(%r, %s, '%s', '%s')" \
               % (self._datasets, self._dim, self._datatype, self._name)

class Const(DataCarrier):
    """Data that is constant for any element of any set."""

    class NonUniqueNameError(RuntimeError):
        pass

    _globalcount = 0
    _modes = [READ]

    _defs = set()

    def __init__(self, dim, data=None, dtype=None, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._dim = as_tuple(dim, int)
        self._data = self._verify_reshape(data, dtype, self._dim)
        self._name = name or "const_%d" % Const._globalcount
        if any(self._name is const._name for const in Const._defs):
            raise Const.NonUniqueNameError(
                "OP2 Constants are globally scoped, %s is already in use" % self._name)
        self._access = READ
        Const._globalcount += 1
        Const._defs.add(self)

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

    def __init__(self, dim, data=None, dtype=None, name=None):
        assert not name or isinstance(name, str), "Name must be of type str"
        self._dim = as_tuple(dim, int)
        self._data = self._verify_reshape(data, dtype, self._dim)
        self._name = name or "global_%d" % Global._globalcount
        Global._globalcount += 1

    def __call__(self, access):
        assert access in self._modes, \
                "Acess descriptor must be one of %s" % self._modes
        return self._arg_type(data=self, access=access)

    def __str__(self):
        return "OP2 Global Argument: %s with dim %s and value %s" \
                % (self._name, self._dim, self._data)

    def __repr__(self):
        return "Global('%s', %r, %r)" % (self._name, self._dim, self._data)

    @property
    def data(self):
        return self._data

class Map(object):
    """OP2 map, a relation between two Sets."""

    _globalcount = 0
    _arg_type = Arg

    def __init__(self, iterset, dataset, dim, values, name=None):
        assert isinstance(iterset, Set), "Iteration set must be of type Set"
        assert isinstance(dataset, Set), "Data set must be of type Set"
        assert isinstance(dim, int), "dim must be a scalar integer"
        assert not name or isinstance(name, str), "Name must be of type str"
        self._iterset = iterset
        self._dataset = dataset
        self._dim = dim
        try:
            self._values = np.asarray(values, dtype=np.int32).reshape(iterset.size, dim)
        except ValueError:
            raise ValueError("Invalid data: expected %d values, got %d" % \
                    (iterset.size*dim, np.asarray(values).size))
        self._name = name or "map_%d" % Map._globalcount
        self._lib_handle = core.op_map(self)
        Map._globalcount += 1

    def __call__(self, index):
        assert isinstance(index, int), "Only integer indices are allowed"
        assert 0 <= index < self._dim, \
                "Index must be in interval [0,%d]" % (self._dim-1)
        return self._arg_type(map=self, idx=index)

    def __str__(self):
        return "OP2 Map: %s from (%s) to (%s) with dim %s" \
               % (self._name, self._iterset, self._dataset, self._dim)

    def __repr__(self):
        return "Map(%r, %r, %s, None, '%s')" \
               % (self._iterset, self._dataset, self._dim, self._name)

    @property
    def values(self):
        return self._values

IdentityMap = Map(Set(0, None), Set(0, None), 1, [], 'identity')

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
        if arg.is_indirect() and arg.idx is not None:
            name += str(arg.idx)
        return name

    def c_vec_name(arg):
        return c_arg_name(arg) + "_vec"

    def c_map_name(arg):
        return c_arg_name(arg) + "_map"

    def c_type(arg):
        return typemap[arg._dat._data.dtype.name]

    def c_wrapper_arg(arg):
        val = "PyObject *_%(name)s" % {'name' : c_arg_name(arg) }
        if arg.is_indirect():
            val += ", PyObject *_%(name)s" % {'name' : c_map_name(arg)}
        return val

    def c_wrapper_dec(arg):
        val = "%(type)s *%(name)s = (%(type)s *)(((PyArrayObject *)_%(name)s)->data)" % \
              {'name' : c_arg_name(arg), 'type' : c_type(arg)}
        if arg.is_indirect():
            val += ";\nint *%(name)s = (int *)(((PyArrayObject *)_%(name)s)->data)" % \
                   {'name' : c_map_name(arg)}
            if arg.idx is None:
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
        if arg.is_indirect():
            if arg.idx is None:
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

    _kernel_args = ', '.join([c_kernel_arg(arg) for arg in args])

    _vec_inits = ';\n'.join([c_vec_init(arg) for arg in args if arg.is_indirect() and arg.idx is None])

    wrapper = """
    void wrap_%(kernel_name)s__(%(wrapper_args)s) {
        %(wrapper_decs)s;
        for ( int i = 0; i < %(size)s; i++ ) {
            %(vec_inits)s;
            %(kernel_name)s(%(kernel_args)s);
        }
    }"""

    code_to_compile =  wrapper % { 'kernel_name' : kernel._name,
                      'wrapper_args' : _wrapper_args,
                      'wrapper_decs' : _wrapper_decs,
                      'size' : it_space.size,
                      'vec_inits' : _vec_inits,
                      'kernel_args' : _kernel_args }

    _fun = inline_with_numpy(code_to_compile, additional_declarations = kernel._code,
                             additional_definitions = _const_decs + kernel._code)

    _args = []
    for arg in args:
        _args.append(arg.data.data)
        if arg.is_indirect():
            _args.append(arg.map.values)

    _fun(*_args)

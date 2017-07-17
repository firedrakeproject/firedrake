from __future__ import absolute_import, print_function, division
from six.moves import range, zip
import numpy as np
import sys
import ufl
import ctypes
from ctypes import POINTER, c_int, c_double, c_void_p

from pyop2 import op2
from pyop2.datatypes import ScalarType, IntType, as_ctypes

from firedrake import functionspaceimpl
from firedrake.logging import warning
from firedrake import utils
from firedrake import vector
try:
    import cachetools
except ImportError:
    warning("cachetools not available, expression assembly will be slowed down")
    cachetools = None


__all__ = ['Function', 'PointNotInDomainError']


class _CFunction(ctypes.Structure):
    """C struct collecting data from a :class:`Function`"""
    _fields_ = [("n_cols", c_int),
                ("n_layers", c_int),
                ("coords", POINTER(c_double)),
                ("coords_map", POINTER(as_ctypes(IntType))),
                # FIXME: what if f does not have type double?
                ("f", POINTER(c_double)),
                ("f_map", POINTER(as_ctypes(IntType))),
                ("sidx", c_void_p)]


class CoordinatelessFunction(ufl.Coefficient):
    """A function on a mesh topology."""

    def __init__(self, function_space, val=None, name=None, dtype=ScalarType):
        """
        :param function_space: the :class:`.FunctionSpace`, or
            :class:`.MixedFunctionSpace` on which to build this
            :class:`Function`.

            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.
        :param val: NumPy array-like (or :class:`pyop2.Dat` or
            :class:`~.Vector`) providing initial values (optional).
            This :class:`Function` will share data with the provided
            value.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        """
        assert isinstance(function_space, (functionspaceimpl.FunctionSpace,
                                           functionspaceimpl.MixedFunctionSpace)), \
            "Can't make a CoordinatelessFunction defined on a " + str(type(function_space))

        ufl.Coefficient.__init__(self, function_space.ufl_element())

        self.comm = function_space.comm
        self._function_space = function_space
        self.uid = utils._new_uid()
        self._name = name or 'function_%d' % self.uid
        self._label = "a function"
        self._split = None

        if isinstance(val, vector.Vector):
            # Allow constructing using a vector.
            val = val.dat
        if isinstance(val, (op2.Dat, op2.DatView, op2.MixedDat, op2.Global)):
            assert val.comm == self.comm
            self.dat = val
        else:
            self.dat = function_space.make_dat(val, dtype, self.name(), uid=self.uid)

    @property
    def topological(self):
        """The underlying coordinateless function."""
        return self

    def copy(self, deepcopy=False):
        """Return a copy of this CoordinatelessFunction.

        :kwarg deepcopy: If ``True``, the new
            :class:`CoordinatelessFunction` will allocate new space
            and copy values.  If ``False``, the default, then the new
            :class:`CoordinatelessFunction` will share the dof values.
        """
        if deepcopy:
            val = type(self.dat)(self.dat)
        else:
            val = self.dat
        return type(self)(self.function_space(),
                          val=val, name=self.name(),
                          dtype=self.dat.dtype)

    def ufl_id(self):
        return self.uid

    def split(self):
        """Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        if self._split is None:
            self._split = tuple(CoordinatelessFunction(fs, dat, name="%s[%d]" % (self.name(), i))
                                for i, (fs, dat) in
                                enumerate(zip(self.function_space(), self.dat)))
        return self._split

    def sub(self, i):
        """Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :meth:`split`.

        If the :class:`Function` is defined on a
        rank-1 :class:`~.FunctionSpace`, this returns a proxy object
        indexing the ith component of the space, suitable for use in
        boundary condition application."""
        if len(self.function_space()) == 1 and self.function_space().rank == 1:
            fs = self.function_space().sub(i)
            return CoordinatelessFunction(fs, val=op2.DatView(self.dat, i),
                                          name="view[%d](%s)" % (i, self.name()))
        return self.split()[i]

    @property
    def cell_set(self):
        """The :class:`pyop2.Set` of cells for the mesh on which this
        :class:`Function` is defined."""
        return self.function_space()._mesh.cell_set

    @property
    def node_set(self):
        """A :class:`pyop2.Set` containing the nodes of this
        :class:`Function`. One or (for rank-1 and 2
        :class:`.FunctionSpace`\s) more degrees of freedom are stored
        at each node.
        """
        return self.function_space().node_set

    @property
    def dof_dset(self):
        """A :class:`pyop2.DataSet` containing the degrees of freedom of
        this :class:`Function`."""
        return self.function_space().dof_dset

    def cell_node_map(self, bcs=None):
        return self.function_space().cell_node_map(bcs)
    cell_node_map.__doc__ = functionspaceimpl.FunctionSpace.cell_node_map.__doc__

    def interior_facet_node_map(self, bcs=None):
        return self.function_space().interior_facet_node_map(bcs)
    interior_facet_node_map.__doc__ = functionspaceimpl.FunctionSpace.interior_facet_node_map.__doc__

    def exterior_facet_node_map(self, bcs=None):
        return self.function_space().exterior_facet_node_map(bcs)
    exterior_facet_node_map.__doc__ = functionspaceimpl.FunctionSpace.exterior_facet_node_map.__doc__

    def vector(self):
        """Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return vector.Vector(self.dat)

    def function_space(self):
        """Return the :class:`.FunctionSpace`, or
        :class:`.MixedFunctionSpace` on which this :class:`Function`
        is defined."""
        return self._function_space

    def name(self):
        """Return the name of this :class:`Function`"""
        return self._name

    def label(self):
        """Return the label (a description) of this :class:`Function`"""
        return self._label

    def rename(self, name=None, label=None):
        """Set the name and or label of this :class:`Function`

        :arg name: The new name of the `Function` (if not `None`)
        :arg label: The new label for the `Function` (if not `None`)
        """
        if name is not None:
            self._name = name
        if label is not None:
            self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Function, self).__str__()


class Function(ufl.Coefficient):
    """A :class:`Function` represents a discretised field over the
    domain defined by the underlying :func:`.Mesh`. Functions are
    represented as sums of basis functions:

    .. math::

            f = \\sum_i f_i \phi_i(x)

    The :class:`Function` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`.FunctionSpace` object
    which provides the basis functions :math:`\\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Function` is vector-valued then this is specified in
    the :class:`.FunctionSpace`.
    """

    def __init__(self, function_space, val=None, name=None, dtype=ScalarType):
        """
        :param function_space: the :class:`.FunctionSpace`,
            or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.
        :param val: NumPy array-like (or :class:`pyop2.Dat`) providing initial values (optional).
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        """

        V = function_space
        if isinstance(V, Function):
            V = V.function_space()
        elif not isinstance(V, functionspaceimpl.WithGeometry):
            raise NotImplementedError("Can't make a Function defined on a "
                                      + str(type(function_space)))

        if isinstance(val, CoordinatelessFunction):
            if val.function_space() != V.topological:
                raise ValueError("Function values have wrong function space.")
            self._data = val
        else:
            self._data = CoordinatelessFunction(V.topological,
                                                val=val, name=name, dtype=dtype)

        self._function_space = V
        ufl.Coefficient.__init__(self, self.function_space().ufl_function_space())
        self._split = None

        if cachetools:
            # LRU cache for expressions assembled onto this function
            self._expression_cache = cachetools.LRUCache(maxsize=50)
        else:
            self._expression_cache = None

        if isinstance(function_space, Function):
            self.assign(function_space)

    @property
    def topological(self):
        """The underlying coordinateless function."""
        return self._data

    def copy(self, deepcopy=False):
        """Return a copy of this Function.

        :kwarg deepcopy: If ``True``, the new :class:`Function` will
            allocate new space and copy values.  If ``False``, the
            default, then the new :class:`Function` will share the dof
            values.
        """
        val = self.topological.copy(deepcopy=deepcopy)
        return type(self)(self.function_space(), val=val)

    def __getattr__(self, name):
        return getattr(self._data, name)

    def split(self):
        """Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        if self._split is None:
            self._split = tuple(Function(fs, dat, name="%s[%d]" % (self.name(), i))
                                for i, (fs, dat) in
                                enumerate(zip(self.function_space(), self.dat)))
        return self._split

    def sub(self, i):
        """Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :meth:`split`.

        If the :class:`Function` is defined on a
        :class:`~.VectorFunctionSpace`, this returns a proxy object
        indexing the ith component of the space, suitable for use in
        boundary condition application."""
        if len(self.function_space()) == 1 and self.function_space().rank == 1:
            fs = self.function_space().sub(i)
            return Function(fs, val=op2.DatView(self.dat, i),
                            name="view[%d](%s)" % (i, self.name()))
        return self.split()[i]

    def project(self, b, *args, **kwargs):
        """Project ``b`` onto ``self``. ``b`` must be a :class:`Function` or an
        :class:`.Expression`.

        This is equivalent to ``project(b, self)``.
        Any of the additional arguments to :func:`~firedrake.projection.project`
        may also be passed, and they will have their usual effect.
        """
        from firedrake import projection
        return projection.project(b, self, *args, **kwargs)

    def function_space(self):
        """Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            on which this :class:`Function` is defined.
        """
        return self._function_space

    def vector(self):
        """Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return vector.Vector(self.dat)

    def interpolate(self, expression, subset=None):
        """Interpolate an expression onto this :class:`Function`.

        :param expression: :class:`.Expression` or a UFL expression to interpolate
        :returns: this :class:`Function` object"""
        from firedrake import interpolation
        return interpolation.interpolate(expression, self, subset=subset)

    @utils.known_pyop2_safe
    def assign(self, expr, subset=None):
        """Set the :class:`Function` value to the pointwise value of
        expr. expr may only contain :class:`Function`\s on the same
        :class:`.FunctionSpace` as the :class:`Function` being assigned to.

        Similar functionality is available for the augmented assignment
        operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
        both Functions on the same :class:`.FunctionSpace` then::

          f += 2 * g

        will add twice `g` to `f`.

        If present, subset must be an :class:`pyop2.Subset` of this
        :class:`Function`'s ``node_set``.  The expression will then
        only be assigned to the nodes on that subset.
        """

        if isinstance(expr, Function) and \
           expr.function_space() == self.function_space():
            expr.dat.copy(self.dat, subset=subset)
            return self

        from firedrake import assemble_expressions
        assemble_expressions.evaluate_expression(
            assemble_expressions.Assign(self, expr), subset)
        return self

    @utils.known_pyop2_safe
    def __iadd__(self, expr):

        if np.isscalar(expr):
            self.dat += expr
            return self
        if isinstance(expr, Function) and \
           expr.function_space() == self.function_space():
            self.dat += expr.dat
            return self

        from firedrake import assemble_expressions
        assemble_expressions.evaluate_expression(
            assemble_expressions.IAdd(self, expr))

        return self

    @utils.known_pyop2_safe
    def __isub__(self, expr):

        if np.isscalar(expr):
            self.dat -= expr
            return self
        if isinstance(expr, Function) and \
           expr.function_space() == self.function_space():
            self.dat -= expr.dat
            return self

        from firedrake import assemble_expressions
        assemble_expressions.evaluate_expression(
            assemble_expressions.ISub(self, expr))

        return self

    @utils.known_pyop2_safe
    def __imul__(self, expr):

        if np.isscalar(expr):
            self.dat *= expr
            return self
        if isinstance(expr, Function) and \
           expr.function_space() == self.function_space():
            self.dat *= expr.dat
            return self

        from firedrake import assemble_expressions
        assemble_expressions.evaluate_expression(
            assemble_expressions.IMul(self, expr))

        return self

    @utils.known_pyop2_safe
    def __idiv__(self, expr):

        if np.isscalar(expr):
            self.dat /= expr
            return self
        if isinstance(expr, Function) and \
           expr.function_space() == self.function_space():
            self.dat /= expr.dat
            return self

        from firedrake import assemble_expressions
        assemble_expressions.evaluate_expression(
            assemble_expressions.IDiv(self, expr))

        return self

    __itruediv__ = __idiv__

    @utils.cached_property
    def _constant_ctypes(self):
        # Retrieve data from Python object
        function_space = self.function_space()
        mesh = function_space.mesh()
        coordinates = mesh.coordinates
        coordinates_space = coordinates.function_space()

        # Store data into ``C struct''
        c_function = _CFunction()
        c_function.n_cols = mesh.num_cells()
        if hasattr(mesh, '_layers'):
            c_function.n_layers = mesh.layers - 1
        else:
            c_function.n_layers = 1
        c_function.coords = coordinates.dat.data.ctypes.data_as(POINTER(c_double))
        c_function.coords_map = coordinates_space.cell_node_list.ctypes.data_as(POINTER(as_ctypes(IntType)))
        # FIXME: What about complex?
        c_function.f = self.dat.data.ctypes.data_as(POINTER(c_double))
        c_function.f_map = function_space.cell_node_list.ctypes.data_as(POINTER(as_ctypes(IntType)))
        return c_function

    @property
    def _ctypes(self):
        mesh = self.ufl_domain()
        c_function = self._constant_ctypes
        c_function.sidx = mesh.spatial_index and mesh.spatial_index.ctypes

        # Return pointer
        return ctypes.pointer(c_function)

    def _c_evaluate(self, tolerance=None):
        cache = self.__dict__.setdefault("_c_evaluate_cache", {})
        try:
            return cache[tolerance]
        except KeyError:
            result = make_c_evaluate(self, tolerance=tolerance)
            result.argtypes = [POINTER(_CFunction), POINTER(c_double), POINTER(c_double)]
            result.restype = c_int
            return cache.setdefault(tolerance, result)

    def evaluate(self, coord, mapping, component, index_values):
        # Called by UFL when evaluating expressions at coordinates
        if component or index_values:
            raise NotImplementedError("Unsupported arguments when attempting to evaluate Function.")
        return self.at(coord)

    def at(self, arg, *args, **kwargs):
        """Evaluate function at points.

        :arg arg: The point to locate.
        :arg args: Additional points.
        :kwarg dont_raise: Do not raise an error if a point is not found.
        :kwarg tolerance: Tolerance to use when checking for points in cell.
        """
        # Need to ensure data is up-to-date for reading
        self.dat._force_evaluation(read=True, write=False)
        from mpi4py import MPI

        if args:
            arg = (arg,) + args
        arg = np.array(arg, dtype=float)

        dont_raise = kwargs.get('dont_raise', False)

        tolerance = kwargs.get('tolerance', None)
        # Handle f.at(0.3)
        if not arg.shape:
            arg = arg.reshape(-1)

        # Immersed not supported
        tdim = self.function_space().mesh().ufl_cell().topological_dimension()
        gdim = self.function_space().mesh().ufl_cell().geometric_dimension()
        if tdim < gdim:
            raise NotImplementedError("Point is almost certainly not on the manifold.")

        # Validate geometric dimension
        if arg.shape[-1] == gdim:
            pass
        elif len(arg.shape) == 1 and gdim == 1:
            arg = arg.reshape(-1, 1)
        else:
            raise ValueError("Point dimension (%d) does not match geometric dimension (%d)." % (arg.shape[-1], gdim))

        # Check if we have got the same points on each process
        root_arg = self.comm.bcast(arg, root=0)
        same_arg = arg.shape == root_arg.shape and np.allclose(arg, root_arg)
        diff_arg = self.comm.allreduce(int(not same_arg), op=MPI.SUM)
        if diff_arg:
            raise ValueError("Points to evaluate are inconsistent among processes.")

        def single_eval(x, buf):
            """Helper function to evaluate at a single point."""
            err = self._c_evaluate(tolerance=tolerance)(self._ctypes,
                                                        x.ctypes.data_as(POINTER(c_double)),
                                                        buf.ctypes.data_as(POINTER(c_double)))
            if err == -1:
                raise PointNotInDomainError(self.function_space().mesh(), x.reshape(-1))

        if not len(arg.shape) <= 2:
            raise ValueError("Function.at expects point or array of points.")
        points = arg.reshape(-1, arg.shape[-1])
        value_shape = self.ufl_shape

        split = self.split()
        mixed = len(split) != 1

        # Local evaluation
        l_result = []
        for i, p in enumerate(points):
            try:
                if mixed:
                    l_result.append((i, tuple(f.at(p) for f in split)))
                else:
                    p_result = np.zeros(value_shape, dtype=float)
                    single_eval(points[i:i+1], p_result)
                    l_result.append((i, p_result))
            except PointNotInDomainError:
                # Skip point
                pass

        # Collecting the results
        def same_result(a, b):
            if mixed:
                for a_, b_ in zip(a, b):
                    if not np.allclose(a_, b_):
                        return False
                return True
            else:
                return np.allclose(a, b)

        all_results = self.comm.allgather(l_result)
        g_result = [None] * len(points)
        for results in all_results:
            for i, result in results:
                if g_result[i] is None:
                    g_result[i] = result
                elif same_result(result, g_result[i]):
                    pass
                else:
                    raise RuntimeError("Point evaluation gave different results across processes.")

        if not dont_raise:
            for i in range(len(g_result)):
                if g_result[i] is None:
                    raise PointNotInDomainError(self.function_space().mesh(), points[i].reshape(-1))

        if len(arg.shape) == 1:
            g_result = g_result[0]
        return g_result


class PointNotInDomainError(Exception):
    """Raised when attempting to evaluate a function outside its domain,
    and no fill value was given.

    Attributes: domain, point
    """
    def __init__(self, domain, point):
        self.domain = domain
        self.point = point

    def __str__(self):
        return "domain %s does not contain point %s" % (self.domain, self.point)


def make_c_evaluate(function, c_name="evaluate", ldargs=None, tolerance=None):
    """Generates, compiles and loads a C function to evaluate the
    given Firedrake :class:`Function`."""

    from os import path
    from firedrake.pointeval_utils import compile_element
    from pyop2 import compilation
    from pyop2.utils import get_petsc_dir
    from pyop2.base import build_itspace
    from pyop2.sequential import generate_cell_wrapper
    import firedrake.pointquery_utils as pq_utils

    mesh = function.ufl_domain()
    src = pq_utils.src_locate_cell(mesh, tolerance=tolerance)
    src += compile_element(function, mesh.coordinates)

    args = []

    arg = mesh.coordinates.dat(op2.READ, mesh.coordinates.cell_node_map())
    arg.position = 0
    args.append(arg)

    arg = function.dat(op2.READ, function.cell_node_map())
    arg.position = 1
    args.append(arg)

    src += generate_cell_wrapper(build_itspace(args, mesh.cell_set), args,
                                 forward_args=["double*", "double*"],
                                 kernel_name="evaluate_kernel",
                                 wrapper_name="wrap_evaluate")

    if ldargs is None:
        ldargs = []
    ldargs += ["-L%s/lib" % sys.prefix, "-lspatialindex_c", "-Wl,-rpath,%s/lib" % sys.prefix]
    return compilation.load(src, "c", c_name,
                            cppargs=["-I%s" % path.dirname(__file__),
                                     "-I%s/include" % sys.prefix] +
                            ["-I%s/include" % d for d in get_petsc_dir()],
                            ldargs=ldargs,
                            comm=function.comm)

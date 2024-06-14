import numpy as np
import rtree
import sys
import ufl
from ufl.duals import is_dual
from ufl.formatting.ufl2unicode import ufl2unicode
from ufl.domain import extract_unique_domain
import cachetools
import ctypes
from ctypes import POINTER, c_int, c_double, c_void_p
from collections.abc import Collection
from numbers import Number
from pathlib import Path

from pyop2 import op2, mpi
from pyop2.exceptions import DataTypeError, DataValueError

from firedrake.utils import ScalarType, IntType, as_ctypes

from firedrake import functionspaceimpl
from firedrake.cofunction import Cofunction
from firedrake import utils
from firedrake import vector
from firedrake.adjoint_utils import FunctionMixin
from firedrake.petsc import PETSc


__all__ = ['Function', 'PointNotInDomainError', 'CoordinatelessFunction']


class _CFunction(ctypes.Structure):
    r"""C struct collecting data from a :class:`Function`"""
    _fields_ = [("n_cols", c_int),
                ("extruded", c_int),
                ("n_layers", c_int),
                ("coords", c_void_p),
                ("coords_map", POINTER(as_ctypes(IntType))),
                ("f", c_void_p),
                ("f_map", POINTER(as_ctypes(IntType))),
                ("sidx", c_void_p)]


class CoordinatelessFunction(ufl.Coefficient):
    r"""A function on a mesh topology."""

    def __init__(self, function_space, val=None, name=None, dtype=ScalarType):
        r"""
        :param function_space: the :class:`.FunctionSpace`, or
            :class:`.MixedFunctionSpace` on which to build this
            :class:`Function`.

            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.
        :param val: NumPy array-like (or :class:`pyop2.types.dat.Dat` or
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

        ufl.Coefficient.__init__(self, function_space.ufl_function_space())

        # User comm
        self.comm = function_space.comm
        # Internal comm
        self._comm = mpi.internal_comm(function_space.comm, self)
        self._function_space = function_space
        self.uid = utils._new_uid(self._comm)
        self._name = name or 'function_%d' % self.uid
        self._label = "a function"

        if isinstance(val, vector.Vector):
            # Allow constructing using a vector.
            val = val.dat
        if isinstance(val, (op2.Dat, op2.DatView, op2.MixedDat, op2.Global)):
            assert val.comm == self._comm
            self.dat = val
        else:
            self.dat = function_space.make_dat(val, dtype, self.name())

    @utils.cached_property
    def topological(self):
        r"""The underlying coordinateless function."""
        return self

    @PETSc.Log.EventDecorator()
    def copy(self, deepcopy=False):
        r"""Return a copy of this CoordinatelessFunction.

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

    @utils.cached_property
    def subfunctions(self):
        r"""Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        return tuple(CoordinatelessFunction(fs, dat, name="%s[%d]" % (self.name(), i))
                     for i, (fs, dat) in
                     enumerate(zip(self.function_space(), self.dat)))

    @PETSc.Log.EventDecorator()
    def split(self):
        import warnings
        warnings.warn("The .split() method is deprecated, please use the .subfunctions property instead", category=FutureWarning)
        return self.subfunctions

    @utils.cached_property
    def _components(self):
        if self.dof_dset.cdim == 1:
            return (self, )
        else:
            return tuple(CoordinatelessFunction(self.function_space().sub(i), val=op2.DatView(self.dat, j),
                                                name="view[%d](%s)" % (i, self.name()))
                         for i, j in enumerate(np.ndindex(self.dof_dset.dim)))

    @PETSc.Log.EventDecorator()
    def sub(self, i):
        r"""Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :attr:`subfunctions`.

        If the :class:`Function` is defined on a
        rank-n :class:`~.FunctionSpace`, this returns a proxy object
        indexing the ith component of the space, suitable for use in
        boundary condition application."""
        if len(self.function_space()) == 1:
            return self._components[i]
        return self.subfunctions[i]

    @property
    def cell_set(self):
        r"""The :class:`pyop2.types.set.Set` of cells for the mesh on which this
        :class:`Function` is defined."""
        return self.function_space()._mesh.cell_set

    @property
    def node_set(self):
        r"""A :class:`pyop2.types.set.Set` containing the nodes of this
        :class:`Function`. One or (for rank-1 and 2
        :class:`.FunctionSpace`\s) more degrees of freedom are stored
        at each node.
        """
        return self.function_space().node_set

    @property
    def dof_dset(self):
        r"""A :class:`pyop2.types.dataset.DataSet` containing the degrees of freedom of
        this :class:`Function`."""
        return self.function_space().dof_dset

    def cell_node_map(self):
        return self.function_space().cell_node_map()
    cell_node_map.__doc__ = functionspaceimpl.FunctionSpace.cell_node_map.__doc__

    def interior_facet_node_map(self):
        return self.function_space().interior_facet_node_map()
    interior_facet_node_map.__doc__ = functionspaceimpl.FunctionSpace.interior_facet_node_map.__doc__

    def exterior_facet_node_map(self):
        return self.function_space().exterior_facet_node_map()
    exterior_facet_node_map.__doc__ = functionspaceimpl.FunctionSpace.exterior_facet_node_map.__doc__

    def vector(self):
        r"""Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return vector.Vector(self)

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or
        :class:`.MixedFunctionSpace` on which this :class:`Function`
        is defined."""
        return self._function_space

    def name(self):
        r"""Return the name of this :class:`Function`"""
        return self._name

    def label(self):
        r"""Return the label (a description) of this :class:`Function`"""
        return self._label

    def rename(self, name=None, label=None):
        r"""Set the name and or label of this :class:`Function`

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
            return ufl2unicode(self)


class Function(ufl.Coefficient, FunctionMixin):
    r"""A :class:`Function` represents a discretised field over the
    domain defined by the underlying :func:`.Mesh`. Functions are
    represented as sums of basis functions:

    .. math::

      f = \sum_i f_i \phi_i(x)

    The :class:`Function` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`.FunctionSpace` object
    which provides the basis functions :math:`\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Function` is vector-valued then this is specified in
    the :class:`.FunctionSpace`.
    """

    def __new__(cls, *args, **kwargs):
        if args[0] and is_dual(args[0]):
            return Cofunction(*args, **kwargs)
        return super().__new__(cls, *args, **kwargs)

    @PETSc.Log.EventDecorator()
    @FunctionMixin._ad_annotate_init
    def __init__(self, function_space, val=None, name=None, dtype=ScalarType,
                 count=None):
        r"""
        :param function_space: the :class:`.FunctionSpace`,
            or :class:`.MixedFunctionSpace` on which to build this :class:`Function`.
            Alternatively, another :class:`Function` may be passed here and its function space
            will be used to build this :class:`Function`.  In this
            case, the function values are copied.
        :param val: NumPy array-like (or :class:`pyop2.types.dat.Dat`) providing initial values (optional).
            If val is an existing :class:`Function`, then the data will be shared.
        :param name: user-defined name for this :class:`Function` (optional).
        :param dtype: optional data type for this :class:`Function`
               (defaults to ``ScalarType``).
        :param count: The :class:`ufl.Coefficient` count which creates the
            symbolic identity of this :class:`Function`.
        """

        V = function_space
        if isinstance(V, Function):
            V = V.function_space()
        elif not isinstance(V, functionspaceimpl.WithGeometry):
            raise NotImplementedError("Can't make a Function defined on a "
                                      + str(type(function_space)))

        if isinstance(val, (Function, CoordinatelessFunction)):
            val = val.topological
            if val.function_space() != V.topological:
                raise ValueError("Function values have wrong function space.")
            self._data = val
        else:
            self._data = CoordinatelessFunction(V.topological,
                                                val=val, name=name, dtype=dtype)

        self._function_space = V
        ufl.Coefficient.__init__(
            self, self.function_space().ufl_function_space(), count=count
        )

        # LRU cache for expressions assembled onto this function
        self._expression_cache = cachetools.LRUCache(maxsize=50)

        if isinstance(function_space, Function):
            self.assign(function_space)

    @property
    def topological(self):
        r"""The underlying coordinateless function."""
        return self._data

    @PETSc.Log.EventDecorator()
    @FunctionMixin._ad_annotate_copy
    def copy(self, deepcopy=False):
        r"""Return a copy of this Function.

        :kwarg deepcopy: If ``True``, the new :class:`Function` will
            allocate new space and copy values.  If ``False``, the
            default, then the new :class:`Function` will share the dof
            values.
        """
        val = self.topological.copy(deepcopy=deepcopy)
        return type(self)(self.function_space(), val=val)

    def __getattr__(self, name):
        val = getattr(self._data, name)
        return val

    def __dir__(self):
        current = super(Function, self).__dir__()
        return list(dict.fromkeys(dir(self._data) + current))

    @utils.cached_property
    @FunctionMixin._ad_annotate_subfunctions
    def subfunctions(self):
        r"""Extract any sub :class:`Function`\s defined on the component spaces
        of this this :class:`Function`'s :class:`.FunctionSpace`."""
        return tuple(type(self)(V, val)
                     for (V, val) in zip(self.function_space(), self.topological.subfunctions))

    @FunctionMixin._ad_annotate_subfunctions
    def split(self):
        import warnings
        warnings.warn("The .split() method is deprecated, please use the .subfunctions property instead", category=FutureWarning)
        return self.subfunctions

    @utils.cached_property
    def _components(self):
        if self.function_space().value_size == 1:
            return (self, )
        else:
            return tuple(type(self)(self.function_space().sub(i), self.topological.sub(i))
                         for i in range(self.function_space().value_size))

    @PETSc.Log.EventDecorator()
    def sub(self, i):
        r"""Extract the ith sub :class:`Function` of this :class:`Function`.

        :arg i: the index to extract

        See also :attr:`subfunctions`.

        If the :class:`Function` is defined on a
        :func:`~.VectorFunctionSpace` or :func:`~.TensorFunctionSpace` this returns a proxy object
        indexing the ith component of the space, suitable for use in
        boundary condition application."""
        if len(self.function_space()) == 1:
            return self._components[i]
        return self.subfunctions[i]

    @PETSc.Log.EventDecorator()
    @FunctionMixin._ad_annotate_project
    def project(self, b, *args, **kwargs):
        r"""Project ``b`` onto ``self``. ``b`` must be a :class:`Function` or a
        UFL expression.

        This is equivalent to ``project(b, self)``.
        Any of the additional arguments to :func:`~firedrake.projection.project`
        may also be passed, and they will have their usual effect.
        """
        from firedrake import projection
        return projection.project(b, self, *args, **kwargs)

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            on which this :class:`Function` is defined.
        """
        return self._function_space

    def vector(self):
        r"""Return a :class:`.Vector` wrapping the data in this :class:`Function`"""
        return vector.Vector(self)

    @PETSc.Log.EventDecorator()
    def interpolate(
        self,
        expression,
        subset=None,
        allow_missing_dofs=False,
        default_missing_val=None,
        ad_block_tag=None
    ):
        r"""Interpolate an expression onto this :class:`Function`.

        :param expression: a UFL expression to interpolate
        :kwarg subset: An optional :class:`pyop2.types.set.Subset` to apply the
            interpolation over. Cannot, at present, be used when interpolating
            across meshes unless the target mesh is a :func:`.VertexOnlyMesh`.
        :kwarg allow_missing_dofs: For interpolation across meshes: allow
            degrees of freedom (aka DoFs/nodes) in the target mesh that cannot be
            defined on the source mesh. For example, where nodes are point
            evaluations, points in the target mesh that are not in the source mesh.
            When ``False`` this raises a ``ValueError`` should this occur. When
            ``True`` the corresponding values are set to zero or to the value
            ``default_missing_val`` if given. Ignored if interpolating within the
            same mesh or onto a :func:`.VertexOnlyMesh` (the behaviour of a
            :func:`.VertexOnlyMesh` in this scenario is, at present, set when
            it is created).
        :kwarg default_missing_val: For interpolation across meshes: the optional
            value to assign to DoFs in the target mesh that are outside the source
            mesh. If this is not set then zero is used. Ignored if interpolating
            within the same mesh or onto a :func:`.VertexOnlyMesh`.
        :kwarg ad_block_tag: An optional string for tagging the resulting assemble block on
            the Pyadjoint tape.
        :returns: this :class:`Function` object"""
        from firedrake import interpolation, assemble
        V = self.function_space()
        interp = interpolation.Interpolate(expression, V,
                                           subset=subset,
                                           allow_missing_dofs=allow_missing_dofs,
                                           default_missing_val=default_missing_val)
        return assemble(interp, tensor=self, ad_block_tag=ad_block_tag)

    def zero(self, subset=None):
        """Set all values to zero.

        Parameters
        ----------
        subset : pyop2.types.set.Subset
                 A subset of the domain indicating the nodes to zero.
                 If `None` then the whole function is zeroed.

        Returns
        -------
        firedrake.function.Function
            Returns `self`
        """
        # Use assign here so we can reuse _ad_annotate_assign instead of needing
        # to write an _ad_annotate_zero function
        return self.assign(0, subset=subset)

    @PETSc.Log.EventDecorator()
    @FunctionMixin._ad_annotate_assign
    def assign(self, expr, subset=None):
        r"""Set the :class:`Function` value to the pointwise value of
        expr. expr may only contain :class:`Function`\s on the same
        :class:`.FunctionSpace` as the :class:`Function` being assigned to.

        Similar functionality is available for the augmented assignment
        operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
        both Functions on the same :class:`.FunctionSpace` then::

          f += 2 * g

        will add twice `g` to `f`.

        If present, subset must be an :class:`pyop2.types.set.Subset` of this
        :class:`Function`'s ``node_set``.  The expression will then
        only be assigned to the nodes on that subset.

        .. note::

            Assignment can only be performed for simple weighted sum expressions and constant
            values. Things like ``u.assign(2*v + Constant(3.0))``. For more complicated
            expressions (e.g. involving the product of functions) :meth:`.Function.interpolate`
            should be used.
        """
        if self.ufl_element().family() == "Real" and isinstance(expr, (Number, Collection)):
            try:
                self.dat.data_wo[...] = expr
            except (DataTypeError, DataValueError) as e:
                raise ValueError(e)
        elif expr == 0:
            self.dat.zero(subset=subset)
        else:
            from firedrake.assign import Assigner
            Assigner(self, expr, subset).assign()
        return self

    def riesz_representation(self, riesz_map='L2'):
        """Return the Riesz representation of this :class:`Function` with respect to the given Riesz map.

        Example: For a L2 Riesz map, the Riesz representation is obtained by taking the action
        of ``M`` on ``self``, where M is the L2 mass matrix, i.e. M = <u, v>
        with u and v trial and test functions, respectively.

        Parameters
        ----------
        riesz_map : str or collections.abc.Callable
                    The Riesz map to use (`l2`, `L2`, or `H1`). This can also be a callable.

        Returns
        -------
        firedrake.cofunction.Cofunction
            Riesz representation of this :class:`Function` with respect to the given Riesz map.
        """
        from firedrake.ufl_expr import action
        from firedrake.assemble import assemble

        V = self.function_space()
        if riesz_map == "l2":
            return Cofunction(V.dual(), val=self.dat)

        elif riesz_map in ("L2", "H1"):
            a = self._define_riesz_map_form(riesz_map, V)
            return assemble(action(a, self))

        elif callable(riesz_map):
            return riesz_map(self)

        else:
            raise NotImplementedError(
                "Unknown Riesz representation %s" % riesz_map)

    @FunctionMixin._ad_annotate_iadd
    def __iadd__(self, expr):
        from firedrake.assign import IAddAssigner
        IAddAssigner(self, expr).assign()
        return self

    @FunctionMixin._ad_annotate_isub
    def __isub__(self, expr):
        from firedrake.assign import ISubAssigner
        ISubAssigner(self, expr).assign()
        return self

    @FunctionMixin._ad_annotate_imul
    def __imul__(self, expr):
        from firedrake.assign import IMulAssigner
        IMulAssigner(self, expr).assign()
        return self

    @FunctionMixin._ad_annotate_idiv
    def __itruediv__(self, expr):
        from firedrake.assign import IDivAssigner
        IDivAssigner(self, expr).assign()
        return self

    def __float__(self):

        if (
            self.ufl_element().family() == "Real"
            and self.function_space().shape == ()
        ):
            return float(self.dat.data_ro[0])
        else:
            raise ValueError("Can only cast scalar 'Real' Functions to float.")

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
        if mesh.layers is not None:
            # TODO: assert constant layer. Can we do variable though?
            c_function.extruded = 1
            c_function.n_layers = mesh.layers - 1
        else:
            c_function.extruded = 0
            c_function.n_layers = 1
        c_function.coords = coordinates.dat.data_ro.ctypes.data_as(c_void_p)
        c_function.coords_map = coordinates_space.cell_node_list.ctypes.data_as(POINTER(as_ctypes(IntType)))
        c_function.f = self.dat.data_ro.ctypes.data_as(c_void_p)
        c_function.f_map = function_space.cell_node_list.ctypes.data_as(POINTER(as_ctypes(IntType)))
        return c_function

    @property
    def _ctypes(self):
        mesh = extract_unique_domain(self)
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
            result.argtypes = [POINTER(_CFunction), POINTER(c_double), c_void_p]
            result.restype = c_int
            return cache.setdefault(tolerance, result)

    def evaluate(self, coord, mapping, component, index_values):
        # Called by UFL when evaluating expressions at coordinates
        if component or index_values:
            raise NotImplementedError("Unsupported arguments when attempting to evaluate Function.")
        return self.at(coord)

    @PETSc.Log.EventDecorator()
    def at(self, arg, *args, **kwargs):
        r"""Evaluate function at points.

        :arg arg: The point to locate.
        :arg args: Additional points.
        :kwarg dont_raise: Do not raise an error if a point is not found.
        :kwarg tolerance: Tolerence to use when checking if a point is
            in a cell. Default is the ``tolerance`` provided when
            creating the :func:`~.Mesh` the function is defined on.
            Changing this from default will cause the spatial index to
            be rebuilt which can take some time.
        """
        # Shortcut if function space is the R-space
        if self.ufl_element().family() == "Real":
            return self.dat.data_ro

        # Need to ensure data is up-to-date for reading
        self.dat.global_to_local_begin(op2.READ)
        self.dat.global_to_local_end(op2.READ)
        from mpi4py import MPI

        if args:
            arg = (arg,) + args
        arg = np.asarray(arg, dtype=utils.ScalarType)
        if utils.complex_mode:
            if not np.allclose(arg.imag, 0):
                raise ValueError("Provided points have non-zero imaginary part")
            arg = arg.real.copy()

        dont_raise = kwargs.get('dont_raise', False)

        tolerance = kwargs.get('tolerance', None)
        mesh = self.function_space().mesh()
        if tolerance is None:
            tolerance = mesh.tolerance
        else:
            mesh.tolerance = tolerance

        # Handle f.at(0.3)
        if not arg.shape:
            arg = arg.reshape(-1)

        if mesh.variable_layers:
            raise NotImplementedError("Point evaluation not implemented for variable layers")

        # Validate geometric dimension
        gdim = mesh.ufl_cell().geometric_dimension()
        if arg.shape[-1] == gdim:
            pass
        elif len(arg.shape) == 1 and gdim == 1:
            arg = arg.reshape(-1, 1)
        else:
            raise ValueError("Point dimension (%d) does not match geometric dimension (%d)." % (arg.shape[-1], gdim))

        # Check if we have got the same points on each process
        root_arg = self._comm.bcast(arg, root=0)
        same_arg = arg.shape == root_arg.shape and np.allclose(arg, root_arg)
        diff_arg = self._comm.allreduce(int(not same_arg), op=MPI.SUM)
        if diff_arg:
            raise ValueError("Points to evaluate are inconsistent among processes.")

        def single_eval(x, buf):
            r"""Helper function to evaluate at a single point."""
            err = self._c_evaluate(tolerance=tolerance)(self._ctypes,
                                                        x.ctypes.data_as(POINTER(c_double)),
                                                        buf.ctypes.data_as(c_void_p))
            if err == -1:
                raise PointNotInDomainError(self.function_space().mesh(), x.reshape(-1))

        if not len(arg.shape) <= 2:
            raise ValueError("Function.at expects point or array of points.")
        points = arg.reshape(-1, arg.shape[-1])
        value_shape = self.ufl_shape

        subfunctions = self.subfunctions
        mixed = len(subfunctions) != 1

        # Local evaluation
        l_result = []
        for i, p in enumerate(points):
            try:
                if mixed:
                    l_result.append((i, tuple(f.at(p) for f in subfunctions)))
                else:
                    p_result = np.zeros(value_shape, dtype=ScalarType)
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

    def __str__(self):
        return ufl2unicode(self)


class PointNotInDomainError(Exception):
    r"""Raised when attempting to evaluate a function outside its domain,
    and no fill value was given.

    Attributes: domain, point
    """

    def __init__(self, domain, point):
        self.domain = domain
        self.point = point

    def __str__(self):
        return "domain %s does not contain point %s" % (self.domain, self.point)


@PETSc.Log.EventDecorator()
def make_c_evaluate(function, c_name="evaluate", ldargs=None, tolerance=None):
    r"""Generates, compiles and loads a C function to evaluate the
    given Firedrake :class:`Function`."""

    from os import path
    from firedrake.pointeval_utils import compile_element
    from pyop2 import compilation
    from pyop2.utils import get_petsc_dir
    from pyop2.parloop import generate_single_cell_wrapper
    import firedrake.pointquery_utils as pq_utils

    mesh = extract_unique_domain(function)
    src = [pq_utils.src_locate_cell(mesh, tolerance=tolerance)]
    src.append(compile_element(function, mesh.coordinates))

    args = []

    arg = mesh.coordinates.dat(op2.READ, mesh.coordinates.cell_node_map())
    args.append(arg)

    arg = function.dat(op2.READ, function.cell_node_map())
    args.append(arg)

    p_ScalarType_c = f"{utils.ScalarType_c}*"
    src.append(generate_single_cell_wrapper(mesh.cell_set, args,
                                            forward_args=[p_ScalarType_c,
                                                          p_ScalarType_c],
                                            kernel_name="evaluate_kernel",
                                            wrapper_name="wrap_evaluate"))

    src = "\n".join(src)

    if ldargs is None:
        ldargs = []
    libspatialindex_so = Path(rtree.core.rt._name).absolute()
    lsi_runpath = f"-Wl,-rpath,{libspatialindex_so.parent}"
    ldargs += [str(libspatialindex_so), lsi_runpath]
    return compilation.load(
        src, "c", c_name,
        cppargs=[
            f"-I{path.dirname(__file__)}",
            f"-I{sys.prefix}/include",
            f"-I{rtree.finder.get_include()}"
        ] + [f"-I{d}/include" for d in get_petsc_dir()],
        ldargs=ldargs,
        comm=function.comm
    )

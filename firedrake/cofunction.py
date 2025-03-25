from functools import cached_property
import numpy as np
import ufl

from ufl.form import BaseForm
from pyop2 import op2, mpi
from pyadjoint.tape import stop_annotating, annotate_tape, get_working_tape
from finat.ufl import MixedElement
import firedrake.assemble
import firedrake.functionspaceimpl as functionspaceimpl
from firedrake import utils, vector, ufl_expr
from firedrake.utils import ScalarType
from firedrake.adjoint_utils.function import CofunctionMixin
from firedrake.adjoint_utils.checkpointing import DelegatedFunctionCheckpoint
from firedrake.adjoint_utils.blocks.function import CofunctionAssignBlock
from firedrake.petsc import PETSc


class Cofunction(ufl.Cofunction, CofunctionMixin):
    r"""A :class:`Cofunction` represents a function on a dual space.

    Like Functions, cofunctions are represented as sums of basis functions:

    .. math::

            f = \\sum_i f_i \phi_i(x)

    The :class:`Cofunction` class provides storage for the coefficients
    :math:`f_i` and associates them with a :class:`.FunctionSpace` object
    which provides the basis functions :math:`\\phi_i(x)`.

    Note that the coefficients are always scalars: if the
    :class:`Cofunction` is vector-valued then this is specified in
    the :class:`.FunctionSpace`.
    """

    @PETSc.Log.EventDecorator()
    @CofunctionMixin._ad_annotate_init
    def __init__(self, function_space, val=None, name=None, dtype=ScalarType,
                 count=None):
        r"""
        :param function_space: the :class:`.FunctionSpace`,
            or :class:`.MixedFunctionSpace` on which to build this :class:`Cofunction`.
            Alternatively, another :class:`Cofunction` may be passed here and its function space
            will be used to build this :class:`Cofunction`.  In this
            case, the function values are copied.
        :param val: NumPy array-like (or :class:`pyop2.types.dat.Dat`) providing initial values (optional).
            If val is an existing :class:`Cofunction`, then the data will be shared.
        :param name: user-defined name for this :class:`Cofunction` (optional).
        :param dtype: optional data type for this :class:`Cofunction`
               (defaults to ``ScalarType``).
        """

        V = function_space
        if isinstance(V, Cofunction):
            V = V.function_space()
            # Deep copy prevents modifications to Vector copies.
            # Also, this discard the value of `val` if it was specified (consistent with Function)
            val = function_space.copy(deepcopy=True).dat
        elif not isinstance(V, functionspaceimpl.FiredrakeDualSpace):
            raise NotImplementedError("Can't make a Cofunction defined on a "
                                      + str(type(function_space)))

        ufl.Cofunction.__init__(self, V.ufl_function_space(), count=count)

        # User comm
        self.comm = V.comm
        # Internal comm
        self._comm = mpi.internal_comm(V.comm, self)
        self._function_space = V
        self.uid = utils._new_uid(self._comm)
        self._name = name or 'cofunction_%d' % self.uid
        self._label = "a cofunction"

        if isinstance(val, Cofunction):
            val = val.dat

        if isinstance(val, (op2.Dat, op2.DatView, op2.MixedDat, op2.Global)):
            assert val.comm == self._comm
            self.dat = val
        else:
            self.dat = function_space.make_dat(val, dtype, self.name())

        if isinstance(function_space, Cofunction):
            self.dat.copy(function_space.dat)

    @PETSc.Log.EventDecorator()
    def copy(self, deepcopy=True):
        r"""Return a copy of this :class:`firedrake.function.CoordinatelessFunction`.

        :kwarg deepcopy: If ``True``, the default, the new
            :class:`firedrake.function.CoordinatelessFunction` will allocate new space
            and copy values.  If ``False``, then the new
            :class:`firedrake.function.CoordinatelessFunction` will share the dof values.
        """
        if deepcopy:
            val = type(self.dat)(self.dat)
        else:
            val = self.dat
        return type(self)(self.function_space(),
                          val=val, name=self.name(),
                          dtype=self.dat.dtype)

    def _analyze_form_arguments(self):
        # Cofunctions have one argument in primal space as they map from V to R.
        self._arguments = (ufl_expr.Argument(self.function_space().dual(), 0),)
        self._coefficients = (self,)

    @utils.cached_property
    @CofunctionMixin._ad_annotate_subfunctions
    def subfunctions(self):
        r"""Extract any sub :class:`Cofunction`\s defined on the component spaces
        of this this :class:`Cofunction`'s :class:`.FunctionSpace`."""
        return tuple(type(self)(fs, dat) for fs, dat in zip(self.function_space(), self.dat))

    def split(self):
        import warnings
        warnings.warn("The .split() method is deprecated, please use the .subfunctions property instead", category=FutureWarning)
        return self.subfunctions

    @utils.cached_property
    def _components(self):
        if self.function_space().rank == 0:
            return (self, )
        else:
            if self.dof_dset.cdim == 1:
                return (type(self)(self.function_space().sub(0), val=self.dat),)
            else:
                return tuple(type(self)(self.function_space().sub(i), val=op2.DatView(self.dat, j))
                             for i, j in enumerate(np.ndindex(self.dof_dset.dim)))

    @PETSc.Log.EventDecorator()
    def sub(self, i):
        r"""Extract the ith sub :class:`Cofunction` of this :class:`Cofunction`.

        :arg i: the index to extract

        See also :attr:`subfunctions`.

        If the :class:`Cofunction` is defined on a
        :func:`~.VectorFunctionSpace` or :func:`~.TensorFunctionSpace`
        this returns a proxy object indexing the ith component of the space,
        suitable for use in boundary condition application."""
        mixed = type(self.function_space().ufl_element()) is MixedElement
        data = self.subfunctions if mixed else self._components
        return data[i]

    def function_space(self):
        r"""Return the :class:`.FunctionSpace`, or :class:`.MixedFunctionSpace`
            on which this :class:`Cofunction` is defined.
        """
        return self._function_space

    def equals(self, other):
        """Check equality."""
        if type(other) is not Cofunction:
            return False
        if self is other:
            return True
        return self._count == other._count and self._function_space == other._function_space

    def zero(self, subset=None):
        """Set values to zero.

        Parameters
        ----------
        subset : pyop2.types.set.Subset
                 A subset of the domain indicating the nodes to zero.
                 If `None` then the whole function is zeroed.

        Returns
        -------
        firedrake.cofunction.Cofunction
            Returns `self`
        """
        return self.assign(0, subset=subset)

    @PETSc.Log.EventDecorator()
    @utils.known_pyop2_safe
    def assign(self, expr, subset=None, expr_from_assemble=False):
        r"""Set the :class:`Cofunction` value to the pointwise value of
        expr. expr may only contain :class:`Cofunction`\s on the same
        :class:`.FunctionSpace` as the :class:`Cofunction` being assigned to.

        Similar functionality is available for the augmented assignment
        operators `+=`, `-=`, `*=` and `/=`. For example, if `f` and `g` are
        both Cofunctions on the same :class:`.FunctionSpace` then::

          f += 2 * g

        will add twice `g` to `f`.

        If present, subset must be an :class:`pyop2.types.set.Subset` of this
        :class:`Cofunction`'s ``node_set``.  The expression will then
        only be assigned to the nodes on that subset.

        The `expr_from_assemble` optional argument indicates whether the
        expression results from an assemble operation performed within the
        current method. `expr_from_assemble` is required for the
        `CofunctionAssignBlock`.
        """
        expr = ufl.as_ufl(expr)
        if isinstance(expr, ufl.classes.Zero):
            with stop_annotating(modifies=(self,)):
                self.dat.zero(subset=subset)
            return self
        elif (isinstance(expr, Cofunction)
              and expr.function_space() == self.function_space()):
            # do not annotate in case of self assignment
            if annotate_tape() and self != expr:
                if subset is not None:
                    raise NotImplementedError("Cofunction subset assignment "
                                              "annotation is not supported.")
                self.block_variable = self.create_block_variable()
                self.block_variable._checkpoint = DelegatedFunctionCheckpoint(
                    expr.block_variable)
                get_working_tape().add_block(
                    CofunctionAssignBlock(
                        self, expr, rhs_from_assemble=expr_from_assemble)
                )

            expr.dat.copy(self.dat, subset=subset)
            return self
        elif isinstance(expr, BaseForm):
            # Enable c.assign(B) where c is a Cofunction and B an appropriate
            # BaseForm object. If annotation is enabled, the following
            # operation will result in an assemble block on the Pyadjoint tape.
            assembled_expr = firedrake.assemble(expr)
            return self.assign(
                assembled_expr, subset=subset,
                expr_from_assemble=True)
        else:
            from firedrake.assign import Assigner
            Assigner(self, expr, subset).assign()
        return self

    def riesz_representation(self, riesz_map='L2', *, bcs=None,
                             solver_options=None,
                             form_compiler_parameters=None):
        """Return the Riesz representation of this :class:`Cofunction`.

        Example: For a L2 Riesz map, the Riesz representation is obtained by
        solving the linear system ``Mx = self``, where M is the L2 mass matrix,
        i.e. M = <u, v> with u and v trial and test functions, respectively.

        Parameters
        ----------
        riesz_map : str or ufl.sobolevspace.SobolevSpace or
        collections.abc.Callable
            The Riesz map to use (`l2`, `L2`, or `H1`). This can also be a
            callable.
        bcs: DirichletBC or list of DirichletBC
            Boundary conditions to apply to the Riesz map.
        solver_options: dict
            A dictionary of PETSc options to be passed to the solver.
        form_compiler_parameters: dict
            A dictionary of form compiler parameters to be passed to the
            variational problem that solves for the Riesz map.

        Returns
        -------
        firedrake.function.Function
            Riesz representation of this :class:`Cofunction` with respect to
            the given Riesz map.
        """
        if not callable(riesz_map):
            riesz_map = RieszMap(
                self.function_space(), riesz_map, bcs=bcs,
                solver_parameters=solver_options,
                form_compiler_parameters=form_compiler_parameters
            )

        return riesz_map(self)

    @CofunctionMixin._ad_annotate_iadd
    @utils.known_pyop2_safe
    def __iadd__(self, expr):

        if np.isscalar(expr):
            self.dat += expr
            return self
        if isinstance(expr, vector.Vector):
            expr = expr.function
        if isinstance(expr, Cofunction) and \
           expr.function_space() == self.function_space():
            self.dat += expr.dat
            return self
        # Let Python hit `BaseForm.__add__` which relies on ufl.FormSum.
        return NotImplemented

    @CofunctionMixin._ad_annotate_isub
    @utils.known_pyop2_safe
    def __isub__(self, expr):

        if np.isscalar(expr):
            self.dat -= expr
            return self
        if isinstance(expr, vector.Vector):
            expr = expr.function
        if isinstance(expr, Cofunction) and \
           expr.function_space() == self.function_space():
            self.dat -= expr.dat
            return self

        # Let Python hit `BaseForm.__sub__` which relies on ufl.FormSum.
        return NotImplemented

    @CofunctionMixin._ad_annotate_imul
    def __imul__(self, expr):

        if np.isscalar(expr):
            self.dat *= expr
            return self
        if isinstance(expr, vector.Vector):
            expr = expr.function
        if isinstance(expr, Cofunction) and \
           expr.function_space() == self.function_space():
            self.dat *= expr.dat
            return self
        return NotImplemented

    def interpolate(self, expression):
        r"""Interpolate an expression onto this :class:`Cofunction`.

        :param expression: a UFL expression to interpolate
        :returns: this :class:`firedrake.cofunction.Cofunction` object"""
        from firedrake import interpolation
        interp = interpolation.Interpolate(ufl_expr.Argument(self.function_space().dual(), 0), expression)
        return firedrake.assemble(interp, tensor=self)

    def vector(self):
        r"""Return a :class:`.Vector` wrapping the data in this
        :class:`Cofunction`"""
        return vector.Vector(self)

    @property
    def cell_set(self):
        r"""The :class:`pyop2.types.set.Set` of cells for the mesh on which this
        :class:`Cofunction` is defined."""
        return self.function_space()._mesh.cell_set

    @property
    def node_set(self):
        r"""A :class:`pyop2.types.set.Set` containing the nodes of this
        :class:`Cofunction`. One or (for rank-1 and 2
        :class:`.FunctionSpace`\s) more degrees of freedom are stored
        at each node.
        """
        return self.function_space().node_set

    @property
    def dof_dset(self):
        r"""A :class:`pyop2.types.dataset.DataSet` containing the degrees of freedom of
        this :class:`Cofunction`."""
        return self.function_space().dof_dset

    def ufl_id(self):
        return self.uid

    def name(self):
        r"""Return the name of this :class:`Cofunction`"""
        return self._name

    def label(self):
        r"""Return the label (a description) of this :class:`Cofunction`"""
        return self._label

    def rename(self, name=None, label=None):
        r"""Set the name and or label of this :class:`Cofunction`

        :arg name: The new name of the `Cofunction` (if not `None`)
        :arg label: The new label for the `Cofunction` (if not `None`)
        """
        if name is not None:
            self._name = name
        if label is not None:
            self._label = label

    def __str__(self):
        if self._name is not None:
            return self._name
        else:
            return super(Cofunction, self).__str__()

    def cell_node_map(self):
        return self.function_space().cell_node_map()


class RieszMap:
    """Return a map between dual and primal function spaces.

    A `RieszMap` can be called on a `Cofunction` in the appropriate space to
    yield the `Function` which is the Riesz representer under the given inner
    product. Conversely, it can be called on a `Function` to apply the given
    inner product and return a `Cofunction`.

    Parameters
    ----------
    function_space_or_inner_product: FunctionSpace or ufl.Form
        The space from which to map, or a bilinear form defining an inner
        product.
    sobolev_space: str or ufl.sobolevspace.SobolevSpace.
        Used to determine the inner product.
    bcs: DirichletBC or list of DirichletBC
        Boundary conditions to apply to the Riesz map.
    solver_parameters: dict
        A dictionary of PETSc options to be passed to the solver.
    form_compiler_parameters: dict
        A dictionary of form compiler parameters to be passed to the
        variational problem that solves for the Riesz map.
    """

    def __init__(self, function_space_or_inner_product=None,
                 sobolev_space=ufl.L2, *, bcs=None, solver_parameters=None,
                 form_compiler_parameters=None):
        if isinstance(function_space_or_inner_product, ufl.Form):
            args = ufl.algorithms.extract_arguments(
                function_space_or_inner_product
            )
            if len(args) != 2:
                raise ValueError(f"inner_product has arity {len(args)}, "
                                 "should be 2.")
            function_space = args[0].function_space()
            inner_product = function_space_or_inner_product
        else:
            function_space = function_space_or_inner_product
            if hasattr(function_space, "function_space"):
                function_space = function_space.function_space()
            if ufl.duals.is_dual(function_space):
                function_space = function_space.dual()

            if str(sobolev_space) == "l2":
                inner_product = "l2"
            else:
                from firedrake import TrialFunction, TestFunction
                u = TrialFunction(function_space)
                v = TestFunction(function_space)
                inner_product = RieszMap._inner_product_form(
                    sobolev_space, u, v
                )

        self._function_space = function_space
        self._inner_product = inner_product
        self._bcs = bcs
        self._solver_parameters = solver_parameters or {}
        self._form_compiler_parameters = form_compiler_parameters or {}

    @staticmethod
    def _inner_product_form(sobolev_space, u, v):
        from firedrake import inner, dx, grad
        inner_products = {
            "L2": lambda u, v: inner(u, v)*dx,
            "H1": lambda u, v: inner(u, v)*dx + inner(grad(u), grad(v))*dx
        }
        try:
            return inner_products[str(sobolev_space)](u, v)
        except KeyError:
            raise ValueError("No inner product defined for Sobolev space "
                             f"{sobolev_space}.")

    @cached_property
    def _solver(self):
        from firedrake import (LinearVariationalSolver,
                               LinearVariationalProblem, Function, Cofunction)
        rhs = Cofunction(self._function_space.dual())
        soln = Function(self._function_space)
        lvp = LinearVariationalProblem(
            self._inner_product, rhs, soln, bcs=self._bcs, restrict=True,
            form_compiler_parameters=self._form_compiler_parameters)
        solver = LinearVariationalSolver(
            lvp, solver_parameters=self._solver_parameters
        )
        return solver.solve, rhs, soln

    def __call__(self, value):
        """Return the Riesz representer of a Function or Cofunction."""
        from firedrake import Function, Cofunction

        if ufl.duals.is_dual(value):
            if value.function_space().dual() != self._function_space:
                raise ValueError("Function space mismatch in RieszMap.")
            output = Function(self._function_space)

            if self._inner_product == "l2":
                for o, c in zip(output.subfunctions, value.subfunctions):
                    o.dat.data[:] = c.dat.data[:]
            else:
                solve, rhs, soln = self._solver
                rhs.assign(value)
                solve()
                output = Function(self._function_space)
                output.assign(soln)
        elif ufl.duals.is_primal(value):
            if value.function_space() != self._function_space:
                raise ValueError("Function space mismatch in RieszMap.")

            if self._inner_product == "l2":
                output = Cofunction(self._function_space.dual())
                for o, c in zip(output.subfunctions, value.subfunctions):
                    o.dat.data[:] = c.dat.data[:]
            else:
                output = firedrake.assemble(
                    firedrake.action(self._inner_product, value)
                )
        else:
            raise ValueError(
                f"Unable to ascertain if {value} is primal or dual."
            )
        return output

from functools import cached_property
from contextlib import contextmanager

import ufl
import pyop3 as op3
from firedrake.petsc import PETSc
from firedrake.ensemble.ensemble_functionspace import (
    EnsembleFunctionSpaceBase, EnsembleFunctionSpace, EnsembleDualSpace)
from firedrake.adjoint_utils import EnsembleFunctionMixin
from firedrake.function import Function
from firedrake.functionspaceimpl import is_mixed
from firedrake.norms import norm


class EnsembleFunctionBase(EnsembleFunctionMixin):
    """
    A mixed (co)function defined on a :class:`~.ensemble.Ensemble`.
    The subcomponents are distributed over the ensemble members, and
    are specified locally in an
    :class:`~.ensemble_functionspace.EnsembleFunctionSpace`.

    Parameters
    ----------

    function_space :
        The function space of the (co)function.

    Notes
    -----
    Passing an :class:`~.ensemble_functionspace.EnsembleDualSpace` to
    :class:`EnsembleFunction` will return an instance of :class:`EnsembleCofunction`.

    This class does not carry UFL symbolic information, unlike a
    :class:`~firedrake.function.Function`. UFL expressions can only be defined
    locally on each ensemble member using a `~firedrake.function.Function`
    from ``EnsembleFunction.subfunctions``.

    See Also
    --------
    .ensemble_functionspace.EnsembleFunctionSpace
    .ensemble_functionspace.EnsembleDualSpace
    EnsembleFunction
    EnsembleCofunction
    """

    @PETSc.Log.EventDecorator()
    @EnsembleFunctionMixin._ad_annotate_init
    def __init__(self, function_space: EnsembleFunctionSpaceBase):
        self._fs = function_space

        # we hold all subcomponents on the local
        # ensemble member in one big mixed function.
        self._full_local_function = Function(function_space._full_local_space)

        # create a Vec containing the data for all subcomponents on all
        # ensemble members. Because we use the Vec of each local mixed
        # function as the storage, if the data in the Function Vec
        # is valid then the data in the EnsembleFunction Vec is valid.

        with self._full_local_function.dat.vec_rw as fvec:
            n = function_space.nlocal_rank_dofs
            N = function_space.nglobal_dofs
            sizes = (n, N)
            self._vec = PETSc.Vec().createWithArray(
                fvec.array, size=sizes,
                comm=function_space.global_comm)
            self._vec.setFromOptions()

    def function_space(self):
        return self._fs

    @cached_property
    def subfunctions(self):
        """Return the (co)functions on the local ensemble member.

        Each (co)function is defined on its corresponding local space and
        shares data with the ensemble (co)function.
        """
        field_axis = self._fs._full_local_space.field_axis
        dat = self._full_local_function.dat
        subfunctions = []
        for i, space in enumerate(self._fs.local_spaces):
            start, end = self._fs._component_indices(i)
            storage_labels = field_axis.component_labels[start:end]
            if is_mixed(space):
                index = op3.Slice(field_axis.label, [
                    op3.AffineSliceComponent(storage_label, label=local_label)
                    for storage_label, local_label in zip(storage_labels, space.field_axis.component_labels, strict=True)
                ])
            else:
                index, = storage_labels
            subfunctions.append(Function(space, val=dat[index]))

        return tuple(subfunctions)

    @PETSc.Log.EventDecorator()
    def riesz_representation(self, riesz_map='L2', **kwargs):
        """
        Return the Riesz representation of this ``EnsembleFunction``
        or ``EnsembleCofunction``.

        The default behaviour is to delegate to a
        :class:`~firedrake.cofunction.RieszMap` for each subfunction.
        Any ``kwargs`` will be forwarded on, for example to specify
        the particular ``riesz_map``.

        A custom Riesz map can also be provided as a callable via the
        ``riesz_map`` kwarg. The callable should act on the entire
        ``EnsembleFunction`` or ``EnsembleCofunction``.

        Parameters
        ----------
        riesz_map: str or ufl.sobolevspace.SobolevSpace
            The Riesz map to use for each subfunction (e.g. ``"l2"``,
            ``"L2"``, or ``"H1"``), or a callable acting on the entire
            ``EnsembleFunction`` (or ``EnsembleCofunction``).

        kwargs
            Other arguments to be forwarded to the
            :class:`~firedrake.cofunction.RieszMap`.
        """
        if not isinstance(riesz_map, (str, ufl.sobolevspace.SobolevSpace)):
            return riesz_map(self)

        riesz = EnsembleFunction(self.function_space().dual())
        for uself, uriesz in zip(self.subfunctions, riesz.subfunctions):
            uriesz.assign(uself.riesz_representation(riesz_map=riesz_map, **kwargs))
        return riesz

    @PETSc.Log.EventDecorator()
    def assign(self, other, subsets=None):
        r"""Set the :class:`EnsembleFunction` to the value of another
        :class:`EnsembleFunction` other.

        Parameters
        ----------

        other : :class:`EnsembleFunction`
            The value to assign from.

        subsets : Collection[Optional[:class:`pyop3.Index`]]
            One subset for each local :class:`firedrake.function.Function`.
            None elements will be ignored. The values of each local function
            will only be assigned on the nodes on the corresponding subset.
        """
        if type(other) is not type(self):
            raise TypeError(
                f"Cannot assign {type(self).__name__} from {type(other).__name__}")
        for i in range(self._fs.nlocal_spaces):
            self.subfunctions[i].assign(
                other.subfunctions[i],
                subset=subsets[i] if subsets else None)
        return self

    @PETSc.Log.EventDecorator()
    def copy(self):
        """
        Return a deep copy of the :class:`EnsembleFunction`.
        """
        new = type(self)(self.function_space())
        new.assign(self)
        return new

    @PETSc.Log.EventDecorator()
    def zero(self, subsets=None):
        """
        Set values to zero.

        Parameters
        ----------

        subsets : Collection[:class:`pyop3.Index` | None]
            One subset for each local :class:`firedrake.function.Function`.
            None elements will be ignored.  The values of each local function
            will only be zeroed on the nodes on the corresponding subset.
        """
        for i in range(self._fs.nlocal_spaces):
            self.subfunctions[i].zero(
                subset=subsets[i] if subsets else None)
        return self

    @PETSc.Log.EventDecorator()
    def __iadd__(self, other):
        for us, uo in zip(self.subfunctions, other.subfunctions):
            us.assign(us + uo)
        return self

    @PETSc.Log.EventDecorator()
    def __imul__(self, other):
        if type(other) is type(self):
            for us, uo in zip(self.subfunctions, other.subfunctions):
                us.assign(us*uo)
        else:
            for us in self.subfunctions:
                us *= other
        return self

    @PETSc.Log.EventDecorator()
    def __isub__(self, other):
        self += -1*other
        return self

    @PETSc.Log.EventDecorator()
    def __add__(self, other):
        new = self.copy()
        new += other
        return new

    @PETSc.Log.EventDecorator()
    def __sub__(self, other):
        new = self.copy()
        new -= other
        return new

    @PETSc.Log.EventDecorator()
    def __mul__(self, other):
        new = self.copy()
        new *= other
        return new

    @PETSc.Log.EventDecorator()
    def __rmul__(self, other):
        new = self.copy()
        if type(other) is type(self):
            for un, uo in zip(new.subfunctions, other.subfunctions):
                un.assign(uo*un)
        else:
            for un in new.subfunctions:
                un.assign(other*un)
        return new

    @PETSc.Log.EventDecorator()
    def __neg__(self):
        return (-1)*self

    @contextmanager
    def vec(self):
        """
        Context manager for the global ``PETSc.Vec`` with
        read/write access.

        It is invalid to access the ``Vec`` outside of a context manager.
        """
        # The globally defined _vec views the _full_local_function.dat.vec.
        # The data in _full_local_function.dat.vec is only valid inside the
        # context manager, so we need to activate that context manager before
        # yielding our _vec otherwise the data will not be up to date.
        # However, because the copies in the _full_local_function.dat.vec
        # context manager are done without _vec knowing, we have to manually
        # increment the state to make sure its still in sync.
        with self._full_local_function.dat.vec_rw:
            self._vec.stateIncrease()
            yield self._vec

    @contextmanager
    def vec_ro(self):
        """
        Context manager for the global ``PETSc.Vec`` with
        read only access.

        It is invalid to access the ``Vec`` outside of a context manager.
        """
        # The globally defined _vec views the _full_local_function.dat.vec.
        # The data in _full_local_function.dat.vec is only valid inside the
        # context manager, so we need to activate that context manager before
        # yielding our _vec otherwise the data will not be up to date.
        with self._full_local_function.dat.vec_ro:
            self._vec.stateIncrease()
            yield self._vec

    @contextmanager
    def vec_wo(self):
        """
        Context manager for the global ``PETSc.Vec`` with
        write only access.

        It is invalid to access the ``Vec`` outside of a context manager.
        """
        # The globally defined _vec views the _full_local_function.dat.vec.
        # The data in _full_local_function.dat.vec is only valid inside the
        # context manager, so we need to activate that context manager before
        # yielding our _vec otherwise the data will not be copied back into
        # the _full_local_function properly when exiting the context manager.
        # Because the _full_local_function.dat.vec_wo context manager doesn't
        # copy any data on entry, this time we don't have to manually increase
        # _vec's state. If the user modifies _vec inside out context manager then
        # _vec will know and will handle incrementing it's state itself.
        with self._full_local_function.dat.vec_wo:
            yield self._vec


class EnsembleFunction(EnsembleFunctionBase):
    """
    A mixed Function defined on a :class:`~.ensemble.Ensemble`.
    The subcomponents are distributed over the ensemble members, and
    are specified locally in an :class:`~firedrake.ensemble.ensemble_functionspace.EnsembleFunctionSpace`.

    Parameters
    ----------

    function_space : :class:`~firedrake.ensemble.ensemble_functionspace.EnsembleFunctionSpace`.
        The function space of the Function.

    Notes
    -----
    Passing an :class:`~firedrake.ensemble.ensemble_functionspace.EnsembleDualSpace`
    to ``EnsembleFunction`` will return an instance of :class:`EnsembleCofunction`.

    This class does not carry UFL symbolic information, unlike a
    :class:`firedrake.function.Function`. UFL expressions can only be defined
    locally on each ensemble member using a :class:`firedrake.function.Function`
    from ``EnsembleFunction.subfunctions``.

    See Also
    --------
    :class:`~.ensemble_functionspace.EnsembleFunctionSpace`
    :class:`~.ensemble_function.EnsembleFunction`
    :class:`~.ensemble_functionspace.EnsembleDualSpace`
    :class:`~.ensemble_function.EnsembleCofunction`
    """
    def __new__(cls, function_space: EnsembleFunctionSpaceBase):
        if isinstance(function_space, EnsembleDualSpace):
            return EnsembleCofunction(function_space)
        return super().__new__(cls)

    def __init__(self, function_space: EnsembleFunctionSpace):
        if not isinstance(function_space, EnsembleFunctionSpace):
            raise TypeError(
                "EnsembleFunction must be created using an EnsembleFunctionSpace")
        super().__init__(function_space)

    def norm(self, *args, **kwargs):
        """Compute the norm of the function.

        Any arguments are forwarded to :func:`~firedrake.norms.norm`.
        """
        return self._fs.ensemble_comm.allreduce(
            sum(norm(u, *args, **kwargs) for u in self.subfunctions))


class EnsembleCofunction(EnsembleFunctionBase):
    """
    A mixed finite element Cofunction distributed over an ensemble.

    Parameters
    ----------

    function_space : :class:`~firedrake.ensemble.ensemble_functionspace.EnsembleDualSpace`
        The function space of the cofunction.
    """
    """
    A mixed Cofunction defined on a :class:`~firedrake.ensemble.ensemble.Ensemble`.
    The subcomponents are distributed over the ensemble members,
    and are specified locally in a
    :class:`~firedrake.ensemble.ensemble_functionspace.EnsembleDualSpace`.

    Parameters
    ----------

    function_space : `~firedrake.ensemble.ensemble_functionspace.EnsembleDualSpace`.
        The dual function space of the Cofunction.

    Notes
    -----
    This class does not carry UFL symbolic information, unlike a
    :class:`~firedrake.cofunction.Cofunction`. UFL expressions can only be defined
    locally on each ensemble member using a `~firedrake.cofunction.Cofunction`
    from :meth:`EnsembleCofunction.subfunctions`.

    See Also
    --------
    :class:`~.ensemble_functionspace.EnsembleFunctionSpace`
    :class:`~.ensemble_function.EnsembleFunction`
    :class:`~.ensemble_functionspace.EnsembleDualSpace`
    :class:`~.ensemble_function.EnsembleCofunction`
    """
    def __init__(self, function_space: EnsembleDualSpace):
        if not isinstance(function_space, EnsembleDualSpace):
            raise TypeError(
                "EnsembleCofunction must be created using an EnsembleDualSpace")
        super().__init__(function_space)

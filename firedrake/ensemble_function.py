from firedrake.petsc import PETSc
from firedrake.adjoint_utils import EnsembleFunctionMixin
from firedrake.functionspace import MixedFunctionSpace
from firedrake.function import Function
from ufl.duals import is_primal, is_dual
from pyop2 import MixedDat

from functools import cached_property
from contextlib import contextmanager

__all__ = ("EnsembleFunction", "EnsembleCofunction")


class EnsembleFunctionBase(EnsembleFunctionMixin):
    """
    A mixed finite element (co)function distributed over an ensemble.

    Parameters
    ----------

    ensemble
        The ensemble communicator. The sub(co)functions are distributed
        over the different ensemble members.

    function_spaces
        A list of function spaces for each (co)function on the
        local ensemble member.
    """

    @PETSc.Log.EventDecorator()
    @EnsembleFunctionMixin._ad_annotate_init
    def __init__(self, ensemble, function_spaces):
        self.ensemble = ensemble
        self.local_function_spaces = function_spaces
        self.local_size = len(function_spaces)

        # the local functions are stored as a big mixed space
        self._function_space = MixedFunctionSpace(function_spaces)
        self._fbuf = Function(self._function_space)

        # create a Vec containing the data for all functions on all
        # ensemble members. Because we use the Vec of each local mixed
        # function as the storage, if the data in the Function Vec
        # is valid then the data in the EnsembleFunction Vec is valid.

        with self._fbuf.dat.vec as fvec:
            local_size = self._function_space.node_set.size
            sizes = (local_size, PETSc.DETERMINE)
            self._vec = PETSc.Vec().createWithArray(fvec.array,
                                                    size=sizes,
                                                    comm=ensemble.global_comm)
            self._vec.setFromOptions()

    @cached_property
    def subfunctions(self):
        """
        The (co)functions on the local ensemble member
        """
        def local_function(i):
            V = self.local_function_spaces[i]
            usubs = self._subcomponents(i)
            if len(usubs) == 1:
                dat = usubs[0].dat
            else:
                dat = MixedDat((u.dat for u in usubs))
            return Function(V, val=dat)

        self._subfunctions = tuple(local_function(i)
                                   for i in range(self.local_size))
        return self._subfunctions

    def _subcomponents(self, i):
        """
        Return the subfunctions of the local mixed function storage
        corresponding to the i-th local function.
        """
        return tuple(self._fbuf.subfunctions[j]
                     for j in self._component_indices(i))

    def _component_indices(self, i):
        """
        Return the indices into the local mixed function storage
        corresponding to the i-th local function.
        """
        V = self.local_function_spaces[i]
        offset = sum(len(V) for V in self.local_function_spaces[:i])
        return tuple(offset + i for i in range(len(V)))

    @PETSc.Log.EventDecorator()
    def riesz_representation(self, riesz_map="L2", **kwargs):
        """
        Return the Riesz representation of this :class:`EnsembleFunction`
        with respect to the given Riesz map.

        Parameters
        ----------

        riesz_map
            The Riesz map to use (`l2`, `L2`, or `H1`). This can also be a callable.

        kwargs
            other arguments to be passed to the firedrake.riesz_map.
        """
        DualType = {
            EnsembleFunction: EnsembleCofunction,
            EnsembleCofunction: EnsembleFunction,
        }[type(self)]
        Vdual = [V.dual() for V in self.local_function_spaces]
        riesz = DualType(self.ensemble, Vdual)
        for uself, uriesz in zip(self.subfunctions, riesz.subfunctions):
            uriesz.assign(uself.riesz_representation(riesz_map=riesz_map, **kwargs))
        return riesz

    @PETSc.Log.EventDecorator()
    def assign(self, other, subsets=None):
        r"""Set the :class:`EnsembleFunction` to the value of another
        :class:`EnsembleFunction` other.

        Parameters
        ----------

        other
            The :class:`EnsembleFunction` to assign from.

        subsets
            An iterable of :class:`pyop2.types.set.Subset`, one for each local :class:`Function`.
            The values of each local function will then only
            be assigned on the nodes on the corresponding subset.
        """
        if type(other) is not type(self):
            raise ValueError(
                f"Cannot assign {type(self)} from {type(other)}")
        if subsets:
            for i in range(self.local_size):
                self.subfunctions[i].assign(
                    other.subfunctions[i], subset=subsets[i])
        else:
            for i in range(self.local_size):
                self.subfunctions[i].assign(other.subfunctions[i])
        return self

    @PETSc.Log.EventDecorator()
    def copy(self):
        """
        Return a deep copy of the :class:`EnsembleFunction`.
        """
        new = type(self)(self.ensemble, self.local_function_spaces)
        new.assign(self)
        return new

    @PETSc.Log.EventDecorator()
    def zero(self, subsets=None):
        """
        Set values to zero.

        Parameters
        ----------

        subsets
            An iterable of :class:`pyop2.types.set.Subset`, one for each local :class:`Function`.
            The values of each local function will then only
            be assigned on the nodes on the corresponding subset.
        """
        if subsets:
            for i in range(self.local_size):
                self.subfunctions[i].zero(subsets[i])
        else:
            for u in self.subfunctions:
                u.zero()
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
    def __add__(self, other):
        new = self.copy()
        for i in range(self.local_size):
            new.subfunctions[i] += other.subfunctions[i]
        return new

    @PETSc.Log.EventDecorator()
    def __mul__(self, other):
        new = self.copy()
        if type(other) is type(self):
            for i in range(self.local_size):
                self.subfunctions[i].assign(other.subfunctions[i]*self.subfunctions[i])
        else:
            for i in range(self.local_size):
                self.subfunctions[i].assign(other*self.subfunctions[i])
        return new

    @PETSc.Log.EventDecorator()
    def __rmul__(self, other):
        return self.__mul__(other)

    @contextmanager
    def vec(self):
        """
        Context manager for the global PETSc Vec with read/write access.

        It is invalid to access the Vec outside of a context manager.
        """
        # _fbuf.vec shares the same storage as _vec, so we need this
        # nested context manager so that the data gets copied to/from
        # the Function.dat storage and _vec.
        # However, this copy is done without _vec knowing, so we have
        # to manually increment the state.
        with self._fbuf.dat.vec:
            self._vec.stateIncrease()
            yield self._vec

    @contextmanager
    def vec_ro(self):
        """
        Context manager for the global PETSc Vec with read only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # _fbuf.vec shares the same storage as _vec, so we need this
        # nested context manager to make sure that the data gets copied
        # to the Function.dat storage and _vec.
        with self._fbuf.dat.vec_ro:
            self._vec.stateIncrease()
            yield self._vec

    @contextmanager
    def vec_wo(self):
        """
        Context manager for the global PETSc Vec with write only access.

        It is invalid to access the Vec outside of a context manager.
        """
        # _fbuf.vec shares the same storage as _vec, so we need this
        # nested context manager to make sure that the data gets copied
        # from the Function.dat storage and _vec.
        with self._fbuf.dat.vec_wo:
            yield self._vec


class EnsembleFunction(EnsembleFunctionBase):
    """
    A mixed finite element Function distributed over an ensemble.

    Parameters
    ----------

    ensemble
        The ensemble communicator. The subfunctions are distributed
        over the different ensemble members.

    function_spaces
        A list of function spaces for each function on the
        local ensemble member.
    """
    def __init__(self, ensemble, function_spaces):
        if not all(is_primal(V) for V in function_spaces):
            raise TypeError(
                "EnsembleFunction must be created using primal FunctionSpaces")
        super().__init__(ensemble, function_spaces)


class EnsembleCofunction(EnsembleFunctionBase):
    """
    A mixed finite element Cofunction distributed over an ensemble.

    Parameters
    ----------

    ensemble
        The ensemble communicator. The subcofunctions are distributed
        over the different ensemble members.

    function_spaces
        A list of dual function spaces for each cofunction on the
        local ensemble member.
    """
    def __init__(self, ensemble, function_spaces):
        if not all(is_dual(V) for V in function_spaces):
            raise TypeError(
                "EnsembleCofunction must be created using dual FunctionSpaces")
        super().__init__(ensemble, function_spaces)

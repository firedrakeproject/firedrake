from firedrake.petsc import PETSc
from firedrake.ensemble.ensemble import Ensemble
from firedrake.functionspace import MixedFunctionSpace
from ufl.duals import is_primal, is_dual
from pyop2.mpi import internal_comm, MPI
from functools import cached_property
from itertools import chain
from typing import Collection

__all__ = ("EnsembleFunctionSpace", "EnsembleDualSpace")


def _is_primal_or_dual(local_spaces, ensemble):
    """
    We need to test primal or dual collectively over all ensemble
    ranks otherwise some may fail when others pass.

    Parameters
    ----------

    local_spaces : Collection
        The list of :function:`firedrake.FunctionSpace` on the local ensemble.comm.
    ensemble : :class:`firedrake.Ensemble`
        The communicator to test collectively over.

    Returns
    -------
    str : Description of the collective space type:
        'primal' if all spaces on all ranks are primal.
        'dual' if all spaces on all ranks are dual.
        'invalid' if any "spaces" are not primal or dual, or if some are primal or some are dual.

    Raises
    ------
    ValueError : If local_spaces are not all defined on ensemble.comm
    """
    local_comm_valid = all(
        V.mesh().comm is ensemble.comm for V in local_spaces)
    global_comm_valid = all(
        ensemble.ensemble_comm.allgather(local_comm_valid))
    if not global_comm_valid:
        raise ValueError(
            "All spaces must be defined on the ensemble.comm")

    local_types = []
    for V in local_spaces:
        if is_primal(V):
            local_types.append('primal')
            continue
        elif is_dual(V):
            local_types.append('dual')
            continue
        local_types.append('invalid')
    global_types = ensemble.ensemble_comm.allgather(local_types)
    global_types = set(chain(*global_types))
    if len(global_types) != 1:
        return 'invalid'
    else:
        return global_types.pop()


class EnsembleFunctionSpaceBase:
    """
    Base class for mixed function spaces defined on an :class:`firedrake.Ensemble`.
    The subcomponents are distributed over the ensemble members, and
    are specified locally.

    Parameters
    ----------
    local_spaces : Collection
        The list of function spaces on the local ensemble.comm.
    ensemble : `firedrake.Ensemble`
        The communicator that the function space is defined over.

    Notes
    -----
    Passing a list of dual local_spaces to :class:`firedrake.EnsembleFunctionSpace`
    will return an instance of :class:`firedrake.EnsembleDualSpace`.

    This class does not carry UFL symbolic information, unlike a
    :function:`firedrake.FunctionSpace`. UFL expressions can only be
    defined locally on each ensemble member using a :function:`firedrake.FunctionSpace` from `EnsembleFunctionSpace.local_spaces`.

    See also:
    - Primal ensemble objects: :class:`firedrake.EnsembleFunctionSpace` and :class:`firedrake.EnsembleFunction`.
    - Dual ensemble objects: :class:`firedrake.EnsembleDualSpace` and :class:`firedrake.EnsembleCofunction`.
    """
    def __init__(self, local_spaces: Collection, ensemble: Ensemble):
        meshes = set(V.mesh() for V in local_spaces)
        nlocal_meshes = len(meshes)
        max_local_meshes = ensemble.ensemble_comm.allreduce(nlocal_meshes, MPI.MAX)
        if max_local_meshes > 1:
            raise ValueError(
                f"{self.__class__.__name__} local_spaces must all be defined on the same mesh.")
        self._mesh = meshes.pop()
        self._ensemble = ensemble
        self._local_spaces = tuple(local_spaces)

        # Internally we'll store everything in a big mixed space. For
        # EnsembleFunctions/Cofunctions, we'll create (possibly mixed)
        # subfunctions that view the correct subfunctions of this big space.
        self._full_local_space = MixedFunctionSpace(self.local_spaces)

        # ensemble._comm is congruent with ensemble.global_comm not ensemble.comm
        # because obj._comm is used for garbage collection, so it needs to be the
        # communicator that the ensemble objects are collective over.
        self._comm = internal_comm(ensemble._comm, self)

    @property
    def ensemble(self):
        """The :class:`firedrake.Ensemble` that the function space is defined over
        """
        return self._ensemble

    @property
    def comm(self):
        """The spatial communicator from the :class:`firedrake.Ensemble` communicator.
        """
        return self._ensemble.comm

    @property
    def ensemble_comm(self):
        """The ensemble communicator from the :class:`firedrake.Ensemble` communicator.
        """
        return self._ensemble.ensemble_comm

    @property
    def global_comm(self):
        """The global communicator from the :class:`firedrake.Ensemble` communicator.
        """
        return self._ensemble.global_comm

    @property
    def local_spaces(self):
        """The :function:`firedrake.FunctionSpace` on the local ensemble.comm.
        """
        return self._local_spaces

    def mesh(self):
        """The :class:`firedrake.Mesh` on the local ensemble.comm.
        """
        return self._mesh

    def dual(self):
        """The dual to this function space.
        A :class:`firedrake.EnsembleDualSpace` if self is a :class:`firedrake.EnsembleFunctionSpace`, and vice-versa.
        """
        return EnsembleFunctionSpace(
            [V.dual() for V in self.local_spaces], self.ensemble)

    @cached_property
    def nlocal_spaces(self):
        """Number of subspaces on this ensemble rank.
        """
        return len(self.local_spaces)

    @cached_property
    def nglobal_spaces(self):
        """The total number of subspaces across all ensemble ranks.
        """
        return self.ensemble_comm.allreduce(self.nlocal_spaces)

    @cached_property
    def nlocal_rank_dofs(self):
        """The total number of dofs across all subspaces on the local MPI rank.
        """
        return self._full_local_space.dof_dset.layout_vec.getLocalSize()

    @cached_property
    def nlocal_comm_dofs(self):
        """The total number of dofs across all subspaces on the local ensemble.comm.
        """
        return self._full_local_space.dof_dset.layout_vec.getSize()

    @cached_property
    def nglobal_dofs(self):
        """The total number of dofs across all subspaces on all ensemble ranks.
        """
        return self.ensemble_comm.allreduce(self.nlocal_comm_dofs)

    def _component_indices(self, i):
        """
        Return the indices into the local mixed function storage
        corresponding to the i-th local function space.
        """
        offset = sum(len(V) for V in self.local_spaces[:i])
        return tuple(offset + j for j in range(len(self.local_spaces[i])))

    def create_vec(self):
        """Return a PETSc Vec on the `ensemble.global_comm` with the same layout as a
        `firedrake.EnsembleFunction` or `firedrake.EnsembleCofunction` in this function space.
        """
        vec = PETSc.Vec().create(comm=self.global_comm)
        vec.setSizes((self.nlocal_rank_dofs, self.nglobal_dofs))
        vec.setUp()
        return vec

    @cached_property
    def layout_vec(self):
        return self.create_vec()

    def local_to_global_ises(self, i):
        # COMM_SELF or global_comm?
        pass

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            local_eq = False
        elif other.ensemble is not self.ensemble:
            # TODO: should we relax this to allow congruent ensembles?
            local_eq = False
        elif self.nlocal_spaces != other.nlocal_spaces:
            local_eq = False
        else:
            local_eq = all(
                lspace == rspace
                for lspace, rspace in zip(self.local_spaces,
                                          other.local_spaces)
            )
        return all(self.ensemble.ensemble_comm.allgather(local_eq))

    def __neq__(self, other):
        return not self == other


class EnsembleFunctionSpace(EnsembleFunctionSpaceBase):
    """
    A mixed primal function space defined on an :class:`firedrake.Ensemble`.
    The subcomponents are distributed over the ensemble members, but
    are specified locally on each ensemble member.

    Parameters
    ----------
    local_spaces : Collection
        The list of primal function spaces on the local ensemble.comm.
    ensemble : `firedrake.Ensemble`
        The communicator that the function space is defined over.

    Notes
    -----
    Passing a list of dual local_spaces to :class:`firedrake.EnsembleFunctionSpace`
    will return an instance of :class:`firedrake.EnsembleDualSpace`.

    This class does not carry UFL symbolic information, unlike a
    :function:`firedrake.FunctionSpace`. UFL expressions can only be
    defined locally on each ensemble member using a :function:`firedrake.FunctionSpace` from `EnsembleFunctionSpace.local_spaces`.

    See Also
    --------
    :class:`firedrake.EnsembleFunctionSpace`
    :class:`firedrake.EnsembleFunction`
    :class:`firedrake.EnsembleDualSpace`
    :class:`firedrake.EnsembleCofunction`
    """
    def __new__(cls, local_spaces: Collection, ensemble: Ensemble):
        # Should be collective
        space_type = _is_primal_or_dual(local_spaces, ensemble)
        if space_type == 'primal':
            return super().__new__(cls)
        elif space_type == 'dual':
            return EnsembleDualSpace(local_spaces, ensemble)
        else:
            raise ValueError(
                "All local_spaces must be either primal or dual")

    def __init__(self, local_spaces: Collection, ensemble: Ensemble):
        space_type = _is_primal_or_dual(local_spaces, ensemble)
        if space_type != 'primal':
            raise ValueError(
                "EnsembleFunctionSpace can only be constructed from primal FunctionSpaces")
        super().__init__(local_spaces, ensemble)


class EnsembleDualSpace(EnsembleFunctionSpaceBase):
    """
    A mixed dual function space defined on an :class:`firedrake.Ensemble`.
    The subcomponents are distributed over the ensemble members, but
    are specified locally on each ensemble member.

    Parameters
    ----------
    local_spaces : Collection
        The list of dual function spaces on the local ensemble.comm.
    ensemble : `firedrake.Ensemble`
        The communicator that the function space is defined over.

    Notes
    -----
    This class does not carry UFL symbolic information, unlike a
    :function:`firedrake.FiredrakeDualSpace`. UFL expressions can only be
    defined locally on each ensemble member using a :function:`firedrake.FiredrakeDualSpace`
    from `EnsembleDualSpace.local_spaces`.

    See Also
    --------
    :class:`firedrake.EnsembleFunctionSpace`
    :class:`firedrake.EnsembleFunction`
    :class:`firedrake.EnsembleDualSpace`
    :class:`firedrake.EnsembleCofunction`
    """
    def __init__(self, local_spaces: Collection, ensemble: Ensemble):
        space_type = _is_primal_or_dual(local_spaces, ensemble)
        if space_type != 'dual':
            raise ValueError(
                "EnsembleDualSpace can only be constructed from dual FunctionSpaces")
        super().__init__(local_spaces, ensemble)

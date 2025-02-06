from firedrake.petsc import PETSc
from firedrake.functionspace import MixedFunctionSpace
from ufl.duals import is_primal, is_dual
from pyop2.mpi import internal_comm
from functools import cached_property
from itertools import chain


def _is_primal_or_dual(local_spaces, ensemble):
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
    def __init__(self, local_spaces, ensemble):
        meshes = set(V.mesh() for V in local_spaces)
        if len(meshes) != 1:
            raise ValueError(
                f"{self.__class__.__name__} local_spaces must be defined on the mesh provided.")
        mesh = meshes.pop()
        if mesh.comm is not ensemble.comm:
            raise ValueError(
                f"{self.__class__.name__} mesh must be defined on the ensemble.comm provided.")

        self._mesh = mesh
        self._ensemble = ensemble
        self._local_spaces = tuple(local_spaces)

        self._full_local_space = MixedFunctionSpace(self.local_spaces)

        # ensemble._comm is congruent with ensemble.global_comm not ensemble.comm!
        # This is because _comm is used for garbage collection so it needs to be
        # the communicator that the ensemble objects are collective over.
        self._comm = internal_comm(ensemble._comm, self)

    @property
    def ensemble(self):
        return self._ensemble

    @property
    def comm(self):
        return self._ensemble.comm

    @property
    def ensemble_comm(self):
        return self._ensemble.ensemble_comm

    @property
    def global_comm(self):
        return self._ensemble.global_comm
    
    @property
    def local_spaces(self):
        return self._local_spaces

    def mesh(self):
        return self._mesh

    def dual(self):
        return EnsembleFunctionSpace(
            [V.dual() for V in self.local_spaces], self.ensemble)

    @cached_property
    def nglobal_spaces(self):
        return self.ensemble_comm.allreduce(len(self.local_spaces))

    @cached_property
    def nlocal_dofs(self):
        return self._full_local_space.node_set.size

    @cached_property
    def nglobal_dofs(self):
        return self.ensemble_comm.allreduce(self.nlocal_dofs)

    def _component_indices(self, i):
        pass

    def make_vec(self):
        pass

    def local_to_global_ises(self, i):
        # COMM_SELF or global_comm?
        pass

    def __eq__(self, other):
        locally_equal = all(
            lspace == rspace
            for lspace, rspace in zip(self.local_spaces, other.local_spaces)
        )
        print(f'{locally_equal = }')
        all_equal = self.ensemble.ensemble_comm.allgather(locally_equal)
        print(f'{all_equal = }')
        return all(all_equal)

    def __neq__(self, other):
        return not self.__eq__(other)


class EnsembleFunctionSpace(EnsembleFunctionSpaceBase):
    def __new__(cls, local_spaces, ensemble):
        # Should be collective
        space_type = _is_primal_or_dual(local_spaces, ensemble)
        if space_type == 'primal':
            return super().__new__(cls)
        elif space_type == 'dual':
            return EnsembleDualSpace(local_spaces, ensemble)
        else:
            raise ValueError(
                "All local_spaces must be either primal or dual")

    def __init__(self, local_spaces, ensemble):
        space_type = _is_primal_or_dual(local_spaces, ensemble)
        if space_type != 'primal':
            raise ValueError(
                "EnsembleFunctionSpace can only be constructed from primal FunctionSpaces")
        super().__init__(local_spaces, ensemble)


class EnsembleDualSpace(EnsembleFunctionSpaceBase):
    def __init__(self, local_spaces, ensemble):
        space_type = _is_primal_or_dual(local_spaces, ensemble)
        if space_type != 'dual':
            raise ValueError(
                "EnsembleDualSpace can only be constructed from dual FunctionSpaces")
        super().__init__(local_spaces, ensemble)

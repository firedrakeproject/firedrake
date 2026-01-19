"""
This module contains the AdaptiveTransferManager used to perform
transfer operations on AdaptiveMeshHierarchies
"""
import numpy as np
from firedrake.function import Function
from firedrake.mg.embedded import TransferManager
from firedrake.mg.utils import get_level
from firedrake.petsc import PETSc


__all__ = ("AdaptiveTransferManager",)


class AdaptiveTransferManager(TransferManager):
    """
    TransferManager for adaptively refined mesh hierarchies
    """
    def __init__(self, *, native_transfers=None, use_averaging=True):
        super().__init__(native_transfers=native_transfers, use_averaging=use_averaging)
        self.tm = TransferManager()
        self.weight_cache = {}
        self.work_function_cache = {}
        self.perm_cache = {}

    def generic_transfer(self, source, target, transfer_op):
        """
        Generalized implementation of transfer operations by wrapping
        transfer operations from TransferManager()
        """
        amh, source_level = get_level(source.function_space().mesh())
        _, target_level = get_level(target.function_space().mesh())

        # decide order of iteration depending on coarse -> fine or fine -> coarse
        order = 1
        if target_level < source_level:
            order = -1

        curr_source = source
        if source_level == target_level:
            target.assign(source)
            return

        for level in range(source_level, target_level, order):
            if level + order == target_level:
                curr_target = target
            else:
                target_mesh = amh.meshes[level + order]
                curr_space = curr_source.function_space()
                target_space = curr_space.reconstruct(mesh=target_mesh)
                curr_target = self.get_work_function(target_space)

            if transfer_op == self.tm.restrict:
                w = self.get_weight(curr_source.function_space())
                wsource = self.get_work_function(curr_source.function_space())
                with (
                    curr_source.dat.vec as svec,
                    w.dat.vec as wvec,
                    wsource.dat.vec as wsvec,
                ):
                    wsvec.pointwiseMult(svec, wvec)
                curr_source = wsource

            if order == 1:
                source_function_splits = amh.split_function(curr_source, child=False)
                target_function_splits = amh.split_function(curr_target, child=True)
            else:
                source_function_splits = amh.split_function(curr_source, child=True)
                target_function_splits = amh.split_function(curr_target, child=False)

            for split_label in source_function_splits:
                if split_label == 1:
                    # we don't want to transfer across unsplit parts,
                    # instead we copy dofs
                    us_func = source_function_splits[1]
                    ut_func = target_function_splits[1]
                    permutations = self.get_perm(
                        us_func,
                        ut_func,
                        transfer_op
                    )
                    ut_func.dat.data_wo[permutations] = us_func.dat.data_ro
                else:
                    transfer_op(
                        source_function_splits[split_label],
                        target_function_splits[split_label],
                    )

            amh.recombine(target_function_splits, curr_target, child=order + 1)
            curr_source = curr_target

    def get_work_function(self, func_space):
        """
        Cache for function on function space
        """
        try:
            return self.work_function_cache[func_space]
        except KeyError:
            return self.work_function_cache.setdefault(func_space, Function(func_space))

    def get_weight(self, V_source):
        """
        Cache for weights from partition of unity used during restriction
        """
        try:
            return self.weight_cache[V_source]
        except KeyError:
            amh, _ = get_level(V_source.mesh())
            return self.weight_cache.setdefault(
                V_source, amh.use_weight(V_source, child=True)
            )

    def get_perm(self, unsplit_source, unsplit_target, transfer_op):
        """
        Cache permutations of DoFs from unsplit source
        to unsplit target. This is used to skip transfer
        across unsplit mesh hierarchies
        """
        key = (unsplit_source.function_space(),
               unsplit_target.function_space())
        try:
            return self.perm_cache[key]
        except KeyError:
            source_nodes = Function(key[0])
            permutation = Function(key[1])
            source_nodes.dat.data_wo[:] = np.arange(len(source_nodes.dat.data_ro))
            transfer_op(source_nodes, permutation)

            return self.perm_cache.setdefault(
                key, np.rint(permutation.dat.data_ro).astype(PETSc.IntType)
            )

    def prolong(self, uc, uf):
        """
        Prolongation of AdaptiveMeshHierarchy
        """
        self.generic_transfer(uc, uf, transfer_op=self.tm.prolong)

    def inject(self, uf, uc):
        """
        Injection of AdaptiveMeshHierarchy
        """
        self.generic_transfer(uf, uc, transfer_op=self.tm.inject)

    def restrict(self, source, target):
        """
        Restriction of AdaptiveMeshHierarchy
        """
        self.generic_transfer(source, target, transfer_op=self.tm.restrict)

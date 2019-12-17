from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks

class Backend:
    @property
    def backend(self):
        import firedrake
        return firedrake

    @property
    def compat(self):
        import firedrake
        return compat(firedrake)

class DirichletBCBlock(blocks.DirichletBCBlock, Backend):
    pass

class ExpressionBlock(blocks.ExpressionBlock, Backend):
    pass

class ConstantAssignBlock(blocks.ConstantAssignBlock, Backend):
    pass

class FunctionEvalBlock(blocks.FunctionEvalBlock, Backend):
    pass

class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
    pass

class FunctionSplitBlock(blocks.FunctionSplitBlock, Backend):
    pass

class FunctionMergeBlock(blocks.FunctionMergeBlock, Backend):
    pass

class FunctionAssignerBlock(blocks.FunctionAssignerBlock, Backend):
    pass

class ALEMoveBlock(blocks.ALEMoveBlock, Backend):
    pass

class BoundaryMeshBlock(blocks.BoundaryMeshBlock, Backend):
    pass

class AssembleBlock(blocks.AssembleBlock, Backend):
    pass

class SolveBlock(blocks.SolveBlock, Backend):
    pass

class SurfaceTransferBlock(blocks.SurfaceTransferBlock, Backend):
    pass

class VolumeTransferBlock(blocks.VolumeTransferBlock, Backend):
    pass

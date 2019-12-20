from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks
from pyadjoint.block import Block


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


class FunctionAssignBlock(blocks.FunctionAssignBlock, Backend):
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


class ProjectBlock(blocks.ProjectBlock, Backend):
    pass


class MeshInputBlock(Block):
    """
    Block which links a MeshGeometry to its coordinates, which is a firedrake
    function.
    """
    def __init__(self, mesh):
        super(MeshInputBlock, self).__init__()
        self.add_dependency(mesh)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        mesh = self.get_dependencies()[0].saved_output
        return mesh.coordinates


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh are changed.
    """
    def __init__(self, func, mesh):
        super(MeshOutputBlock, self).__init__()
        self.add_dependency(func)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, idx, block_variable,
                                   relevant_dependencies, prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        vector = self.get_dependencies()[0].saved_output
        mesh = vector.function_space().mesh()
        mesh.coordinates.assign(vector, annotate=False)
        return mesh._ad_create_checkpoint()


class NonlinearVariationalSolveBlock(blocks.NonlinearVariationalSolveBlock, Backend):
    pass

from dolfin_adjoint_common.compat import compat
from dolfin_adjoint_common import blocks
from pyadjoint.block import Block

import firedrake.utils as utils


class Backend:
    @utils.cached_property
    def backend(self):
        import firedrake
        return firedrake

    @utils.cached_property
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


class FunctionSplitBlock(blocks.FunctionSplitBlock, Backend):
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
        super().__init__()
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
        super().__init__()
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


class PointwiseOperatorBlock(Block, Backend):
    def __init__(self, point_op, *args, **kwargs):
        super(PointwiseOperatorBlock, self).__init__()
        self.point_op = point_op
        self.add_dependency(self.point_op, no_duplicates=True)
        for c in self.point_op.ufl_operands:
            self.add_dependency(c, no_duplicates=True)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        N, ops = inputs[0], inputs[1:]
        return N._ufl_expr_reconstruct_(*ops)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        print('PointopBlock eval_adj_comp')
        if self.point_op == block_variable.output:
            # We are not able to calculate derivatives wrt initial guess.
            #self.point_op_rep = block_variable.saved_output
            return None

        q_rep = block_variable.saved_output
        N = prepared

        i_ops = list(i for i, e in enumerate(N.ufl_operands) if e == q_rep)[0] 
        dNdm_adj = N.adjoint_action(adj_inputs[0], i_ops)
        #dNdm_adj = self.compat.assemble_adjoint_value(dNdm_adj)
        import ipdb; ipdb.set_trace()
        return dNdm_adj

    def recompute_component(self, inputs, block_variable, idx, prepared):
        print('PointopBlock recompute_comp')
        p, ops = inputs[0], inputs[1:]
        q = type(p).copy(p)
        return q.evaluate()
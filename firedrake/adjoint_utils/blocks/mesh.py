from pyadjoint import Block
from firedrake.adjoint_utils.checkpointing import maybe_disk_checkpoint


class MeshInputBlock(Block):
    """
    Block which links a MeshGeometry to its coordinates, which is a firedrake
    function.
    """
    def __init__(self, mesh, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(mesh)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   idx, block_variable, relevant_dependencies,
                                   prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        mesh = self.get_dependencies()[0].saved_output
        return maybe_disk_checkpoint(mesh.coordinates)


class MeshOutputBlock(Block):
    """
    Block which is called when the coordinates of a mesh are changed.
    """
    def __init__(self, func, mesh, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(func)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        return adj_inputs[0]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        return tlm_inputs[0]

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   idx, block_variable, relevant_dependencies,
                                   prepared=None):
        return hessian_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        vector = self.get_dependencies()[0].saved_output
        mesh = vector.function_space().mesh()
        mesh.coordinates.assign(vector, annotate=False)
        return mesh._ad_create_checkpoint()

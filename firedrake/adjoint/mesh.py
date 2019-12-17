from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.tape import no_annotations


class MeshGeometryMixin(OverloadedType):
    @staticmethod
    def _ad_annotate_init(init):
        def wrapper(self, *args, **kwargs):
            self.backend.mesh.MeshGeometry.__init__(self, *args, **kwargs)
            init(self, *args, **kwargs)
        return wrapper

    @no_annotations
    def _ad_create_checkpoint(self):
        return self.coordinates.copy(deepcopy=True)

    @no_annotations
    def _ad_restore_at_checkpoint(self, checkpoint):
        self.coordinates.assign(checkpoint)
        return self

    @staticmethod
    def _ad_annotate_coordinates_function(coordinates_function):
        def wrapper(self, *args, **kwargs):
            from firedrake_adjoint.types.mesh import MeshInputBlock, MeshOutputBlock
            f = coordinates_function()
            f.block_class = MeshInputBlock
            f._ad_floating_active = True
            f._ad_args = [self]

            f._ad_output_args = [self]
            f.output_block_class = MeshOutputBlock
            f._ad_outputs = [self]
            return f
        return wrapper

from functools import wraps
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.tape import no_annotations


class MeshGeometryMixin(OverloadedType):
    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            OverloadedType.__init__(self, *args, **kwargs)
            init(self, *args, **kwargs)
            self._ad_coordinate_space = None
            self.coordinates._ad_type = "coordinates_function"
        return wrapper

    @no_annotations
    def _ad_create_checkpoint(self):
        result = self.coordinates.copy(deepcopy=True)
        result._ad_type = self.coordinates._ad_type
        return result
    
    def _ad_clear_checkpoint(self, checkpoint):
        if checkpoint.dat.dat_version == self.coordinates.dat.dat_version:
            return checkpoint
        else:
            return None

    @no_annotations
    def _ad_restore_at_checkpoint(self, checkpoint):
        if not self.coordinates.dat.dat_version == checkpoint.dat.dat_version:
            self.coordinates.assign(checkpoint)
        return self

    @staticmethod
    def _ad_annotate_coordinates_function(coordinates_function):
        @wraps(coordinates_function)
        def wrapper(self, *args, **kwargs):
            from .blocks import MeshInputBlock, MeshOutputBlock
            f = coordinates_function(self)
            f.block_class = MeshInputBlock
            f._ad_floating_active = True
            f._ad_args = [self]

            f._ad_output_args = [self]
            f.output_block_class = MeshOutputBlock
            f._ad_outputs = [self]
            return f
        return wrapper

    def _ad_function_space(self):
        if self._ad_coordinate_space is None:
            self._ad_coordinate_space = self.coordinates.function_space().ufl_function_space()
        return self._ad_coordinate_space

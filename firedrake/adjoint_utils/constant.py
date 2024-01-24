from functools import wraps
from pyadjoint.adjfloat import AdjFloat
from pyadjoint.tape import get_working_tape, annotate_tape
from pyadjoint.overloaded_type import OverloadedType, create_overloaded_object
from pyadjoint.reduced_functional_numpy import gather

from firedrake.functionspace import FunctionSpace
from firedrake.adjoint_utils.blocks import ConstantAssignBlock

from ufl.domain import extract_unique_domain

import numpy


class ConstantMixin(OverloadedType):

    @staticmethod
    def _ad_annotate_init(init):
        @wraps(init)
        def wrapper(self, *args, **kwargs):
            annotate = kwargs.pop("annotate", True)
            OverloadedType.__init__(self, *args,
                                    block_class=kwargs.pop("block_class", None),
                                    _ad_floating_active=kwargs.pop("_ad_floating_active", False),
                                    _ad_args=kwargs.pop("_ad_args", None),
                                    output_block_class=kwargs.pop("output_block_class", None),
                                    _ad_output_args=kwargs.pop("_ad_output_args", None),
                                    _ad_outputs=kwargs.pop("_ad_outputs", None),
                                    annotate=annotate, **kwargs)
            init(self, *args, **kwargs)

            other = args[0]
            if isinstance(other, (type(self), AdjFloat)):
                self.assign(other, annotate=annotate)
        return wrapper

    @staticmethod
    def _ad_annotate_assign(assign):
        @wraps(assign)
        def wrapper(self, *args, **kwargs):
            ad_block_tag = kwargs.pop("ad_block_tag", None)
            annotate = annotate_tape(kwargs)
            if annotate:
                other = args[0]
                if not isinstance(other, OverloadedType):
                    other = create_overloaded_object(other)

                block = ConstantAssignBlock(other, ad_block_tag=ad_block_tag)
                tape = get_working_tape()
                tape.add_block(block)

            ret = assign(self, *args, **kwargs)

            if annotate:
                block.add_output(self.create_block_variable())

            return ret

        return wrapper

    def get_derivative(self, options={}):
        return self._ad_convert_type(self.adj_value, options=options)

    def _ad_convert_type(self, value, options={}):
        if value is None:
            # TODO: Should the default be 0 constant here or return just None?
            return type(self)(numpy.zeros(self.ufl_shape))
        value = gather(value)
        return self._constant_from_values(value)

    def _ad_function_space(self, mesh):
        element = self.ufl_element()
        fs_element = element.reconstruct(cell=mesh.ufl_cell())
        return FunctionSpace(mesh, fs_element)

    def _ad_create_checkpoint(self):
        return self._constant_from_values()

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_mul(self, other):
        return self._constant_from_values(self.dat.data_ro.reshape(-1) * other)

    def _ad_add(self, other):
        return self._constant_from_values(
            self.dat.data_ro.reshape(-1) + other.dat.data_ro.reshape(-1)
        )

    def _ad_dot(self, other, options=None):
        if type(other) is AdjFloat:
            return sum(self.dat.data_ro.reshape(-1) * other)
        else:
            return sum(self.dat.data_ro.reshape(-1) * other.dat.data_ro.reshape(-1))

    @staticmethod
    def _ad_assign_numpy(dst, src, offset):
        l = dst.ufl_element().value_size
        dst.assign(numpy.reshape(src[offset:offset + l], dst.ufl_shape), annotate=False)
        offset += l
        return dst, offset

    @staticmethod
    def _ad_to_list(m):
        return m.dat.data_ro.reshape(-1).tolist()

    def _ad_copy(self):
        return self._constant_from_values()

    def _ad_dim(self):
        return self.dat.cdim

    def _ad_imul(self, other):
        self.assign(self._constant_from_values(self.dat.data_ro.reshape(-1) * other))

    def _ad_iadd(self, other):
        self.assign(self._constant_from_values(
            self.dat.data_ro.reshape(-1) + other.dat.data_ro.reshape(-1)
        ))

    def _reduce(self, r, r0):
        npdata = self.dat.data_ro.reshape(-1)
        for i in range(len(npdata)):
            r0 = r(npdata[i], r0)
        return r0

    def _applyUnary(self, f):
        npdata = self.dat.data_ro.reshape(-1)
        npdatacopy = npdata.copy()
        for i in range(len(npdata)):
            npdatacopy[i] = f(npdata[i])
        self.assign(self._constant_from_values(npdatacopy))

    def _applyBinary(self, f, y):
        npdata = self.dat.data_ro.reshape(-1)
        npdatacopy = self.dat.data_ro.reshape(-1).copy()
        npdatay = y.dat.data_ro.reshape(-1)
        for i in range(len(npdata)):
            npdatacopy[i] = f(npdata[i], npdatay[i])
        self.assign(self._constant_from_values(npdatacopy))

    def __deepcopy__(self, memodict={}):
        return self._constant_from_values()

    def _constant_from_values(self, values=None):
        """Returns a new Constant with self.dat.data_ro.reshape(-1) while preserving self.ufl_shape.

        If the optional argument `values` is provided, then `values` will be the values of the
        new Constant instead, still preserving the ufl_shape of self.

        Args:
            values (numpy.array): An optional argument to use instead of ``self.dat.data_ro.reshape(-1)``.

        Returns:
            Constant: The created Constant

        """
        values = self.dat.data_ro.reshape(-1) if values is None else values
        return type(self)(numpy.reshape(values, self.ufl_shape), domain=extract_unique_domain(self))

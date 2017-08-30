from abc import ABCMeta, abstractproperty

from coffee import base as ast

from collections import OrderedDict

from firedrake.slate.slate import (TensorBase, Tensor,
                                   TensorOp, Action, Negative)
from firedrake.slate.slac.utils import (topological_sort,
                                        traverse_dags,
                                        collect_reference_count)
from firedrake.utils import cached_property

from functools import reduce

from ufl import MixedElement


class KernelBuilderBase(object, metaclass=ABCMeta):
    """A base helper class for constructing Slate kernels."""

    def __init__(self, expression, tsfc_parameters=None):
        """Constructor for the KernelBuilderBase class.

        :arg expression: a :class:`TensorBase` object.
        :arg tsfc_parameters: an optional `dict` of parameters to provide to
                              TSFC when constructing subkernels associated
                              with the expression.
        """
        assert isinstance(expression, TensorBase)

        if expression.ufl_domain().variable_layers:
            raise NotImplementedError("Variable layers not yet handled in Slate.")

        # Collect reference counts
        ref_counts = collect_reference_count([expression])

        # Collect terminals and expressions
        temps = OrderedDict()
        action_coeffs = OrderedDict()
        tensor_ops = []
        seen_coeff = set()
        for tensor in traverse_dags([expression]):
            if isinstance(tensor, Tensor):
                temps.setdefault(tensor, ast.Symbol("T%d" % len(temps)))

            elif isinstance(tensor, TensorOp):
                # Actions will always require a coefficient temporary.
                if isinstance(tensor, Action):
                    actee, = tensor.actee
                    if actee not in seen_coeff:
                        shapes = [(V.finat_element.space_dimension(),
                                   V.value_size)
                                  for V in actee.function_space().split()]
                        shp = sum(n * d for (n, d) in shapes)
                        offset = 0
                        for i, shape in enumerate(shapes):
                            # Return a tuple containing the function space
                            # index, the offset index, the shape of the
                            # coefficient temp, and the actee.
                            cinfo = (i, offset, shp, actee)
                            action_coeffs.setdefault(shape, []).append(cinfo)
                            offset += reduce(lambda x, y: x*y, shape)

                        seen_coeff.add(actee)

                # Operations which have "high" reference count will have
                # auxiliary temporaries created. Negative and Transpose
                # operations will not have extra temporaries.
                if ref_counts[tensor] > 1 and not isinstance(tensor, Negative):
                    tensor_ops.append(tensor)

        self.expression = expression
        self.tsfc_parameters = tsfc_parameters
        self.temps = temps
        self.aux_exprs = topological_sort(tensor_ops)
        self.action_coefficients = action_coeffs

    @cached_property
    def coefficient_map(self):
        """Generates a mapping from a coefficient to its kernel argument
        symbol. If the coefficient is mixed, all of its split components
        will be returned.
        """
        coefficient_map = OrderedDict()
        for i, coefficient in enumerate(self.expression.coefficients()):
            if type(coefficient.ufl_element()) == MixedElement:
                csym_info = []
                for j, _ in enumerate(coefficient.split()):
                    csym_info.append(ast.Symbol("w_%d_%d" % (i, j)))
            else:
                csym_info = (ast.Symbol("w_%d" % i),)

            coefficient_map[coefficient] = tuple(csym_info)

        return coefficient_map

    def coefficient(self, coefficient):
        """Extracts the kernel arguments corresponding to a particular coefficient.
        This handles both the case when the coefficient is defined on a mixed
        or non-mixed function space.
        """
        return self.coefficient_map[coefficient]

    @cached_property
    def context_kernels(self):
        """Gathers all :class:`~.ContextKernel`\s containing all TSFC kernels,
        and integral type information.
        """
        from firedrake.slate.slac.tsfc_driver import compile_terminal_form

        cxt_list = [compile_terminal_form(expr, prefix="subkernel%d_" % i,
                                          tsfc_parameters=self.tsfc_parameters)
                    for i, expr in enumerate(self.temps)]

        cxt_kernels = [cxt_k for cxt_tuple in cxt_list
                       for cxt_k in cxt_tuple]
        return cxt_kernels

    @abstractproperty
    def integral_type(self):
        """Returns the integral type associated with a Slate kernel. This
        is used to determine how the Slate kernel should be iterated over
        the mesh."""

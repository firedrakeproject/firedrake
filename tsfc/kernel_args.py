import abc


class KernelArg(abc.ABC):
    """Abstract base class wrapping a loopy argument.

    Defining this type system allows Firedrake (or other libraries) to
    prepare arguments for the kernel without needing to worry about their
    ordering. Instead this can be offloaded to tools such as
    :func:`functools.singledispatch`.
    """

    def __init__(self, arg):
        self.loopy_arg = arg

    @property
    def dtype(self):
        return self.loopy_arg.dtype


class OutputKernelArg(KernelArg):
    ...


class CoordinatesKernelArg(KernelArg):
    ...


class CoefficientKernelArg(KernelArg):
    ...


class ConstantKernelArg(KernelArg):
    ...


class CellOrientationsKernelArg(KernelArg):
    ...


class CellSizesKernelArg(KernelArg):
    ...


class TabulationKernelArg(KernelArg):
    ...


class ExteriorFacetKernelArg(KernelArg):
    ...


class InteriorFacetKernelArg(KernelArg):
    ...


class ExteriorFacetOrientationKernelArg(KernelArg):
    ...


class InteriorFacetOrientationKernelArg(KernelArg):
    ...

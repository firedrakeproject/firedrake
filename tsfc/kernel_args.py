import abc

import coffee.base as coffee
import loopy as lp


class KernelArg(abc.ABC):

    def __init__(self, ast_arg):
        self._ast_arg = ast_arg

    @property
    def dtype(self):
        if self._is_coffee_backend:
            return self._ast_arg.typ
        elif self._is_loopy_backend:
            return self._ast_arg.dtype

    @property
    def coffee_arg(self):
        if not self._is_coffee_backend:
            raise RuntimeError("Invalid type requested")
        return self._ast_arg

    @property
    def loopy_arg(self):
        if not self._is_loopy_backend:
            raise RuntimeError("Invalid type requested")
        return self._ast_arg

    @property
    def _is_coffee_backend(self):
        return isinstance(self._ast_arg, coffee.Decl)

    @property
    def _is_loopy_backend(self):
        return isinstance(self._ast_arg, lp.ArrayArg)

class OutputKernelArg(KernelArg):
    ...


class CoordinatesKernelArg(KernelArg):
    ...


class CoefficientKernelArg(KernelArg):
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



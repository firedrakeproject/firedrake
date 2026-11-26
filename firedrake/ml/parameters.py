"""Symbolic handles for parameters stored in PyTorch."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Self

from mpi4py import MPI
import numpy as np
import ufl

from firedrake.cofunction import Cofunction
from firedrake.function import Function
from firedrake.realspace import RealSpace
from firedrake.utils import ScalarType, complex_mode

if TYPE_CHECKING:
    import torch


class MLParameters(ufl.Coefficient):
    """Represent live PyTorch parameters by a single coefficient.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor]
        The model tensors in parameter order. Their shapes, dtypes and devices
        must remain fixed. Construct the handle after loading and moving the model.

    Notes
    -----
    Each process owns its model replica. Operators on any mesh can refer to the
    same handle. Gradients and directions use coordinates in parameter order.
    In real scalar builds, the real and imaginary parts of each complex entry
    occupy adjacent coordinates.
    """

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        return object.__new__(cls)

    def __init__(self, parameters: Iterable[torch.Tensor]) -> None:
        import torch

        self._tensors = tuple(parameters)
        if any(not isinstance(t, torch.Tensor) for t in self._tensors):
            raise TypeError("Model parameters must be PyTorch tensors.")
        if any(not (t.is_floating_point() or t.is_complex()) for t in self._tensors):
            raise TypeError("Model parameters must have floating-point or complex values.")
        if any(torch.finfo(t.dtype).bits > np.finfo(ScalarType).bits for t in self._tensors):
            raise ValueError("Firedrake scalar precision must accommodate the model parameters.")
        self._layout = tuple((t.shape, t.dtype, t.device) for t in self._tensors)
        self._sizes = tuple(t.numel() * (2 if t.is_complex() and not complex_mode else 1) for t in self._tensors)
        size = sum(self._sizes)
        if not size:
            raise ValueError("A parameter vector must contain at least one scalar.")
        super().__init__(RealSpace(MPI.COMM_SELF, (size,)))

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the original tensors in parameter order."""
        if any((t.shape, t.dtype, t.device) != spec for t, spec in zip(self._tensors, self._layout, strict=True)):
            raise ValueError("The model parameter layout has changed. Create a new MLParameters handle.")
        return self._tensors

    def function_space(self) -> RealSpace:
        """Return the space of parameter directions."""
        return self.ufl_function_space()

    def _pack(self, blocks: Iterable[torch.Tensor], out: Function | Cofunction) -> None:
        import torch

        dest = out.dat.data_wo.reshape(-1)
        dtype = torch.from_numpy(np.empty((), dtype=ScalarType)).dtype
        start = 0
        for block, size in zip(blocks, self._sizes, strict=True):
            values = block.detach().resolve_conj().resolve_neg()
            if values.is_complex() and not complex_mode:
                values = torch.view_as_real(values)
            dest[start:start + size] = values.reshape(-1).to(device="cpu", dtype=dtype).numpy()
            start += size

    def _unpack(self, value: Function | Cofunction) -> tuple[torch.Tensor, ...]:
        import torch

        data = value.dat.data_ro.reshape(-1)
        start = 0
        blocks = []
        for t, size in zip(self.tensors, self._sizes, strict=True):
            values = data[start:start + size]
            if t.is_complex() and not complex_mode:
                values = values[0::2] + 1j * values[1::2]
            elif not t.is_complex() and np.iscomplexobj(values):
                if np.any(values.imag):
                    raise ValueError("A real model parameter received a non-real coordinate.")
                values = values.real
            blocks.append(torch.tensor(values.reshape(t.shape), device=t.device, dtype=t.dtype))
            start += size
        return tuple(blocks)

    def backward(self, gradient: Cofunction) -> None:
        """Accumulate an assembled cotangent in the live parameters' gradient buffers.

        Parameters
        ----------
        gradient : firedrake.Cofunction
            Cotangent in the dual of this handle's space.
        """
        import torch

        if not isinstance(gradient, Cofunction) or gradient.function_space() != self.function_space().dual():
            raise ValueError("Expected a gradient assembled with respect to these parameters.")
        tensors = self.tensors
        if not all(t.requires_grad for t in tensors):
            raise ValueError("Only parameters that require gradients can receive backward contributions.")
        torch.autograd.backward(tensors, self._unpack(gradient))

    def copy(self) -> Function:
        """Return a snapshot in the parameter space.

        Returns
        -------
        firedrake.Function
            Independent values from which the live model can be restored.
        """
        saved = Function(self.function_space())
        self._pack(self.tensors, saved)
        return saved

    def assign(self, value: Function) -> Self:
        """Restore the model's values in place.

        Parameters
        ----------
        value : firedrake.Function
            Values in this handle's parameter space.

        Returns
        -------
        MLParameters
            This handle.
        """
        import torch

        if not isinstance(value, Function) or value.function_space() != self.function_space():
            raise ValueError("Expected a function in this parameter space.")
        blocks = self._unpack(value)
        with torch.no_grad():
            for t, block in zip(self.tensors, blocks, strict=True):
                t.copy_(block)
        return self

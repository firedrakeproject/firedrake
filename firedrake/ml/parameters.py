"""Symbolic handles for parameters stored in PyTorch."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import ufl

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class _ParameterElement(ufl.AbstractFiniteElement):
    """Describe a parameter vector on the reference point by its shape."""

    cell = ufl.vertex
    reference_value_shape: tuple[int, ...] = field()
    sobolev_space = ufl.HInf
    pullback = ufl.identity_pullback
    embedded_subdegree = embedded_superdegree = 0
    sub_elements = ()

    def __str__(self) -> str:
        return repr(self)


class MLParameters(ufl.Coefficient):
    """Represent PyTorch tensors as a single symbolic parameter vector.

    Parameters
    ----------
    parameters : Iterable[torch.Tensor]
        Tensors to reference, normally ``model.parameters()``. Their order and
        shapes define the parameter vector and must remain fixed. All tensors
        must have floating-point or complex values.

    Notes
    -----
    Construction retains the supplied tensors and their PyTorch storage.
    UFL records their total size. Operators on different meshes can share
    the same parameter vector.
    Each MPI rank retains its own model parameters.

    Create the handle after moving the model to its intended dtype and device.
    In-place optimiser steps remain visible through the handle. If the model
    replaces its parameter objects, create a new handle.

    Examples
    --------
    >>> import torch
    >>> from firedrake.ml.pytorch import MLParameters
    >>> model = torch.nn.Linear(2, 1)
    >>> parameters = MLParameters(model.parameters())
    >>> saved_parameters = parameters.copy()
    >>> _ = parameters.assign(saved_parameters)
    """

    def __new__(cls, *args: object, **kwargs: object) -> Self:
        # A parameter block is always a primal coefficient.
        return object.__new__(cls)

    def __init__(self, parameters: Iterable[torch.Tensor]) -> None:
        import torch

        self._tensors = tuple(parameters)
        if any(not isinstance(t, torch.Tensor) for t in self._tensors):
            raise TypeError("Model parameters must be PyTorch tensors.")
        if any(not (t.is_floating_point() or t.is_complex()) for t in self._tensors):
            raise TypeError("Model parameters must have floating-point or complex values.")
        size = sum(t.numel() for t in self._tensors)
        if size == 0:
            raise ValueError("A parameter vector must contain at least one scalar.")
        element = _ParameterElement((size,))
        super().__init__(ufl.FunctionSpace(None, element))

    @property
    def tensors(self) -> tuple[torch.Tensor, ...]:
        """Return the native parameter tensors.

        Returns
        -------
        tuple[torch.Tensor]
            References to the tensors in parameter order.
        """
        return self._tensors

    def copy(self) -> "MLParameters":
        """Copy parameter values without retaining their autograd history.

        Returns
        -------
        MLParameters
            Independent tensors with the same shapes, dtypes and devices.
        """
        return type(self)(t.detach().clone() for t in self._tensors)

    def assign(self, other: "MLParameters") -> "MLParameters":
        """Copy values while preserving the original tensors and their storage.

        Parameters
        ----------
        other : MLParameters
            Values with the same tensor shapes, dtypes and devices.

        Returns
        -------
        MLParameters
            This parameter vector.

        Raises
        ------
        ValueError
            If the tensor shapes, dtypes or devices differ.
        """
        import torch

        source = other.tensors
        target = self._tensors
        if len(source) != len(target) or any(
            (s.shape, s.dtype, s.device) != (t.shape, t.dtype, t.device)
            for s, t in zip(source, target)
        ):
            raise ValueError("Parameter tensors must have matching shapes, dtypes and devices.")
        with torch.no_grad():
            for t, s in zip(target, source):
                t.copy_(s)
        return self


class MLParameterGradient(ufl.Cofunction):
    """A replicated parameter cotangent with one PyTorch tensor per weight.

    Parameters
    ----------
    parameters : MLParameters
        The coefficient with respect to which the derivative was taken.
    tensors : tuple of torch.Tensor
        The gradient blocks in the order and shapes of the parameter tensors.

    Notes
    -----
    Assembly leaves the model's gradient buffers untouched. To accumulate the
    result for a PyTorch optimizer, use
    ``torch.autograd.backward(theta.tensors, gradient.tensors)`` with the
    trainable parameters in ``theta``.
    """

    def __new__(cls, parameters: MLParameters, tensors: tuple[torch.Tensor, ...]) -> Self:
        return super().__new__(cls, parameters.ufl_function_space().dual())

    def __init__(self, parameters: MLParameters, tensors: tuple[torch.Tensor, ...]) -> None:
        super().__init__(parameters.ufl_function_space().dual())
        self.parameters = parameters
        self.tensors = tensors

    def assign(self, other: MLParameterGradient) -> Self:
        """Assign the gradient of the same parameter block.

        Parameters
        ----------
        other : MLParameterGradient
            Source of the gradient values.

        Returns
        -------
        MLParameterGradient
            This gradient.
        """
        if other.parameters is not self.parameters:
            raise ValueError("The gradients belong to different parameter blocks.")
        for dest, src in zip(self.tensors, other.tensors, strict=True):
            dest.copy_(src)
        return self

    @classmethod
    def _linear_combination(cls, weights, gradients):
        parameters = gradients[0].parameters
        if any(g.parameters is not parameters for g in gradients):
            raise ValueError("The gradients belong to different parameter blocks.")
        blocks = zip(*(g.tensors for g in gradients), strict=True)
        return cls(parameters, tuple(sum(w * t for w, t in zip(weights, block, strict=True))
                                     for block in blocks))

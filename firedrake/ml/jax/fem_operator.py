import os
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    if "FIREDRAKE_BUILDING_DOCS" in os.environ:
        # If building docs and jax is not installed, produce a mock `jax.custom_vjp` function.
        # This is sufficient for the intersphinx reference to resolve.
        from types import SimpleNamespace
        jax = SimpleNamespace()

        def custom_vjp(_, **kwargs):
            pass

        jax.custom_vjp = custom_vjp
    else:
        raise ImportError("JAX is not installed and is required to use the FiredrakeJaxOperator.")

import collections
import numpy as np
from functools import partial
from typing import Union, Optional

from firedrake.function import Function
from firedrake.cofunction import Cofunction
from firedrake.vector import Vector
from firedrake.functionspaceimpl import WithGeometry
from firedrake.constant import Constant
from firedrake_citations import Citations

from pyadjoint.reduced_functional import ReducedFunctional


__all__ = ['FiredrakeJaxOperator', 'fem_operator', 'to_jax', 'from_jax']


class FiredrakeJaxOperator:
    """JAX custom operator representing a set of Firedrake operations expressed as a reduced functional `F`.

    `FiredrakeJaxOperator` executes forward and backward passes by directly calling the reduced functional `F`.

    Parameters
    ----------
    F
        The reduced functional to wrap.
    """

    def __init__(self, F: ReducedFunctional):
        super(FiredrakeJaxOperator, self).__init__()
        self.F = F
        self.V_controls = [c.control.function_space() for c in F.controls]
        self.V_output = _extract_function_space(F.functional)
        # Register forward and backward passes
        self.forward.defvjp(type(self).fwd, type(self).bwd)

    @partial(jax.custom_vjp, nondiff_argnums=(0,))
    def forward(self, *x_P: "jax.Array") -> "jax.Array":
        """Forward pass of the JAX custom operator.

        Parameters
        ----------
        *x_P
            JAX tensors representing the inputs to the Firedrake operator `F`.

        Returns
        -------
        jax.Array
            JAX tensor representing the output of the Firedrake operator `F`.
        """
        # Convert JAX input (i.e. controls) to Firedrake
        x_F = [from_jax(xi, Vi) for xi, Vi in zip(x_P, self.V_controls)]
        # Forward operator: delegated to pyadjoint.ReducedFunctional which recomputes the blocks on the tape
        y_F = self.F(x_F)
        # Convert Firedrake output to JAX
        y_P = to_jax(y_F)
        return y_P

    def fwd(self, *x_P: "jax.Array") -> "jax.Array":
        """Forward pass of the JAX custom operator.
        """
        return type(self).forward(self, *x_P), ()

    def bwd(self, _, grad_output: "jax.Array") -> "jax.Array":
        """Backward pass of the JAX custom operator.
        """
        V = self.V_output
        # Convert JAX gradient to Firedrake
        V_adj = V.dual() if V else V
        adj_input = from_jax(grad_output, V_adj)
        if isinstance(adj_input, Constant) and adj_input.ufl_shape == ():
            # This will later on result in an `AdjFloat` adjoint input instead of a Constant
            adj_input = float(adj_input)

        # Compute adjoint model of `F`: delegated to pyadjoint.ReducedFunctional
        adj_output = self.F.derivative(adj_input=adj_input)

        # Tuplify adjoint output
        adj_output = (adj_output,) if not isinstance(adj_output, collections.abc.Sequence) else adj_output

        return tuple(to_jax(di) for di in adj_output)


def fem_operator(F: ReducedFunctional) -> FiredrakeJaxOperator:
    """Cast a Firedrake reduced functional to a JAX operator.

    The resulting :class:`~FiredrakeJaxOperator` will take JAX tensors as inputs and return JAX tensors as outputs.

    Parameters
    ----------
    F
        The reduced functional to wrap.

    Returns
    -------
    firedrake.ml.jax.fem_operator.FiredrakeJaxOperator
        A JAX custom operator that wraps the reduced functional `F`.
    """
    Citations().register("Bouziani2024")

    if not isinstance(F, ReducedFunctional):
        raise ValueError("F must be a ReducedFunctional")

    jax_op = FiredrakeJaxOperator(F)
    # `jax_op.forward` currently does not work and causes issues related to the function
    #  signature during JAX compilation. As a workaround, we use `functools.partial` instead.
    return partial(FiredrakeJaxOperator.forward, jax_op)


def _extract_function_space(x: Union[float, Function, Vector]) -> Union[WithGeometry, None]:
    """Extract the function space from a Firedrake object `x`.

    Parameters
    ----------
    x
        Firedrake object from which to extract the function space.

    Returns
    -------
    firedrake.functionspaceimpl.WithGeometry or None
        Extracted function space.
    """
    if isinstance(x, (Function, Cofunction)):
        return x.function_space()
    elif isinstance(x, Vector):
        return _extract_function_space(x.function)
    elif isinstance(x, float):
        return None
    else:
        raise ValueError("Cannot infer the function space of %s" % x)


def to_jax(x: Union[Function, Vector, Constant], gather: Optional[bool] = False, batched: Optional[bool] = False, **kwargs) -> "jax.Array":
    """Convert a Firedrake object `x` into a JAX tensor.

    Parameters
    ----------
    x
        Firedrake object to convert.
    gather
             If True, gather data from all processes
    batched
              If True, add a batch dimension to the tensor
    kwargs
             Additional arguments to be passed to the :class:`jax.Array` constructor such as:
                - device: device on which the tensor is allocated
                - dtype: the desired data type of returned tensor (default: type of `x.dat.data`)

    Returns
    -------
    jax.Array
        JAX tensor representing the Firedrake object `x`.
    """
    if isinstance(x, (Function, Cofunction, Vector)):
        if gather:
            # Gather data from all processes
            x_P = jnp.array(x.vector().gather(), **kwargs)
        else:
            # Use local data
            x_P = jnp.array(x.vector().get_local(), **kwargs)
        if batched:
            # Default behaviour: add batch dimension after converting to JAX
            return x_P[None, :]
        return x_P
    elif isinstance(x, Constant):
        return jnp.array(x.values(), **kwargs)
    elif isinstance(x, (float, int)):
        if isinstance(x, float):
            # Set double-precision
            kwargs['dtype'] = jnp.double
        return jnp.array(x, **kwargs)
    else:
        raise ValueError("Cannot convert %s to a JAX tensor" % str(type(x)))


def from_jax(x: "jax.Array", V: Optional[WithGeometry] = None) -> Union[Function, Constant]:
    """Convert a JAX tensor `x` into a Firedrake object.

    Parameters
    ----------
    x
        JAX tensor to convert.
    V
        Function space of the corresponding :class:`.Function` or None when `x` is to be mapped to a :class:`.Constant`.

    Returns
    -------
    firedrake.function.Function or firedrake.constant.Constant
        Firedrake object representing the JAX tensor `x`.
    """

    if isinstance(x, jax.core.Tracer):
        x = jax.core.get_aval(x)

    if isinstance(x, jax.core.ShapedArray):
        if not isinstance(x, jax.core.ConcreteArray):
            raise TypeError("Cannot convert a JAX abstract array to a Firedrake object.")
        x = x.val

    if not isinstance(x, np.ndarray) and x.device.platform != "cpu":
        raise NotImplementedError("Firedrake does not support GPU/TPU tensors")

    if V is None:
        val = np.asarray(x)
        if val.shape == (1,):
            val = val[0]
        return Constant(val)
    else:
        x = np.asarray(x)
        x_F = Function(V)
        x_F.vector().set_local(x)
        return x_F

r"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
from __future__ import annotations

import collections
import functools
import warnings
from cachetools import LRUCache
from immutabledict import immutabledict as idict
from typing import Any

import FIAT
import finat
import loopy
import numpy as np
import pyop3 as op3
import ufl
from pyop3.cache import heavy_caches, serial_cache
from pyop3 import READ, WRITE, RW, INC, MIN_WRITE as MIN, MAX_WRITE as MAX
from pyop3.expr.visitors import evaluate as eval_expr
from pyop3.utils import readonly
from ufl.indexed import Indexed
from ufl.domain import join_domains

from firedrake import utils
from firedrake.constant import Constant
from firedrake.functionspaceimpl import WithGeometry, MixedFunctionSpace
from firedrake.matrix import Matrix
from firedrake.mesh import get_iteration_spec
from firedrake.pack import pack
from firedrake.petsc import PETSc
from firedrake.parameters import target
from firedrake.ufl_expr import extract_domains
from firedrake.utils import IntType, assert_empty, tuplify


# Set a default loopy language version (should be in __init__.py)
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa: F401


kernel_cache = LRUCache(maxsize=128)


__all__ = ['par_loop', 'direct']


class _DirectLoop(object):
    r"""A singleton object which can be used in a :func:`par_loop` in place
    of the measure in order to indicate that the loop is a direct loop
    over degrees of freedom."""

    def integral_type(self):
        return "direct"

    def __repr__(self):

        return "direct"


direct = _DirectLoop()
r"""A singleton object which can be used in a :func:`par_loop` in place
of the measure in order to indicate that the loop is a direct loop
over degrees of freedom."""


def _form_loopy_kernel(kernel_domains, instructions, measure, args, **kwargs) -> op3.Function:
    intents = []
    kargs = []
    for var, (func, intent) in args.items():
        is_input = intent in [INC, READ, RW]
        is_output = intent in [INC, RW, WRITE]
        if isinstance(func, Constant):
            if intent is not READ:
                raise RuntimeError("Only READ access is allowed to Constant")
            # Constants modelled as Globals, so no need for double
            # indirection
            ndof = func.dat.axes.block_size
            kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,), is_input=is_input, is_output=is_output))
        else:
            # Do we have a component of a mixed function?
            if isinstance(func, Indexed):
                func = _extract_subfunction(func)
                ndof = func.function_space().finat_element.space_dimension()
                cdim = func.function_space().block_size
                dtype = func.dat.dtype
            else:
                if func.function_space().ufl_element().family() == "Real":
                    ndof = func.function_space().dim()  # == 1
                    kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,), is_input=is_input, is_output=is_output))
                    continue
                else:
                    if len(func.function_space()) > 1:
                        raise NotImplementedError("Must index mixed function in par_loop.")
                    ndof = func.function_space().finat_element.space_dimension()
                    cdim = func.function_space().block_size
                    dtype = func.dat.dtype
            if measure.integral_type() == 'interior_facet':
                ndof *= 2
            # FIXME: shape for facets [2][ndof]?
            kargs.append(loopy.GlobalArg(var, dtype=dtype, shape=(ndof, cdim), is_input=is_input, is_output=is_output))
        kernel_domains = kernel_domains.replace(var+".dofs", str(ndof))

        intents.append(intent)

    if kernel_domains == "":
        kernel_domains = "[] -> {[]}"
    key = (kernel_domains, tuple(instructions), tuple(map(tuple, kwargs.items())))
    # Add shape, dtype and intent to the cache key
    for func, intent in args.values():
        if isinstance(func, Indexed):
            func = _extract_subfunction(func)
        key += (func.dat.axes, func.dat.dtype, intent)
    try:
        return kernel_cache[key]
    except KeyError:
        kargs.append(...)
        knl = loopy.make_kernel(
            kernel_domains,
            instructions,
            kargs,
            name="par_loop_kernel",
            target=target,
            seq_dependencies=True,
            silenced_warnings=["summing_if_branches_ops"],
        )
        knl = op3.Function(knl, intents)
        return kernel_cache.setdefault(key, knl)


@PETSc.Log.EventDecorator()
def par_loop(kernel, measure, args, kernel_kwargs=None, **kwargs):
    r"""A :func:`par_loop` is a user-defined operation which reads and
    writes :class:`.Function`\s by looping over the mesh cells or facets
    and accessing the degrees of freedom on adjacent entities.

    :arg kernel: A 2-tuple of (domains, instructions) to create
        a loopy kernel . The domains and instructions should be specified
        in loopy kernel syntax. See the `loopy tutorial
        <https://documen.tician.de/loopy/tutorial.html>`_ for details.

    :arg measure: is a UFL :class:`~ufl.measure.Measure` which determines the
        manner in which the iteration over the mesh is to occur.
        Alternatively, you can pass :data:`direct` to designate a direct loop.
    :arg args: is a dictionary mapping variable names in the kernel to
        :class:`.Function`\s or components of mixed :class:`.Function`\s and
        indicates how these :class:`.Function`\s are to be accessed.
    :arg kernel_kwargs: keyword arguments to be passed to the
        ``pyop3.Function`` constructor
    :arg kwargs: additional keyword arguments are passed to the underlying
        ``pyop3.loop``

    **Example**

    Assume that `A` is a :class:`.Function` in CG1 and `B` is a
    :class:`.Function` in DG0. Then the following code sets each DoF in
    `A` to the maximum value that `B` attains in the cells adjacent to
    that DoF::

      A.assign(numpy.finfo(0.).min)
      domain = '{[i]: 0 <= i < A.dofs}'
      instructions = '''
      for i
          A[i] = fmax(A[i], B[0])
      end
      '''
      par_loop((domain, instructions), dx, {'A' : (A, RW), 'B': (B, READ)})


    **Argument definitions**

    Each item in the `args` dictionary maps a string to a tuple
    containing a :class:`.Function` or :class:`.Constant` and an
    argument intent. The string is the c language variable name by
    which this function will be accessed in the kernel. The argument
    intent indicates how the kernel will access this variable:

    `READ`
       The variable will be read but not written to.
    `WRITE`
       The variable will be written to but not read. If multiple kernel
       invocations write to the same DoF, then the order of these writes
       is undefined.
    `RW`
       The variable will be both read and written to. If multiple kernel
       invocations access the same DoF, then the order of these accesses
       is undefined, but it is guaranteed that no race will occur.
    `INC`
       The variable will be added into using +=. As before, the order in
       which the kernel invocations increment the variable is undefined,
       but there is a guarantee that no races will occur.

    .. note::

       Only `READ` intents are valid for :class:`.Constant`
       coefficients, and an error will be raised in other cases.

    **The measure**

    The measure determines the mesh entities over which the iteration
    will occur, and the size of the kernel stencil. The iteration will
    occur over the same mesh entities as if the measure had been used
    to define an integral, and the stencil will likewise be the same
    as the integral case. That is to say, if the measure is a volume
    measure, the kernel will be called once per cell and the DoFs
    accessible to the kernel will be those associated with the cell,
    its facets, edges and vertices. If the measure is a facet measure
    then the iteration will occur over the corresponding class of
    facets and the accessible DoFs will be those on the cell(s)
    adjacent to the facet, and on the facets, edges and vertices
    adjacent to those facets.

    For volume measures the DoFs are guaranteed to be in the FInAT
    local DoFs order. For facet measures, the DoFs will be in sorted
    first by the cell to which they are adjacent. Within each cell,
    they will be in FInAT order. Note that if a continuous
    :class:`.Function` is accessed via an internal facet measure, the
    DoFs on the interface between the two facets will be accessible
    twice: once via each cell. The orientation of the cell(s) relative
    to the current facet is currently arbitrary.

    A direct loop over nodes without any indirections can be specified
    by passing :data:`direct` as the measure. In this case, all of the
    arguments must be :class:`.Function`\s in the same
    :class:`.FunctionSpace`.

    **The kernel code**

    Indirect free variables referencing :class:`.Function`\s are all
    of type `double*`. For spaces with rank greater than zero (Vector
    or TensorElement), the data are laid out XYZ... XYZ... XYZ....
    With the vector/tensor component moving fastest.

    In loopy syntax, these may be addressed using 2D indexing::

       A[i, j]

    Where ``i`` runs over nodes, and ``j`` runs over components.

    In a direct :func:`par_loop`, the variables will all be of type
    `double*` with the single index being the vector component.

    :class:`.Constant`\s are always of type `double*`, both for
    indirect and direct :func:`par_loop` calls.

    """
    if kernel_kwargs is None:
        kernel_kwargs = {}

    meshes = []
    for func, _ in args.values():
        meshes.extend(extract_domains(func))
    # Assume only one domain
    mesh, = join_domains(meshes)

    kernel_domains, instructions = kernel
    function = _form_loopy_kernel(kernel_domains, instructions, measure, args, **kernel_kwargs)

    with heavy_caches([mesh.topology]):
        if measure is direct:
            iterset = None
            for (func, _) in args.values():
                func = _extract_subfunction(func)

                if (
                    isinstance(func, Constant)
                    or func.function_space().ufl_element().family() == "Real"
                ):
                    continue

                if iterset is None:
                    iterset = func.function_space().nodal_axes
                else:
                    if func.function_space().nodal_axes != iterset:
                        raise ValueError("Cannot mix node sets in direct loop")
            if iterset is None:
                raise TypeError("No functions passed to direct loop")

            loop_index = iterset.iter()
            packed_args = []
            for (func, _) in args.values():
                func = _extract_subfunction(func)
                if isinstance(func, Constant):
                    packed_args.append(func.dat)
                else:
                    packed_args.append(func.dat[loop_index])

        else:
            iter_spec = get_iteration_spec(
                mesh, measure.integral_type(), measure.subdomain_id()
            )
            loop_index = iter_spec.loop_index
            packed_args = []
            for func, _ in args.values():
                func = _extract_subfunction(func)
                packed_args.append(pack(func, iter_spec))

        op3.loop(loop_index, function(*packed_args), eager=True)


def _extract_subfunction(func):
    if isinstance(func, Indexed):
        c, i = func.ufl_operands
        return c.subfunctions[i._indices[0]._value]
    else:
        return func

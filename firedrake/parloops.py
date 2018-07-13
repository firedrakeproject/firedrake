"""This module implements parallel loops reading and writing
:class:`.Function`\s. This provides a mechanism for implementing
non-finite element operations such as slope limiters."""
import collections

from ufl.indexed import Indexed
from ufl.domain import join_domains

from pyop2 import READ, WRITE, RW, INC, MIN, MAX
import pyop2
import loopy

from firedrake import constant


__all__ = ['par_loop', 'direct', 'READ', 'WRITE', 'RW', 'INC', 'MIN', 'MAX']


class _DirectLoop(object):
    """A singleton object which can be used in a :func:`par_loop` in place
    of the measure in order to indicate that the loop is a direct loop
    over degrees of freedom."""

    def integral_type(self):
        return "direct"

    def __repr__(self):

        return "direct"


direct = _DirectLoop()
"""A singleton object which can be used in a :func:`par_loop` in place
of the measure in order to indicate that the loop is a direct loop
over degrees of freedom."""


def indirect_measure(mesh, measure):
    return mesh.measure_set(measure.integral_type(),
                            measure.subdomain_id())


_maps = {
    'cell': {
        'nodes': lambda x: x.cell_node_map(),
        'itspace': indirect_measure
    },
    'interior_facet': {
        'nodes': lambda x: x.interior_facet_node_map(),
        'itspace': indirect_measure
    },
    'exterior_facet': {
        'nodes': lambda x: x.exterior_facet_node_map(),
        'itspace': indirect_measure
    },
    'direct': {
        'nodes': lambda x: None,
        'itspace': lambda mesh, measure: mesh
    }
}
"""Map a measure to the correct maps."""


def _form_kernel(kernel_domains, instructions, measure, args, **kwargs):

    kargs = []

    for var, (func, intent) in args.items():
        if isinstance(func, constant.Constant):
            if intent is not READ:
                raise RuntimeError("Only READ access is allowed to Constant")
            # Constants modelled as Globals, so no need for double
            # indirection
            ndof = func.dat.cdim
            kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,)))
        else:
            # Do we have a component of a mixed function?
            if isinstance(func, Indexed):
                c, i = func.ufl_operands
                idx = i._indices[0]._value
                ndof = c.function_space()[idx].finat_element.space_dimension()
                cdim = c.dat[idx].cdim
                dtype = c.dat[idx].dtype
            else:
                if func.function_space().ufl_element().family() == "Real":
                    ndof = func.function_space().dim  # == 1
                    kargs.append(loopy.GlobalArg(var, dtype=func.dat.dtype, shape=(ndof,)))
                else:
                    if len(func.function_space()) > 1:
                        raise NotImplementedError("Must index mixed function in par_loop.")
                    ndof = func.function_space().finat_element.space_dimension()
                    cdim = func.dat.cdim
                    dtype = func.dat.dtype
            if measure.integral_type() == 'interior_facet':
                ndof *= 2
            # FIXME: shape for facets [2][ndof]?
            kargs.append(loopy.GlobalArg(var, dtype=dtype, shape=(ndof, cdim)))
        kernel_domains = kernel_domains.replace(var+".dofs", str(ndof))

    if kernel_domains is "":
        kernel_domains = "[] -> {[]}"
    kargs.append(...)
    knl = loopy.make_kernel(kernel_domains, instructions, kargs, seq_dependencies=True, name="par_loop_kernel", lang_version=(2018, 1))

    return pyop2.Kernel(knl, "par_loop_kernel", **kwargs)


def par_loop(kernel_domains, instructions, measure, args, **kwargs):
    """A :func:`par_loop` is a user-defined operation which reads and
    writes :class:`.Function`\s by looping over the mesh cells or facets
    and accessing the degrees of freedom on adjacent entities.

    :arg kernel_domains: the iteration domain for the kernel, in Integer Set
        Libarary (ISL) syntax. function_name.dofs can be used as a magic word.
    :arg instructions: is a string containing the instruction for the kernel
        in loo.py syntax.
    :arg measure: is a UFL :class:`~ufl.measure.Measure` which determines the
        manner in which the iteration over the mesh is to occur.
        Alternatively, you can pass :data:`direct` to designate a direct loop.
    :arg args: is a dictionary mapping variable names in the kernel to
        :class:`.Function`\s or components of mixed :class:`.Function`\s and
        indicates how these :class:`.Function`\s are to be accessed.
    :arg kwargs: additional keyword arguments are passed to the
        :class:`~pyop2.op2.Kernel` constructor

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
      par_loop(domain, instructions, dx, {'A' : (A, RW), 'B': (B, READ)})


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

    The kernel code is plain C in which the variables specified in the
    `args` dictionary are available to be read or written in according
    to the argument intent specified. Most basic C operations are
    permitted. However there are some restrictions:

    * Only functions from `math.h
      <http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/math.h.html>`_
      may be called.
    * Pointer operations other than dereferencing arrays are prohibited.

    Indirect free variables referencing :class:`.Function`\s are all
    of type `double**` in which the first index is the local node
    number, while the second index is the vector (or tensor)
    component. The latter only applies to :class:`.Function`\s over a
    :class:`.FunctionSpace` with :attr:`.FunctionSpace.rank` greater
    than zero (spaces with a VectorElement or TensorElement).  In the
    case of scalar :class:`FunctionSpace`\s, the second index is
    always 0.

    In a direct :func:`par_loop`, the variables will all be of type
    `double*` with the single index being the vector component.

    :class:`.Constant`\s are always of type `double*`, both for
    indirect and direct :func:`par_loop` calls.

    """

    _map = _maps[measure.integral_type()]
    # Ensure that the dict args passed in are consistently ordered
    # (sorted by the string key).
    sorted_args = collections.OrderedDict()
    for k in sorted(args.keys()):
        sorted_args[k] = args[k]
    args = sorted_args

    if measure is direct:
        mesh = None
        for (func, intent) in args.values():
            if isinstance(func, Indexed):
                c, i = func.ufl_operands
                idx = i._indices[0]._value
                if mesh and c.node_set[idx] is not mesh:
                    raise ValueError("Cannot mix sets in direct loop.")
                mesh = c.node_set[idx]
            else:
                try:
                    if mesh and func.node_set is not mesh:
                        raise ValueError("Cannot mix sets in direct loop.")
                    mesh = func.node_set
                except AttributeError:
                    # Argument was a Global.
                    pass
        if not mesh:
            raise TypeError("No Functions passed to direct par_loop")
    else:
        domains = []
        for func, _ in args.values():
            domains.extend(func.ufl_domains())
        domains = join_domains(domains)
        # Assume only one domain
        domain, = domains
        mesh = domain

    op2args = [_form_kernel(kernel_domains, instructions, measure, args, **kwargs)]

    op2args.append(_map['itspace'](mesh, measure))

    def mkarg(f, intent):
        if isinstance(f, Indexed):
            c, i = f.ufl_operands
            idx = i._indices[0]._value
            m = _map['nodes'](c)
            return c.dat[idx](intent, m.split[idx] if m else None)
        return f.dat(intent, _map['nodes'](f))
    op2args += [mkarg(func, intent) for (func, intent) in args.values()]

    return pyop2.par_loop(*op2args)

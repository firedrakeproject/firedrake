"""
Firedrake uses PETSc for its linear and nonlinear solvers.  The
interaction is carried out through DM objects.  These carry around any
user-defined application context and can be used to inform the solvers
how to create field decompositions (for fieldsplit preconditioning) as
well as creating sub-DMs (which only contain some fields), along with
multilevel information (for geometric multigrid)

The way Firedrake interacts with these DMs is, broadly, as follows:

A DM is tied to a :class:`~.FunctionSpace` and remembers what function
space that is.  To avoid reference cycles defeating the garbage
collector, the DM holds a weakref to the FunctionSpace (which holds a
strong reference to the DM).  Use :func:`get_function_space` to get
the function space attached to the DM, and :func:`set_function_space`
to attach it.

Similarly, when a DM is used in a solver, an application context is
attached to it, such that when PETSc calls back into Firedrake, we can
grab the relevant information (how to make the Jacobian, etc...).
This functions in a similar way using :func:`push_appctx` and
:func:`get_appctx` on the DM.  You can set whatever you like in here,
but most of the rest of Firedrake expects to find either ``None`` or
else a :class:`firedrake.solving_utils._SNESContext` object.

A crucial part of this, for composition with multi-level solvers
(``-pc_type mg`` and ``-snes_type fas``) is decomposing the DMs.  When
a field decomposition is created, the callback
:func:`create_field_decomposition` checks to see if an application
context exists.  If so, it splits it apart (one for each of fields)
and attaches these split contexts to the subdms returned to PETSc.
This facilitates runtime composition with multilevel solvers.  When
coarsening a DM, the application context is coarsened and transferred
to the coarse DM.  The combination of these two symbolic transfer
operations allow us to nest geometric multigrid preconditioning inside
fieldsplit preconditioning, without having to set everything up in
advance.
"""

import weakref
import numpy

import firedrake
from firedrake.petsc import PETSc


def get_function_space(dm):
    """Get the :class:`~.FunctionSpace` attached to this DM.

    :arg dm: The DM to get the function space from.
    :raises RuntimeError: if no function space was found.
    """
    V = dm.getAttr("__fs_info__")()
    if V is None:
        raise RuntimeError("FunctionSpace not found on DM")
    return V


def set_function_space(dm, V):
    """Set the :class:`~.FunctionSpace` on this DM.

    :arg dm: The DM
    :arg V: The function space.

    .. note::

       This stores a weakref to the function space in the DM, so you
       should hold a strong reference somewhere else.

    """
    dm.setAttr("__fs_info__", weakref.ref(V))


def push_appctx(dm, ctx):
    """Push an application context onto a DM.

    :arg DM: The DM.
    :arg ctx: The context.

    .. note::

       This stores a weakref to the context in the DM, so you should
       hold a strong reference somewhere else.
    """
    stack = dm.getAppCtx()
    if stack is None:
        stack = []
        dm.setAppCtx(stack)

    def finalize(ref):
        stack = dm.getAppCtx()
        try:
            stack.remove(ref)
        except ValueError:
            pass

    stack.append(weakref.ref(ctx, finalize))


def pop_appctx(dm, match=None):
    """Pop the most recent application context from a DM.

    :arg DM: The DM.
    :returns: Either an application context, or ``None``.
    """
    stack = dm.getAppCtx()
    if stack == [] or stack is None:
        return None
    ctx = stack[-1]()
    if match is not None:
        if ctx == match:
            return stack.pop()()
    else:
        return stack.pop()()


def get_appctx(dm):
    """Get the most recent application context from a DM.

    :arg DM: The DM.
    :returns: Either the stored application context, or ``None`` if
       none was found.
    """
    stack = dm.getAppCtx()
    if stack == [] or stack is None:
        return None
    else:
        return stack[-1]()


class appctx(object):
    def __init__(self, dm, ctx):
        self.ctx = ctx
        self.dm = dm

    def __enter__(self):
        push_appctx(self.dm, self.ctx)

    def __exit__(self, typ, value, traceback):
        ctx = self.ctx
        while ctx._coarse is not None:
            ctx = ctx._coarse

        def get_dm(c):
            return c._problem.u.function_space().dm

        while ctx._fine is not None:
            dm = get_dm(ctx)
            pop_appctx(dm, ctx)
            ctx = ctx._fine
        pop_appctx(self.dm, self.ctx)


def create_matrix(dm):
    """
    Callback to create a matrix from this DM.

    :arg DM: The DM.

    .. note::

       This only works if an application context is set, in which case
       it returns the stored Jacobian.  This *does not* make a new
       matrix.
    """
    ctx = get_appctx(dm)
    if ctx is None:
        raise ValueError("Cannot create matrix from DM with no AppCtx")
    # TODO, should make new matrix and change solver
    # form_function/jacobian to be able to assemble into a provided
    # PETSc matrix.
    return ctx._jac.petscmat


def create_field_decomposition(dm, *args, **kwargs):
    """Callback to decompose a DM.

    :arg DM: The DM.

    This grabs the function space in the DM, splits it apart (only
    makes sense for mixed function spaces) and returns the DMs on each
    of the subspaces.  If an application context is present on the
    input DM, it is split into individual field contexts and set on
    the appropriate subdms as well.
    """
    W = get_function_space(dm)
    # Don't pass split number if name is None (this way the
    # recursively created splits have the names you want)
    names = [s.name for s in W]
    dms = [V.dm for V in W]
    ctx = get_appctx(dm)
    if ctx is not None:
        # DM from a hierarchy, so let's split apart in case we want to use it
        # Inside a solve, ctx to split.  If we're not from a
        # hierarchy, this information is not used, so don't bother
        # splitting, since it costs some time.
        if dm.getRefineLevel() - dm.getCoarsenLevel() != 0:
            ctxs = ctx.split([i for i in range(len(W))])
            for d, c in zip(dms, ctxs):
                push_appctx(d, c)
    return names, W._ises, dms


def create_subdm(dm, fields, *args, **kwargs):
    """Callback to create a sub-DM describing the specified fields.

    :arg DM: The DM.
    :arg fields: The fields in the new sub-DM.

    .. note::

       This should, but currently does not, transfer appropriately
       split application contexts onto the sub-DMs.
    """
    W = get_function_space(dm)
    # TODO: Correct splitting of SNESContext for len(fields) > 1 case
    if len(fields) == 1:
        # Subspace is just a single FunctionSpace.
        idx = fields[0]
        subdm = W[idx].dm
        iset = W._ises[idx]
        return iset, subdm
    else:
        try:
            # Look up the subspace in the cache
            iset, subspace = W._subspaces[tuple(fields)]
            return iset, subspace.dm
        except KeyError:
            pass
        # Need to build an MFS for the subspace
        subspace = firedrake.MixedFunctionSpace([W[f] for f in fields])
        # Index set mapping from W into subspace.
        iset = PETSc.IS().createGeneral(numpy.concatenate([W._ises[f].indices
                                                           for f in fields]),
                                        comm=W.comm)
        # Keep hold of strong reference to created subspace (given we
        # only hold a weakref in the shell DM), and so we can
        # reuse it later.
        W._subspaces[tuple(fields)] = iset, subspace
        return iset, subspace.dm


def coarsen(dm, comm):
    """Callback to coarsen a DM.

    :arg DM: The DM to coarsen.
    :arg comm: The communicator for the new DM (ignored)

    This transfers a coarse application context over to the coarsened
    DM (if found on the input DM).
    """
    from firedrake.mg.utils import get_level
    from firedrake.mg.ufl_utils import coarsen
    V = get_function_space(dm)
    if V is None:
        raise RuntimeError("No functionspace found on DM")
    hierarchy, level = get_level(V.mesh())
    if level < 1:
        raise RuntimeError("Cannot coarsen coarsest DM")
    if hasattr(V, "_coarse"):
        cdm = V._coarse.dm
    else:
        V._coarse = firedrake.FunctionSpace(hierarchy[level - 1], V.ufl_element())
        cdm = V._coarse.dm
    ctx = get_appctx(dm)
    if ctx is not None:
        push_appctx(cdm, coarsen(ctx))
        # Necessary for MG inside a fieldsplit in a SNES.
        cdm.setKSPComputeOperators(firedrake.solving_utils._SNESContext.compute_operators)
    V._coarse._fine = V
    return cdm


def refine(dm, comm):
    """Callback to refine a DM.

    :arg DM: The DM to refine.
    :arg comm: The communicator for the new DM (ignored)
    """
    from firedrake.mg.utils import get_level
    V = get_function_space(dm)
    if V is None:
        raise RuntimeError("No functionspace found on DM")
    hierarchy, level = get_level(V.mesh())
    if level >= len(hierarchy) - 1:
        raise RuntimeError("Cannot refine finest DM")
    if hasattr(V, "_fine"):
        fdm = V._fine.dm
    else:
        V._fine = firedrake.FunctionSpace(hierarchy[level + 1], V.ufl_element())
        fdm = V._fine.dm
    V._fine._coarse = V
    return fdm


def attach_hooks(dm, level=None, sf=None, section=None):
    """Attach callback hooks to a DM.

    :arg DM: The DM to attach callbacks to.
    :arg level: Optional refinement level.
    :arg sf: Optional PETSc SF object describing the DM's ``points``.
    :arg section: Optional PETSc Section object describing the DM's
        data layout.
    """
    from firedrake.mg.ufl_utils import create_interpolation, create_injection
    # Data layout
    if sf is not None:
        dm.setPointSF(sf)
    if section is not None:
        dm.setDefaultSection(section)

    # Multilevel hierarchies
    dm.setRefine(refine)
    dm.setCoarsen(coarsen)
    dm.setCreateMatrix(create_matrix)
    dm.setCreateInterpolation(create_interpolation)
    dm.setCreateInjection(create_injection)
    if level is not None:
        dm.setRefineLevel(level)

    # Field splitting (these will never be called if the DM references
    # a non-mixed space)
    dm.setCreateFieldDecomposition(create_field_decomposition)
    dm.setCreateSubDM(create_subdm)

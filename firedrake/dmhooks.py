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
from functools import partial

import firedrake
from firedrake.petsc import PETSc


@PETSc.Log.EventDecorator()
def get_function_space(dm):
    """Get the :class:`~.FunctionSpace` attached to this DM.

    :arg dm: The DM to get the function space from.
    :raises RuntimeError: if no function space was found.
    """
    info = dm.getAttr("__fs_info__")
    meshref, element, indices, (name, names) = info
    mesh = meshref()
    if mesh is None:
        raise RuntimeError("Somehow your mesh was collected, this should never happen")
    V = firedrake.FunctionSpace(mesh, element, name=name)
    if len(V) > 1:
        for V_, name in zip(V, names):
            V_.topological.name = name
    for index in indices:
        V = V.sub(index)
    return V


@PETSc.Log.EventDecorator()
def set_function_space(dm, V):
    """Set the :class:`~.FunctionSpace` on this DM.

    :arg dm: The DM
    :arg V: The function space.

    .. note::

       This stores the information necessary to make a function space given a DM.

    """
    mesh = V.mesh()

    indices = []
    names = []
    while V.parent is not None:
        if V.index is not None:
            assert V.component is None
            indices.append(V.index)
        if V.component is not None:
            assert V.index is None
            indices.append(V.component)
        V = V.parent
    if len(V) > 1:
        names = tuple(V_.name for V_ in V)
    element = V.ufl_element()

    info = (weakref.ref(mesh), element, tuple(reversed(indices)), (V.name, names))
    dm.setAttr("__fs_info__", info)


# Attribute management on DMs. Since they are reused in multiple
# places, use a stack.
def push_attr(attr, dm, obj):
    stack = dm.getAttr(attr)
    if stack is None:
        stack = []
        dm.setAttr(attr, stack)
    stack.append(obj)


def pop_attr(attr, dm, match=None):
    stack = dm.getAttr(attr)
    if not stack:
        return None
    obj = stack.pop()
    if match is not None and obj != match:
        stack.append(obj)
        return None
    else:
        return obj


def get_attr(attr, dm, default=None):
    stack = dm.getAttr(attr)
    if not stack:
        return default
    return stack[-1]


class SetupHooks(object):
    """Hooks run for setup and teardown of DMs inside solvers.

    Used for transferring problem-specific data onto subproblems.

    You probably don't want to use this directly, instead see
    :class:`~add_hooks` or :func:`add_hook`."""
    def __init__(self):
        self._setup = []
        self._teardown = []

    def add_setup(self, f):
        self._setup.append(f)

    def add_teardown(self, f):
        self._teardown.append(f)

    def setup(self):
        for f in self._setup:
            f()

    def teardown(self):
        # We push onto a stack, so to pop off in the correct order, go backwards.
        for f in reversed(self._teardown):
            f()


def add_hook(dm, setup=None, teardown=None, call_setup=False, call_teardown=False):
    """Add a hook to a DM to be called for setup/teardown of
    subproblems.

    :arg dm: The DM to save the hooks on. This is normally the DM
         associated with the Firedrake solver.
    :arg setup: function of no arguments to call to set up subproblem
         data.
    :arg teardown: function of no arguments to call to remove
         subproblem data.
    :arg call_setup: Should the setup function be called now?
    :arg call_teardown: Should the teardown function be called now?

    See also :class:`add_hooks` which provides a context manager which
    manages everything."""
    stack = dm.getAttr("__setup_hooks__")
    if not stack:
        raise ValueError("Expecting non-empty stack")
    obj = stack[-1]
    if setup is not None:
        obj.add_setup(setup)
        if call_setup:
            setup()
    if teardown is not None:
        obj.add_teardown(teardown)
        if call_teardown:
            teardown()


class add_hooks(object):
    """Context manager for adding subproblem setup hooks to a DM.

    :arg DM: The DM to remember setup/teardown for.
    :arg obj: The object that we're going to setup, typically a solver
       of some kind: this is where the hooks are saved.
    :arg save: Save this round of setup? Set this to False if all
        you're going to do is setFromOptions.
    :arg appctx: An application context to attach to the top-level DM
        that describes the problem-specific data.

    This is your normal entry-point for setting up problem specific
    data on subdms. You would likely do something like, for a Python PC.

    .. code::

       # In setup
       pc = ...
       pc.setDM(dm)
       with dmhooks.add_hooks(dm, self, appctx=ctx, save=False):
           pc.setFromOptions()

       ...

       # in apply
       dm = pc.getDM()
       with dmhooks.add_hooks(dm, self, appctx=self.ctx):
          pc.apply(...)
    """
    def __init__(self, dm, obj, *, save=True, appctx=None):
        self.dm = dm
        self.obj = obj
        self.first_time = not hasattr(obj, "setup_hooks")
        self.save = save
        self.appctx = appctx
        if not (self.save or self.first_time):
            raise ValueError("Can't have save=False for non-first-time usage")

    def __enter__(self):
        if not self.first_time:
            # We've already run setup, so just attach the data to the subdms.
            hooks = self.obj.setup_hooks
            push_attr("__setup_hooks__", self.dm, hooks)
            hooks.setup()
        else:
            # Not yet seen, let's save the relevant information.
            hooks = SetupHooks()
            if self.save:
                # Remember it for later
                self.obj.setup_hooks = hooks
            push_attr("__setup_hooks__", self.dm, hooks)
            if self.appctx is not None:
                add_hook(self.dm, setup=partial(push_appctx, self.dm, self.appctx),
                         teardown=partial(pop_appctx, self.dm, self.appctx),
                         call_setup=True)

    def __exit__(self, typ, value, traceback):
        hooks = pop_attr("__setup_hooks__", self.dm)
        if self.first_time:
            assert hooks is not None
        else:
            assert hooks == self.obj.setup_hooks
        hooks.teardown()


# Things we're going to transfer around DMs
push_parent = partial(push_attr, "__parent__")
pop_parent = partial(pop_attr, "__parent__")


def get_parent(dm):
    return get_attr("__parent__", dm, default=dm)


push_appctx = partial(push_attr, "__appctx__")
pop_appctx = partial(pop_attr, "__appctx__")
get_appctx = partial(get_attr, "__appctx__")


def get_transfer_manager(dm):
    appctx = get_appctx(dm)
    if appctx is None:
        # We're not in a solve, so all we can do is make a new one (not cached)
        import warnings
        warnings.warn("Creating new TransferManager to transfer data to coarse grids", RuntimeWarning)
        warnings.warn("This might be slow (you probably want to save it on an appctx)", RuntimeWarning)
        transfer = firedrake.TransferManager()
    else:
        transfer = appctx.transfer_manager
    return transfer


push_ctx_coarsener = partial(push_attr, "__ctx_coarsener__")
pop_ctx_coarsener = partial(pop_attr, "__ctx_coarsener__")


def get_ctx_coarsener(dm):
    from firedrake.mg.ufl_utils import coarsen
    return get_attr("__ctx_coarsener__", dm, default=coarsen)


class ctx_coarsener(object):
    def __init__(self, V, coarsen=None):
        from firedrake.mg.ufl_utils import coarsen as symbolic_coarsen
        self.V = V
        if coarsen is None:
            coarsen = symbolic_coarsen
        self.coarsen = coarsen

    def __enter__(self):
        push_ctx_coarsener(self.V.dm, self.coarsen)

    def __exit__(self, typ, value, traceback):
        pop_ctx_coarsener(self.V.dm, self.coarsen)


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


@PETSc.Log.EventDecorator()
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
    coarsen = get_ctx_coarsener(dm)
    parent = get_parent(dm)
    for d in dms:
        add_hook(parent, setup=partial(push_parent, d, parent), teardown=partial(pop_parent, d, parent),
                 call_setup=True)
    if ctx is not None and len(W) > 1:
        ctxs = ctx.split([(i, ) for i in range(len(W))])
        for d, c in zip(dms, ctxs):
            add_hook(parent, setup=partial(push_appctx, d, c), teardown=partial(pop_appctx, d, c),
                     call_setup=True)
            add_hook(parent, setup=partial(push_ctx_coarsener, d, coarsen), teardown=partial(pop_ctx_coarsener, d, coarsen),
                     call_setup=True)
    return names, W._ises, dms


@PETSc.Log.EventDecorator()
def create_subdm(dm, fields, *args, **kwargs):
    """Callback to create a sub-DM describing the specified fields.

    :arg DM: The DM.
    :arg fields: The fields in the new sub-DM.
    """
    W = get_function_space(dm)
    ctx = get_appctx(dm)
    coarsen = get_ctx_coarsener(dm)
    parent = get_parent(dm)
    if len(fields) == 1:
        # Subspace is just a single FunctionSpace.
        idx, = fields
        subdm = W[idx].dm
        iset = W._ises[idx]
        add_hook(parent, setup=partial(push_parent, subdm, parent), teardown=partial(pop_parent, subdm, parent),
                 call_setup=True)

        if ctx is not None:
            ctx, = ctx.split([(idx, )])
            add_hook(parent, setup=partial(push_appctx, subdm, ctx), teardown=partial(pop_appctx, subdm, ctx),
                     call_setup=True)
            add_hook(parent, setup=partial(push_ctx_coarsener, subdm, coarsen), teardown=partial(pop_ctx_coarsener, subdm, coarsen),
                     call_setup=True)
        return iset, subdm
    else:
        # Need to build an MFS for the subspace
        subspace = firedrake.MixedFunctionSpace([W[f] for f in fields])

        add_hook(parent, setup=partial(push_parent, subspace.dm, parent), teardown=partial(pop_parent, subspace.dm, parent),
                 call_setup=True)
        # Index set mapping from W into subspace.
        iset = PETSc.IS().createGeneral(numpy.concatenate([W._ises[f].indices
                                                           for f in fields]),
                                        comm=W._comm)
        if ctx is not None:
            ctx, = ctx.split([fields])
            add_hook(parent, setup=partial(push_appctx, subspace.dm, ctx),
                     teardown=partial(pop_appctx, subspace.dm, ctx),
                     call_setup=True)
            add_hook(parent, setup=partial(push_ctx_coarsener, subspace.dm, coarsen),
                     teardown=partial(pop_ctx_coarsener, subspace.dm, coarsen),
                     call_setup=True)
        return iset, subspace.dm


@PETSc.Log.EventDecorator()
def coarsen(dm, comm):
    """Callback to coarsen a DM.

    :arg DM: The DM to coarsen.
    :arg comm: The communicator for the new DM (ignored)

    This transfers a coarse application context over to the coarsened
    DM (if found on the input DM).
    """
    from firedrake.mg.utils import get_level
    V = get_function_space(dm)
    hierarchy, level = get_level(V.mesh())
    if level < 1:
        raise RuntimeError("Cannot coarsen coarsest DM")
    coarsen = get_ctx_coarsener(dm)
    Vc = coarsen(V, coarsen)
    cdm = Vc.dm
    parent = get_parent(dm)
    add_hook(parent, setup=partial(push_parent, cdm, parent), teardown=partial(pop_parent, cdm, parent),
             call_setup=True)
    if len(V) > 1:
        for V_, Vc_ in zip(V, Vc):
            add_hook(parent, setup=partial(push_parent, Vc_.dm, parent), teardown=partial(pop_parent, Vc_.dm, parent),
                     call_setup=True)
    add_hook(parent, setup=partial(push_ctx_coarsener, cdm, coarsen),
             teardown=partial(pop_ctx_coarsener, cdm, coarsen),
             call_setup=True)
    ctx = get_appctx(dm)
    if ctx is not None:
        cctx = coarsen(ctx, coarsen)
        add_hook(parent, setup=partial(push_appctx, cdm, cctx),
                 teardown=partial(pop_appctx, cdm, cctx),
                 call_setup=True)
        # Necessary for MG inside a fieldsplit in a SNES.
        dm.setKSPComputeOperators(firedrake.solving_utils._SNESContext.compute_operators)
        cdm.setKSPComputeOperators(firedrake.solving_utils._SNESContext.compute_operators)
    return cdm


@PETSc.Log.EventDecorator()
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

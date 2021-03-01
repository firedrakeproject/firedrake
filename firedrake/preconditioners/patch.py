from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.petsc import PETSc
from firedrake.cython.patchimpl import set_patch_residual, set_patch_jacobian
from firedrake.solving_utils import _SNESContext
from firedrake.utils import cached_property, complex_mode, IntType
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
from firedrake.functionspace import FunctionSpace
from firedrake.interpolation import interpolate

from collections import namedtuple
import operator
from itertools import chain
from functools import partial
import numpy
from ufl import VectorElement, MixedElement
from tsfc.kernel_interface.firedrake_loopy import make_builder
import weakref

import ctypes
from pyop2 import op2
from pyop2 import base as pyop2
from pyop2 import sequential as seq
from pyop2.compilation import load
from pyop2.utils import get_petsc_dir
from pyop2.codegen.builder import Pack, MatPack, DatPack
from pyop2.codegen.representation import Comparison, Literal
from pyop2.codegen.rep2loopy import register_petsc_function

__all__ = ("PatchPC", "PlaneSmoother", "PatchSNES")


class DenseSparsity(object):
    def __init__(self, rset, cset):
        self.shape = (1, 1)
        self._nrows = rset.size
        self._ncols = cset.size
        self._dims = (((1, 1), ), )
        self.dims = self._dims
        self.dsets = rset, cset

    def __getitem__(self, *args):
        return self

    def __contains__(self, *args):
        return True


class LocalPack(Pack):
    def pick_loop_indices(self, loop_index, layer_index, entity_index):
        return (entity_index, layer_index)


class LocalMatPack(LocalPack, MatPack):
    insertion_names = {False: "MatSetValues",
                       True: "MatSetValues"}


class LocalMat(pyop2.Mat):
    pack = LocalMatPack

    def __init__(self, dset):
        self._sparsity = DenseSparsity(dset, dset)
        self.dtype = numpy.dtype(PETSc.ScalarType)


class LocalDatPack(LocalPack, DatPack):
    def __init__(self, needs_mask, *args, **kwargs):
        self.needs_mask = needs_mask
        super().__init__(*args, **kwargs)

    def _mask(self, map_):
        if self.needs_mask:
            return Comparison(">=", map_, Literal(numpy.int32(0)))
        else:
            return None


class LocalDat(pyop2.Dat):
    def __init__(self, dset, needs_mask=False):
        self._dataset = dset
        self.dtype = numpy.dtype(PETSc.ScalarType)
        self._shape = (dset.total_size,) + (() if dset.cdim == 1 else dset.dim)
        self.needs_mask = needs_mask

    @cached_property
    def _wrapper_cache_key_(self):
        return super()._wrapper_cache_key_ + (self.needs_mask, )

    @property
    def pack(self):
        return partial(LocalDatPack, self.needs_mask)


register_petsc_function("MatSetValues")


CompiledKernel = namedtuple('CompiledKernel', ["funptr", "kinfo"])


def matrix_funptr(form, state):
    from firedrake.tsfc_interface import compile_form
    test, trial = map(operator.methodcaller("function_space"), form.arguments())
    if test != trial:
        raise NotImplementedError("Only for matching test and trial spaces")

    if state is not None:
        interface = make_builder(dont_split=(state, ))
    else:
        interface = None

    kernels = compile_form(form, "subspace_form", split=False, interface=interface)

    cell_kernels = []
    int_facet_kernels = []
    for kernel in kernels:
        kinfo = kernel.kinfo

        if kinfo.subdomain_id != "otherwise":
            raise NotImplementedError("Only for full domain integrals")
        if kinfo.integral_type not in {"cell", "interior_facet"}:
            raise NotImplementedError("Only for cell or interior facet integrals")

        # OK, now we've validated the kernel, let's build the callback
        args = []

        if kinfo.integral_type == "cell":
            get_map = operator.methodcaller("cell_node_map")
            kernels = cell_kernels
        elif kinfo.integral_type == "interior_facet":
            get_map = operator.methodcaller("interior_facet_node_map")
            kernels = int_facet_kernels
        else:
            get_map = None

        toset = op2.Set(1, comm=test.comm)
        dofset = op2.DataSet(toset, 1)
        arity = sum(m.arity*s.cdim
                    for m, s in zip(get_map(test),
                                    test.dof_dset))
        iterset = get_map(test).iterset
        entity_node_map = op2.Map(iterset,
                                  toset, arity,
                                  values=numpy.zeros(iterset.total_size*arity, dtype=IntType))
        mat = LocalMat(dofset)

        arg = mat(op2.INC, (entity_node_map, entity_node_map))
        arg.position = 0
        args.append(arg)
        statedat = LocalDat(dofset)
        state_entity_node_map = op2.Map(iterset,
                                        toset, arity,
                                        values=numpy.zeros(iterset.total_size*arity, dtype=IntType))
        statearg = statedat(op2.READ, state_entity_node_map)

        mesh = form.ufl_domains()[kinfo.domain_number]
        arg = mesh.coordinates.dat(op2.READ, get_map(mesh.coordinates))
        arg.position = 1
        args.append(arg)
        if kinfo.oriented:
            c = form.ufl_domain().cell_orientations()
            arg = c.dat(op2.READ, get_map(c))
            arg.position = len(args)
            args.append(arg)
        if kinfo.needs_cell_sizes:
            c = form.ufl_domain().cell_sizes
            arg = c.dat(op2.READ, get_map(c))
            arg.position = len(args)
            args.append(arg)
        for n in kinfo.coefficient_map:
            c = form.coefficients()[n]
            if c is state:
                statearg.position = len(args)
                args.append(statearg)
                continue
            for (i, c_) in enumerate(c.split()):
                map_ = get_map(c_)
                arg = c_.dat(op2.READ, map_)
                arg.position = len(args)
                args.append(arg)

        if kinfo.integral_type == "interior_facet":
            arg = test.ufl_domain().interior_facets.local_facet_dat(op2.READ)
            arg.position = len(args)
            args.append(arg)
        iterset = op2.Subset(iterset, [0])
        mod = seq.JITModule(kinfo.kernel, iterset, *args)
        kernels.append(CompiledKernel(mod._fun, kinfo))
    return cell_kernels, int_facet_kernels


def residual_funptr(form, state):
    from firedrake.tsfc_interface import compile_form
    test, = map(operator.methodcaller("function_space"), form.arguments())

    if state.function_space() != test:
        raise NotImplementedError("State and test space must be dual to one-another")

    if state is not None:
        interface = make_builder(dont_split=(state, ))
    else:
        interface = None

    kernels = compile_form(form, "subspace_form", split=False, interface=interface)

    cell_kernels = []
    int_facet_kernels = []
    for kernel in kernels:
        kinfo = kernel.kinfo

        if kinfo.subdomain_id != "otherwise":
            raise NotImplementedError("Only for full domain integrals")
        if kinfo.integral_type not in {"cell", "interior_facet"}:
            raise NotImplementedError("Only for cell integrals or interior_facet integrals")
        args = []

        if kinfo.integral_type == "cell":
            get_map = operator.methodcaller("cell_node_map")
            kernels = cell_kernels
        elif kinfo.integral_type == "interior_facet":
            get_map = operator.methodcaller("interior_facet_node_map")
            kernels = int_facet_kernels
        else:
            get_map = None

        toset = op2.Set(1, comm=test.comm)
        dofset = op2.DataSet(toset, 1)
        arity = sum(m.arity*s.cdim
                    for m, s in zip(get_map(test),
                                    test.dof_dset))
        iterset = get_map(test).iterset
        entity_node_map = op2.Map(iterset,
                                  toset, arity,
                                  values=numpy.zeros(iterset.total_size*arity, dtype=IntType))
        dat = LocalDat(dofset, needs_mask=True)

        statedat = LocalDat(dofset)
        state_entity_node_map = op2.Map(iterset,
                                        toset, arity,
                                        values=numpy.zeros(iterset.total_size*arity, dtype=IntType))
        statearg = statedat(op2.READ, state_entity_node_map)

        arg = dat(op2.INC, entity_node_map)
        arg.position = 0
        args.append(arg)

        mesh = form.ufl_domains()[kinfo.domain_number]
        arg = mesh.coordinates.dat(op2.READ, get_map(mesh.coordinates))
        arg.position = 1
        args.append(arg)

        if kinfo.oriented:
            c = form.ufl_domain().cell_orientations()
            arg = c.dat(op2.READ, get_map(c))
            arg.position = len(args)
            args.append(arg)
        if kinfo.needs_cell_sizes:
            c = form.ufl_domain().cell_sizes
            arg = c.dat(op2.READ, get_map(c))
            arg.position = len(args)
            args.append(arg)
        for n in kinfo.coefficient_map:
            c = form.coefficients()[n]
            if c is state:
                statearg.position = len(args)
                args.append(statearg)
                continue
            for (i, c_) in enumerate(c.split()):
                map_ = get_map(c_)
                arg = c_.dat(op2.READ, map_)
                arg.position = len(args)
                args.append(arg)

        if kinfo.integral_type == "interior_facet":
            arg = test.ufl_domain().interior_facets.local_facet_dat(op2.READ)
            arg.position = len(args)
            args.append(arg)
        iterset = op2.Subset(iterset, [0])
        mod = seq.JITModule(kinfo.kernel, iterset, *args)
        kernels.append(CompiledKernel(mod._fun, kinfo))
    return cell_kernels, int_facet_kernels


# We need to set C function pointer callbacks for PCPatch to work.
# Although petsc4py provides a high-level Python wrapper for them,
# this is very costly when going back and forth from C to Python only
# to extract function pointers and send them straight back to C. Here,
# since we know what the calling convention of the C function is, we
# just wrap up everything as a C function pointer and use that
# directly.
def make_struct(op_coeffs, op_maps, jacobian=False):
    import ctypes
    coeffs = []
    maps = []
    for i, c in enumerate(op_coeffs):
        if c is None:
            coeffs.append("state")
        else:
            coeffs.append("c{}".format(i))
    for i, m in enumerate(op_maps):
        if m is None:
            maps.append("dofArrayWithAll")
        else:
            maps.append("m{}".format(i))
    coeff_struct = ";\n".join("  const PetscScalar *c{}".format(i) for i, c in enumerate(op_coeffs) if c is not None)
    map_struct = ";\n".join("  const PetscInt    *m{}".format(i) for i, m in enumerate(op_maps) if m is not None)
    coeff_decl = ", ".join("const PetscScalar *restrict {}".format(c) for c in coeffs)
    map_decl = ", ".join("const PetscInt *restrict {}".format(m) for m in maps)
    coeff_call = ", ".join(c if c == "state" else "ctx->{}".format(c) for c in coeffs)
    map_call = ", ".join(m if m == "dofArrayWithAll" else "ctx->{}".format(m) for m in maps)
    if jacobian:
        out = "Mat J"
    else:
        out = "PetscScalar * restrict F"
    function = "  void (*pyop2_call)(int start, int end, const PetscInt * restrict cells, {}, {}, const PetscInt *restrict dofArray, {})".format(out, coeff_decl, map_decl)

    fields = []
    for c in coeffs:
        if c != "state":
            fields.append((c, ctypes.c_voidp))
    for m in maps:
        if m != "dofArrayWithAll":
            fields.append((m, ctypes.c_voidp))
    fields.append(("point2facet", ctypes.c_voidp))
    fields.append(("pyop2_call", ctypes.c_voidp))

    class Struct(ctypes.Structure):
        _fields_ = fields
    struct = """
typedef struct {{
{};
{};
  const PetscInt    *point2facet;
{};
}} UserCtx;""".format(coeff_struct, map_struct, function)
    call = "pyop2_call(0, npoints, whichPoints, out, {}, dofArray, {})".format(coeff_call, map_call)

    return struct, call, Struct


def make_residual_wrapper(coeffs, maps):
    struct_decl, pyop2_call, struct = make_struct(coeffs, maps, jacobian=False)

    return """
#include <petsc.h>
{}
static PetscInt pointbuf[128];
PetscErrorCode ComputeResidual(PC pc,
                               PetscInt point,
                               Vec x,
                               Vec F,
                               IS points,
                               PetscInt ndof,
                               const PetscInt *dofArray,
                               const PetscInt *dofArrayWithAll,
                               void *ctx_)
{{
   const PetscScalar *state       = NULL;
   const PetscInt    *whichPoints = NULL;
   PetscScalar       *out         = NULL;
   UserCtx           *ctx         = (UserCtx *)ctx_;
   PetscInt           npoints;
   PetscErrorCode     ierr;
   PetscFunctionBeginUser;
   ierr = ISGetSize(points, &npoints);CHKERRQ(ierr);
   if (!npoints) PetscFunctionReturn(0);
   ierr = VecSet(F, 0.0);CHKERRQ(ierr);
   if (x) {{
     ierr = VecGetArrayRead(x, &state);CHKERRQ(ierr);
   }}
   ierr = VecGetArray(F, &out);CHKERRQ(ierr);
   ierr = ISGetIndices(points, &whichPoints);CHKERRQ(ierr);
   if (ctx->point2facet) {{
     PetscInt *pointsArray = NULL;
     if (npoints > 128) {{
       ierr = PetscMalloc1(npoints, &pointsArray);CHKERRQ(ierr);
     }} else {{
       pointsArray = pointbuf;
     }}
     for (PetscInt i = 0; i < npoints; i++) {{
       pointsArray[i] = ctx->point2facet[whichPoints[i]];
     }}
     ierr = ISRestoreIndices(points, &whichPoints);CHKERRQ(ierr);
     whichPoints = pointsArray;
   }}
   ctx->{};
   if (ctx->point2facet) {{
     if (npoints > 128) {{
       ierr = PetscFree(whichPoints);
     }}
   }} else {{
     ierr = ISRestoreIndices(points, &whichPoints);CHKERRQ(ierr);
   }}
   ierr = VecRestoreArray(F, &out);CHKERRQ(ierr);
   if (x) {{
     ierr = VecRestoreArrayRead(x, &state);CHKERRQ(ierr);
   }}
   PetscFunctionReturn(0);
}}
""".format(struct_decl, pyop2_call), struct


def make_jacobian_wrapper(coeffs, maps):
    struct_decl, pyop2_call, struct = make_struct(coeffs, maps, jacobian=True)

    return """
#include <petsc.h>
{}

static PetscInt pointbuf[128];
PetscErrorCode ComputeJacobian(PC pc,
                               PetscInt point,
                               Vec x,
                               Mat out,
                               IS points,
                               PetscInt ndof,
                               const PetscInt *dofArray,
                               const PetscInt *dofArrayWithAll,
                               void *ctx_)
{{
   const PetscScalar *state       = NULL;
   const PetscInt    *whichPoints = NULL;
   UserCtx           *ctx         = (UserCtx *)ctx_;
   PetscInt           npoints;
   PetscErrorCode     ierr;
   PetscFunctionBeginUser;
   ierr = ISGetSize(points, &npoints);CHKERRQ(ierr);
   if (!npoints) PetscFunctionReturn(0);
   if (x) {{
     ierr = VecGetArrayRead(x, &state);CHKERRQ(ierr);
   }}
   ierr = ISGetIndices(points, &whichPoints);CHKERRQ(ierr);
   if (ctx->point2facet) {{
     PetscInt *pointsArray = NULL;
     if (npoints > 128) {{
       ierr = PetscMalloc1(npoints, &pointsArray);CHKERRQ(ierr);
     }} else {{
       pointsArray = pointbuf;
     }}
     for (PetscInt i = 0; i < npoints; i++) {{
       pointsArray[i] = ctx->point2facet[whichPoints[i]];
     }}
     ierr = ISRestoreIndices(points, &whichPoints);CHKERRQ(ierr);
     whichPoints = pointsArray;
   }}
   ctx->{};
   if (ctx->point2facet) {{
     if (npoints > 128) {{
       ierr = PetscFree(whichPoints);
     }}
   }} else {{
     ierr = ISRestoreIndices(points, &whichPoints);CHKERRQ(ierr);
   }}
   if (x) {{
     ierr = VecRestoreArrayRead(x, &state);CHKERRQ(ierr);
   }}
   PetscFunctionReturn(0);
}}
""".format(struct_decl, pyop2_call), struct


def load_c_function(code, name, comm):
    cppargs = ["-I%s/include" % d for d in get_petsc_dir()]
    ldargs = (["-L%s/lib" % d for d in get_petsc_dir()]
              + ["-Wl,-rpath,%s/lib" % d for d in get_petsc_dir()]
              + ["-lpetsc", "-lm"])
    return load(code, "c", name,
                argtypes=[ctypes.c_voidp, ctypes.c_int, ctypes.c_voidp,
                          ctypes.c_voidp, ctypes.c_voidp, ctypes.c_int,
                          ctypes.c_voidp, ctypes.c_voidp, ctypes.c_voidp],
                restype=ctypes.c_int, cppargs=cppargs, ldargs=ldargs,
                comm=comm)


def make_c_arguments(form, kernel, state, get_map, require_state=False,
                     require_facet_number=False):
    coeffs = [form.ufl_domain().coordinates]
    if kernel.kinfo.oriented:
        coeffs.append(form.ufl_domain().cell_orientations())
    if kernel.kinfo.needs_cell_sizes:
        coeffs.append(form.ufl_domain().cell_sizes)
    for n in kernel.kinfo.coefficient_map:
        coeffs.append(form.coefficients()[n])
    if require_state:
        assert state in coeffs, "Couldn't find state vector in form coefficients"
    data_args = []
    map_args = []
    seen = set()
    for c in coeffs:
        if c is state:
            data_args.append(None)
            map_args.append(None)
        else:
            data_args.extend(c.dat._kernel_args_)
        map_ = get_map(c)
        if map_ is not None:
            for k in map_._kernel_args_:
                if k not in seen:
                    map_args.append(k)
                    seen.add(k)
    if require_facet_number:
        data_args.extend(form.ufl_domain().interior_facets.local_facet_dat._kernel_args_)
    return data_args, map_args


def make_c_struct(data_args, map_args, function, struct, point2facet=None):
    args = [a for a in chain(data_args, map_args) if a is not None]
    if point2facet is None:
        args.append(0)
    else:
        args.append(point2facet)
    return struct(*args, ctypes.cast(function, ctypes.c_voidp).value)


def bcdofs(bc, ghost=True):
    # Return the global dofs fixed by a DirichletBC
    # in the numbering given by concatenation of all the
    # subspaces of a mixed function space
    Z = bc.function_space()
    while Z.parent is not None:
        Z = Z.parent

    indices = bc._indices
    offset = 0

    for (i, idx) in enumerate(indices):
        if isinstance(Z.ufl_element(), VectorElement):
            offset += idx
            assert i == len(indices)-1  # assert we're at the end of the chain
            assert Z.sub(idx).value_size == 1
        elif isinstance(Z.ufl_element(), MixedElement):
            if ghost:
                offset += sum(Z.sub(j).dof_count for j in range(idx))
            else:
                offset += sum(Z.sub(j).dof_dset.size * Z.sub(j).value_size for j in range(idx))
        else:
            raise NotImplementedError("How are you taking a .sub?")

        Z = Z.sub(idx)

    if Z.parent is not None and isinstance(Z.parent.ufl_element(), VectorElement):
        bs = Z.parent.value_size
        start = 0
        stop = 1
    else:
        bs = Z.value_size
        start = 0
        stop = bs
    nodes = bc.nodes
    if not ghost:
        nodes = nodes[nodes < Z.dof_dset.size]

    return numpy.concatenate([nodes*bs + j for j in range(start, stop)]) + offset


def select_entity(p, dm=None, exclude=None):
    """Filter entities based on some label.

    :arg p: the entity.
    :arg dm: the DMPlex object to query for labels.
    :arg exclude: The label marking points to exclude."""
    if exclude is None:
        return True
    else:
        # If the exclude label marks this point (the value is not -1),
        # we don't want it.
        return dm.getLabelValue(exclude, p) == -1


class PlaneSmoother(object):
    @staticmethod
    def coords(dm, p, coordinates):
        coordinatesV = coordinates.function_space()
        data = coordinates.dat.data_ro_with_halos
        coordinatesDM = coordinatesV.dm
        coordinatesSection = coordinatesDM.getDefaultSection()

        closure_of_p = [x for x in dm.getTransitiveClosure(p, useCone=True)[0] if coordinatesSection.getDof(x) > 0]

        gdim = data.shape[1]
        bary = numpy.zeros(gdim)
        for p_ in closure_of_p:
            (dof, offset) = (coordinatesSection.getDof(p_), coordinatesSection.getOffset(p_))
            bary += data[offset:offset+dof].reshape(gdim)
        bary /= len(closure_of_p)
        return bary

    def sort_entities(self, dm, axis, dir, ndiv):
        # compute
        # [(pStart, (x, y, z)), (pEnd, (x, y, z))]

        mesh = dm.getAttr("__firedrake_mesh__")
        ele = mesh.coordinates.function_space().ufl_element()
        V = mesh.coordinates.function_space()
        if V.finat_element.entity_dofs() == V.finat_element.entity_closure_dofs():
            # We're using DG or DQ for our coordinates, so we got
            # a periodic mesh. We need to interpolate to CGk
            # with access descriptor MAX to define a consistent opinion
            # about where the vertices are.
            CGkele = ele.reconstruct(family="Lagrange")
            # Need to supply the actual mesh to the FunctionSpace constructor,
            # not its weakref proxy (the variable `mesh`)
            # as interpolation fails because they are not hashable
            CGk = FunctionSpace(mesh.coordinates.function_space().mesh(), CGkele)
            coordinates = interpolate(mesh.coordinates, CGk, access=op2.MAX)
        else:
            coordinates = mesh.coordinates

        select = partial(select_entity, dm=dm, exclude="pyop2_ghost")
        entities = [(p, self.coords(dm, p, coordinates)) for p in
                    filter(select, range(*dm.getChart()))]

        minx = min(entities, key=lambda z: z[1][axis])[1][axis]
        maxx = max(entities, key=lambda z: z[1][axis])[1][axis]

        def keyfunc(z):
            coords = tuple(z[1])
            return (coords[axis], ) + tuple(coords[:axis] + coords[axis+1:])

        s = sorted(entities, key=keyfunc, reverse=(dir == -1))

        divisions = numpy.linspace(minx, maxx, ndiv+1)
        (entities, coords) = zip(*s)
        coords = [c[axis] for c in coords]
        indices = numpy.searchsorted(coords[::dir], divisions)

        out = []
        for k in range(ndiv):
            out.append(entities[indices[k]:indices[k+1]])
        out.append(entities[indices[-1]:])

        return out

    def __call__(self, pc):
        if complex_mode:
            raise NotImplementedError("Sorry, plane smoothers not yet implemented in complex mode")
        dm = pc.getDM()
        prefix = pc.getOptionsPrefix()
        sentinel = object()
        sweeps = PETSc.Options(prefix).getString("pc_patch_construct_ps_sweeps", default=sentinel)
        if sweeps == sentinel:
            raise ValueError("Must set %spc_patch_construct_ps_sweeps" % prefix)

        patches = []
        for sweep in sweeps.split(':'):
            axis = int(sweep[0])
            dir = {'+': +1, '-': -1}[sweep[1]]
            ndiv = int(sweep[2:])

            entities = self.sort_entities(dm, axis, dir, ndiv)
            for patch in entities:
                iset = PETSc.IS().createGeneral(patch, comm=PETSc.COMM_SELF)
                patches.append(iset)

        iterationSet = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=PETSc.COMM_SELF)
        return (patches, iterationSet)


class PatchBase(PCSNESBase):

    def initialize(self, obj):

        if isinstance(obj, PETSc.PC):
            A, P = obj.getOperators()
        elif isinstance(obj, PETSc.SNES):
            A, P = obj.ksp.pc.getOperators()
        else:
            raise ValueError("Not a PC or SNES?")

        ctx = get_appctx(obj.getDM())
        if ctx is None:
            raise ValueError("No context found on form")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r", ctx)

        if P.getType() == "python":
            ictx = P.getPythonContext()
            if ictx is None:
                raise ValueError("No context found on matrix")
            if not isinstance(ictx, ImplicitMatrixContext):
                raise ValueError("Don't know how to get form from %r", ictx)
            J = ictx.a
            bcs = ictx.row_bcs
            if bcs != ictx.col_bcs:
                raise NotImplementedError("Row and column bcs must match")
        else:
            J = ctx.Jp or ctx.J
            bcs = ctx._problem.bcs

        mesh = J.ufl_domain()
        self.plex = mesh.topology_dm
        # We need to attach the mesh to the plex, so that
        # PlaneSmoothers (and any other user-customised patch
        # constructors) can use firedrake's opinion of what
        # the coordinates are, rather than plex's.
        self.plex.setAttr("__firedrake_mesh__", weakref.proxy(mesh))
        self.ctx = ctx

        if mesh.cell_set._extruded:
            raise NotImplementedError("Not implemented on extruded meshes")

        if "overlap_type" not in mesh._distribution_parameters:
            if mesh.comm.size > 1:
                # Want to do
                # warnings.warn("You almost surely want to set an overlap_type in your mesh's distribution_parameters.")
                # but doesn't warn!
                PETSc.Sys.Print("Warning: you almost surely want to set an overlap_type in your mesh's distribution_parameters.")

        patch = obj.__class__().create(comm=obj.comm)
        patch.setOptionsPrefix(obj.getOptionsPrefix() + "patch_")
        self.configure_patch(patch, obj)
        patch.setType("patch")

        if isinstance(obj, PETSc.SNES):
            Jstate = ctx._problem.u
            is_snes = True
        else:
            Jstate = None
            is_snes = False

        V, _ = map(operator.methodcaller("function_space"), J.arguments())

        if len(bcs) > 0:
            ghost_bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=True)
                                                             for bc in bcs]))
            global_bc_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False)
                                                              for bc in bcs]))
        else:
            ghost_bc_nodes = numpy.empty(0, dtype=PETSc.IntType)
            global_bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

        Jcell_kernels, Jint_facet_kernels = matrix_funptr(J, Jstate)
        Jcell_kernel, = Jcell_kernels
        Jop_data_args, Jop_map_args = make_c_arguments(J, Jcell_kernel, Jstate,
                                                       operator.methodcaller("cell_node_map"))
        code, Struct = make_jacobian_wrapper(Jop_data_args, Jop_map_args)
        Jop_function = load_c_function(code, "ComputeJacobian", obj.comm)
        Jop_struct = make_c_struct(Jop_data_args, Jop_map_args, Jcell_kernel.funptr, Struct)

        Jhas_int_facet_kernel = False
        if len(Jint_facet_kernels) > 0:
            Jint_facet_kernel, = Jint_facet_kernels
            Jhas_int_facet_kernel = True
            facet_Jop_data_args, facet_Jop_map_args = make_c_arguments(J, Jint_facet_kernel, Jstate,
                                                                       operator.methodcaller("interior_facet_node_map"),
                                                                       require_facet_number=True)
            code, Struct = make_jacobian_wrapper(facet_Jop_data_args, facet_Jop_map_args)
            facet_Jop_function = load_c_function(code, "ComputeJacobian", obj.comm)
            point2facet = J.ufl_domain().interior_facets.point2facetnumber.ctypes.data
            facet_Jop_struct = make_c_struct(facet_Jop_data_args, facet_Jop_map_args,
                                             Jint_facet_kernel.funptr, Struct,
                                             point2facet=point2facet)

        set_residual = hasattr(ctx, "F") and isinstance(obj, PETSc.SNES)
        if set_residual:
            F = ctx.F
            Fstate = ctx._problem.u
            Fcell_kernels, Fint_facet_kernels = residual_funptr(F, Fstate)

            Fcell_kernel, = Fcell_kernels

            Fop_data_args, Fop_map_args = make_c_arguments(F, Fcell_kernel, Fstate,
                                                           operator.methodcaller("cell_node_map"),
                                                           require_state=True)

            code, Struct = make_residual_wrapper(Fop_data_args, Fop_map_args)
            Fop_function = load_c_function(code, "ComputeResidual", obj.comm)
            Fop_struct = make_c_struct(Fop_data_args, Fop_map_args, Fcell_kernel.funptr, Struct)

            Fhas_int_facet_kernel = False
            if len(Fint_facet_kernels) > 0:
                Fint_facet_kernel, = Fint_facet_kernels
                Fhas_int_facet_kernel = True

                facet_Fop_data_args, facet_Fop_map_args = make_c_arguments(F, Fint_facet_kernel, Fstate,
                                                                           operator.methodcaller("interior_facet_node_map"),
                                                                           require_state=True,
                                                                           require_facet_number=True)
                code, Struct = make_jacobian_wrapper(facet_Fop_data_args, facet_Fop_map_args)
                facet_Fop_function = load_c_function(code, "ComputeResidual", obj.comm)
                point2facet = F.ufl_domain().interior_facets.point2facetnumber.ctypes.data
                facet_Fop_struct = make_c_struct(facet_Fop_data_args, facet_Fop_map_args,
                                                 Fint_facet_kernel.funptr, Struct,
                                                 point2facet=point2facet)

        patch.setDM(self.plex)
        patch.setPatchCellNumbering(mesh._cell_numbering)

        offsets = numpy.append([0], numpy.cumsum([W.dof_count
                                                  for W in V])).astype(PETSc.IntType)
        patch.setPatchDiscretisationInfo([W.dm for W in V],
                                         numpy.array([W.value_size for
                                                      W in V], dtype=PETSc.IntType),
                                         [W.cell_node_list for W in V],
                                         offsets,
                                         ghost_bc_nodes,
                                         global_bc_nodes)
        self.Jop_struct = Jop_struct
        set_patch_jacobian(patch, ctypes.cast(Jop_function, ctypes.c_voidp).value,
                           ctypes.addressof(Jop_struct), is_snes=is_snes)
        if Jhas_int_facet_kernel:
            self.facet_Jop_struct = facet_Jop_struct
            set_patch_jacobian(patch, ctypes.cast(facet_Jop_function, ctypes.c_voidp).value,
                               ctypes.addressof(facet_Jop_struct), is_snes=is_snes,
                               interior_facets=True)
        if set_residual:
            self.Fop_struct = Fop_struct
            set_patch_residual(patch, ctypes.cast(Fop_function, ctypes.c_voidp).value,
                               ctypes.addressof(Fop_struct), is_snes=is_snes)
            if Fhas_int_facet_kernel:
                set_patch_residual(patch, ctypes.cast(facet_Fop_function, ctypes.c_voidp).value,
                                   ctypes.addressof(facet_Fop_struct), is_snes=is_snes,
                                   interior_facets=True)

        patch.setPatchConstructType(PETSc.PC.PatchConstructType.PYTHON, operator=self.user_construction_op)
        patch.setAttr("ctx", ctx)
        patch.incrementTabLevel(1, parent=obj)
        patch.setFromOptions()
        patch.setUp()
        self.patch = patch

    def destroy(self, obj):
        # In this destructor we clean up the __firedrake_mesh__ we set on the plex.
        d = self.plex.getDict()
        try:
            del d["__firedrake_mesh__"]
        except KeyError:
            pass

    def user_construction_op(self, obj, *args, **kwargs):
        prefix = obj.getOptionsPrefix()
        sentinel = object()
        usercode = PETSc.Options(prefix).getString("%s_patch_construct_python_type" % self._objectname, default=sentinel)
        if usercode == sentinel:
            raise ValueError("Must set %s%s_patch_construct_python_type" % (prefix, self._objectname))

        (modname, funname) = usercode.rsplit('.', 1)
        mod = __import__(modname)
        fun = getattr(mod, funname)
        if isinstance(fun, type):
            fun = fun()
        return fun(obj, *args, **kwargs)

    def update(self, pc):
        self.patch.setUp()

    def view(self, pc, viewer=None):
        self.patch.view(viewer=viewer)


class PatchPC(PCBase, PatchBase):

    def configure_patch(self, patch, pc):
        (A, P) = pc.getOperators()
        patch.setOperators(A, P)

    def apply(self, pc, x, y):
        self.patch.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.patch.applyTranspose(x, y)


class PatchSNES(SNESBase, PatchBase):
    def configure_patch(self, patch, snes):
        patch.setTolerances(max_it=1)
        patch.setConvergenceTest("skip")

        (f, residual) = snes.getFunction()
        assert residual is not None
        (fun, args, kargs) = residual
        patch.setFunction(fun, f.duplicate(), args=args, kargs=kargs)

        # Need an empty RHS for the solve,
        # PCApply can't deal with RHS = NULL,
        # and this goes through a call to PCApply at some point
        self.dummy = f.duplicate()

    def step(self, snes, x, f, y):
        push_appctx(self.plex, self.ctx)
        x.copy(y)
        self.patch.solve(snes.vec_rhs or self.dummy, y)
        y.axpy(-1, x)
        y.scale(-1)
        snes.setConvergedReason(self.patch.getConvergedReason())
        pop_appctx(self.plex)

import abc
import collections
import dataclasses
import itertools
import textwrap
from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.preconditioners.asm import validate_overlap
from firedrake.petsc import PETSc
import firedrake.cython.patchimpl
from firedrake.solving_utils import _SNESContext
from firedrake.utils import complex_mode, IntType
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
from firedrake.interpolation import interpolate
from firedrake.tsfc_interface import compile_form
from firedrake.ufl_expr import extract_domains

import loopy as lp
from mpi4py import MPI

from collections import namedtuple
import operator
from itertools import chain
from functools import cached_property, partial
import numpy
from finat.ufl import VectorElement, MixedElement
from ufl.domain import extract_unique_domain
from tsfc.ufl_utils import extract_firedrake_constants
import weakref
import petsctools

import ctypes
import pyop2.compilation
from pyop2 import op2
import pyop2.types
from pyop2.codegen.builder import Pack, MatPack, DatPack
from pyop2.codegen.representation import Comparison, Literal
from pyop2.codegen.rep2loopy import register_petsc_function
from pyop2.global_kernel import compile_global_kernel
from pyop2.mpi import COMM_SELF
from pyop2.utils import get_petsc_dir

__all__ = ("PatchPC", "PlaneSmoother", "PatchSNES")


# We need to set C function pointer callbacks for PCPatch to work.
# Although petsc4py provides a high-level Python wrapper for them,
# this is very costly when going back and forth from C to Python only
# to extract function pointers and send them straight back to C. Here,
# since we know what the calling convention of the C function is, we
# just wrap up everything as a C function pointer and use that
# directly.

class PatchCallableGeneratorContext:
    def __init__(self, form, kernel):
        self.form = form
        self.kernel = kernel

        self.temp_name_counter = itertools.count()
        self.index_name_counter = itertools.count()
        coeff_counter = itertools.count()
        self.coeff_names = collections.defaultdict(lambda: f"coeff_{next(coeff_counter)}")
        map_counter = itertools.count()
        self.map_names = collections.defaultdict(lambda: f"map_{next(map_counter)}")

        self.coefficients = []
        self.maps = []
        self.pack_insns = []

        if len(form.arguments()) == 2:  # put in subclasses of jacpatch bits etc
            row_space, column_space = map(operator.methodcaller("function_space"), form.arguments())
            _, row_num_nodes = row_space.cell_node_list.shape
            row_num_dofs = numpy.prod(row_space.shape, dtype=int)
            _, column_num_nodes = column_space.cell_node_list.shape
            column_num_dofs = numpy.prod(column_space.shape, dtype=int)
            self.temp_decls = [
                f"PetscScalar out_local[{row_num_nodes}*{row_num_dofs}*{column_num_nodes}*{column_num_dofs}];"
            ]
            myindex = self._generate_index_name()
            pack_insn = f"""\
for (int32_t {myindex}=0; {myindex}<{row_num_nodes}*{row_num_dofs}*{column_num_nodes}*{column_num_dofs}; {myindex}++)
  out_local[{myindex}] = 0.0;
"""
            self.pack_insns.append(pack_insn)
        else:
            raise NotImplementedError

        self.local_kernel_args = ["out_local"]

        row_size = row_num_nodes * row_num_dofs
        column_size = column_num_nodes * column_num_dofs

        self.unpack_insn = f"MatSetValues(out, {row_size}, &(dofArray[{row_size}*i]), {column_size}, &(dofArray[{column_size}*i]), out_local, ADD_VALUES);"

        self._set_up()

    @property
    def local_kernel_call_insn(self) -> str:
        return f"{self.kernel.kinfo.kernel.name}({', '.join(self.local_kernel_args)});"

    def _generate_temp_name(self):
        return f"t_{next(self.temp_name_counter)}"

    def _generate_index_name(self):
        return f"i_{next(self.index_name_counter)}"

    def _set_up(self):
        all_meshes = extract_domains(self.form)
        kinfo = self.kernel.kinfo

        for i in kinfo.active_domain_numbers.coordinates:
            self._add_coefficient(all_meshes[i].coordinates)
        for i in kinfo.active_domain_numbers.cell_orientations:
            raise NotImplementedError
            # c = all_meshes[i].cell_orientations()
        for i in kinfo.active_domain_numbers.cell_sizes:
            raise NotImplementedError
            # c = all_meshes[i].cell_sizes
        for n, indices in kinfo.coefficient_numbers:
            coeff = self.form.coefficients()[n]
            # if c is state:
            #     raise NotImplementedError
            #     if indices != (0, ):
            #         raise ValueError(f"Active indices of state (dont_split) function must be (0, ), not {indices}")
            #     args.append(statearg)
            #     continue
            for i in indices:
                coeff_ = c.subfunctions[i]
                self._add_coefficient(coeff_)

        # all_constants = extract_firedrake_constants(form)
        # for constant_index in kinfo.constant_numbers:
        #     args.append(all_constants[constant_index].dat)
        #
        # if kinfo.integral_type == "interior_facet":
        #     arg = mesh.interior_facets.local_facet_dat
        #     args.append(arg)

    def _add_coefficient(self, coeff):
        space = coeff.function_space()

        _, num_nodes = space.cell_node_list.shape
        num_dofs_per_node = numpy.prod(space.shape, dtype=int)

        temp_name = self._generate_temp_name()
        coeff_name = self.coeff_names[coeff]

        self.coefficients.append(coeff)

        if space.finat_element not in self.map_names:
            assert self.kernel.kinfo.integral_type == "cell"
            self.maps.append(space.cell_node_map())  # FIXME: what if not a cell loop
        map_name = self.map_names[space.finat_element]
        node_index = self._generate_index_name()
        dof_index = self._generate_index_name()

        temp_decl = f"PetscScalar {temp_name}[{num_nodes}*{num_dofs_per_node}];"
        self.temp_decls.append(temp_decl)

        pack_insn = f"""\
for (int32_t {node_index}=0; {node_index}<{num_nodes}; {node_index}++)
  for (int32_t {dof_index}=0; {dof_index}<{num_dofs_per_node}; {dof_index}++)
    {temp_name}[{num_dofs_per_node}*{node_index}+{dof_index}] = {coeff_name}[{map_name}[p*{num_nodes}+{node_index}]*{num_dofs_per_node}+{dof_index}];"""
        self.pack_insns.append(pack_insn)

        self.local_kernel_args.append(temp_name)


@dataclasses.dataclass(frozen=True)
class PatchCallable(abc.ABC):
    context: PatchCallableGeneratorContext

    @cached_property
    def ctypes_callable(self):
        cppargs = petsctools.get_petsc_dirs(prefix="-I", subdir="include")
        ldargs = [
            *(petsctools.get_petsc_dirs(prefix="-L", subdir="lib")),
            *(petsctools.get_petsc_dirs(prefix="-Wl,-rpath,", subdir="lib")),
            "-lpetsc",
            "-lm",
        ]
        comm = self.context.form.arguments()[0].function_space().comm
        dll = pyop2.compilation.load(
            self._callback_code, "c", cppargs=cppargs, ldargs=ldargs, comm=comm
        )
        fn = getattr(dll, self._callback_name)
        fn.argtypes = [
            ctypes.c_voidp,  # PC pc
            ctypes.c_int,    # PetscInt point
            ctypes.c_voidp,  # Vec x
            ctypes.c_voidp,  # Mat J / Vec F
            ctypes.c_voidp,  # IS points
            ctypes.c_int,    # PetscInt ndof
            ctypes.c_voidp,  # const PetscInt *dofArray
            ctypes.c_voidp,  # const PetscInt *dofArrayWithAll
            ctypes.c_voidp,  # void *ctx_
        ]
        fn.restype = ctypes.c_int
        return ctypes.cast(fn, ctypes.c_voidp).value

    @cached_property
    def ctypes_struct_address(self):
        return ctypes.addressof(self._ctypes_struct)

    @cached_property
    def _coeff_names(self):
        return tuple(self.context.coeff_names.values())

    @cached_property
    def _coefficients(self):
        return tuple(self.context.coefficients)

    @cached_property
    def _map_names(self):
        return tuple(self.context.map_names.values())

    @cached_property
    def _maps(self):
        return tuple(self.context.maps)

    @property
    @abc.abstractmethod
    def _callback_code(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def _callback_name(self) -> str:
        pass

    @cached_property
    def _local_kernel_code(self) -> str:
        return lp.generate_code_v2(self.context.kernel.kinfo.kernel.code).device_code()

    @property
    @abc.abstractmethod
    def _wrapper_kernel_args_sig(self) -> str:
        pass

    def _make_wrapper_kernel_args_sig(self, out_type: str) -> str:
        coeff_sigs = ", ".join(
            f"const PetscScalar *restrict {coeff_name}" for coeff_name in self._coeff_names
        )
        map_sigs = ", ".join(
            f"const PetscInt *restrict {map_name}" for map_name in self._map_names
        )
        return (
            f"PetscInt n, const PetscInt * restrict points, {out_type}, {coeff_sigs}, const PetscInt *restrict dofArray, {map_sigs}"
        )


    @property
    def _wrapper_kernel_code(self) -> str:
        return f"""\
void wrapper_kernel({self._wrapper_kernel_args_sig})
{{
{'\n'.join(textwrap.indent(temp_decl, " "*2) for temp_decl in self.context.temp_decls)}
  PetscInt p;

  for (int32_t i = 0; i < n; i++)
  {{
    p = points[i];
{'\n'.join(textwrap.indent(pack_insn, " "*4) for pack_insn in self.context.pack_insns)}

    {self.context.local_kernel_call_insn}

    {self.context.unpack_insn}
  }}
}}
"""

    @property
    def _wrapper_kernel_call_insn(self) -> str:
        coeff_args = ", ".join(f"ctx->{coeff_name}" for coeff_name in self._coeff_names)
        map_args = ", ".join(f"ctx->{map_name}" for map_name in self._map_names)
        return f"wrapper_kernel(npoints, whichPoints, out, {coeff_args}, dofArray, {map_args})"

    @property
    def _local_kernel_num_flops(self) -> int:
        return self.context.kernel.kinfo.kernel.num_flops

    @cached_property
    def _ctypes_struct(self) -> ctypes.Structure:
        fields = [
            *((coeff_name, ctypes.c_voidp) for coeff_name in self._coeff_names),
            *((map_name, ctypes.c_voidp) for map_name in self._map_names),
            ("point2facet", ctypes.c_voidp),
        ]

        class Struct(ctypes.Structure):
            _fields_ = fields

        struct_args = [
            *(arg for coeff in self._coefficients for arg in coeff.dat._kernel_args_),
            *(arg for map_ in self._maps for arg in map_._kernel_args_),
            0,  # point2facet (for now)
        ]
        return Struct(*struct_args)

    @cached_property
    def _struct_code(self) -> str:
        coeff_decls = ";\n".join(
            f"const PetscScalar *{coeff_name}" for coeff_name in self._coeff_names
        )
        map_decls = ";\n".join(
            f"const PetscInt *{map_name}" for map_name in self._map_names
        )

        return f"""\
typedef struct {{
{textwrap.indent(coeff_decls, "  ")};
{textwrap.indent(map_decls, "  ")};
  const PetscInt *point2facet;
}} UserCtx;"""


class JacobianPatchCallable(PatchCallable):

    _callback_name = "ComputeJacobian"

    @property
    def _wrapper_kernel_args_sig(self) -> str:
        return self._make_wrapper_kernel_args_sig("Mat out")

    @cached_property
    def _callback_code(self) -> str:
        return f"""\
#include <petsc.h>

{self._local_kernel_code}

{self._wrapper_kernel_code}

{self._struct_code}

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

   PetscFunctionBeginUser;
   PetscCall(ISGetSize(points, &npoints));;
   if (!npoints) PetscFunctionReturn(0);
   if (x) {{
     PetscCall(VecGetArrayRead(x, &state));
   }}
   PetscCall(ISGetIndices(points, &whichPoints));
   if (ctx->point2facet) {{
     PetscInt *pointsArray = NULL;
     if (npoints > 128) {{
       PetscCall(PetscMalloc1(npoints, &pointsArray));
     }} else {{
       pointsArray = pointbuf;
     }}
     for (PetscInt i = 0; i < npoints; i++) {{
       pointsArray[i] = ctx->point2facet[whichPoints[i]];
     }}
     PetscCall(ISRestoreIndices(points, &whichPoints));
     whichPoints = pointsArray;
   }}

   {self._wrapper_kernel_call_insn};

   if (ctx->point2facet) {{
     if (npoints > 128) {{
       PetscCall(PetscFree(whichPoints));
     }}
   }} else {{
     PetscCall(ISRestoreIndices(points, &whichPoints));
   }}
   if (x) {{
     PetscCall(VecRestoreArrayRead(x, &state));
   }}
   PetscLogFlops({self._local_kernel_num_flops} * npoints);
   PetscFunctionReturn(0);
}}"""

class ResidualPatchCallable(PatchCallable):
    @property
    def _wrapper_kernel_args_sig(self) -> str:
        return self._make_wrapper_kernel_args_sig("PetscScalar * restrict out")


def generate_patch_callable(form, kernel) -> PatchCallable:
    ctx = PatchCallableGeneratorContext(form, kernel)

    if len(form.arguments()) == 2:
        return JacobianPatchCallable(ctx)
    else:
        assert len(form.arguments()) == 1
        return ResidualPatchCallable(ctx)


def get_map(V, base_mesh, base_integral_type):
    return V.topological.entity_node_map(base_mesh.topology, base_integral_type, None, None)


# TODO: dont need to differentiate between callable types here
def make_jacobian_callables(form, state):
    test, trial = map(operator.methodcaller("function_space"), form.arguments())
    if test != trial:
        raise NotImplementedError("Only for matching test and trial spaces")

    if state is not None:
        dont_split = (state,)
    else:
        dont_split = ()
    kernels = compile_form(form, "subspace_form", split=False, dont_split=dont_split)
    cell_callable = None
    interior_facet_callable = None
    for kernel in kernels:
        kinfo = kernel.kinfo
        if kinfo.subdomain_id != ("otherwise",):
            raise NotImplementedError("Only for full domain integrals")
        if kinfo.integral_type not in {"cell", "interior_facet"}:
            raise NotImplementedError("Only for cell or interior facet integrals")

        callable = generate_patch_callable(form, kernel)
        if kinfo.integral_type == "cell":
            assert cell_callable is None, "Only a single cell callable allowed"
            cell_callable = callable
        else:
            assert kinfo.integral_type == "interior_facet"
            assert interior_facet_callable is None, "Only a single interior facet callable allowed"
            interior_facet_callable = callable
    return cell_callable, interior_facet_callable


def residual_funptr(form, state):
    from firedrake.tsfc_interface import compile_form
    test, = map(operator.methodcaller("function_space"), form.arguments())

    if state.function_space() != test:
        raise NotImplementedError("State and test space must be dual to one-another")

    if state is not None:
        dont_split = (state, )
    else:
        dont_split = ()

    kernels = compile_form(form, "subspace_form", split=False, dont_split=dont_split)

    all_meshes = extract_domains(form)
    cell_kernels = []
    int_facet_kernels = []
    for kernel in kernels:
        kinfo = kernel.kinfo
        mesh = all_meshes[kinfo.domain_number]  # integration domain

        if kinfo.subdomain_id != ("otherwise",):
            raise NotImplementedError("Only for full domain integrals")
        if kinfo.integral_type not in {"cell", "interior_facet"}:
            raise NotImplementedError("Only for cell integrals or interior_facet integrals")
        args = []

        if kinfo.integral_type == "cell":
            kernels = cell_kernels
        elif kinfo.integral_type == "interior_facet":
            kernels = int_facet_kernels

        toset = op2.Set(1, comm=test.comm)
        dofset = op2.DataSet(toset, 1)
        arity = sum(m.arity*s.cdim
                    for m, s in zip(get_map(test, mesh, integral_type),
                                    test.dof_dset))
        iterset = get_map(test, mesh, integral_type).iterset
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
        args.append(arg)

        for i in kinfo.active_domain_numbers.coordinates:
            c = all_meshes[i].coordinates
            arg = c.dat(op2.READ, get_map(c.function_space(), mesh, integral_type))
            args.append(arg)
        for i in kinfo.active_domain_numbers.cell_orientations:
            c = all_meshes[i].cell_orientations()
            arg = c.dat(op2.READ, get_map(c.function_space(), mesh, integral_type))
            args.append(arg)
        for i in kinfo.active_domain_numbers.cell_sizes:
            c = all_meshes[i].cell_sizes
            arg = c.dat(op2.READ, get_map(c.function_space(), mesh, integral_type))
            args.append(arg)
        for n, indices in kinfo.coefficient_numbers:
            c = form.coefficients()[n]
            if c is state:
                if indices != (0, ):
                    raise ValueError(f"Active indices of state (dont_split) function must be (0, ), not {indices}")
                args.append(statearg)
                continue
            for ind in indices:
                c_ = c.subfunctions[ind]
                map_ = get_map(c_.function_space(), mesh, integral_type)
                arg = c_.dat(op2.READ, map_)
                args.append(arg)

        all_constants = extract_firedrake_constants(form)
        for constant_index in kinfo.constant_numbers:
            args.append(all_constants[constant_index].dat(op2.READ))

        if kinfo.integral_type == "interior_facet":
            arg = extract_unique_domain(test).interior_facets.local_facet_dat(op2.READ)
            args.append(arg)
        iterset = op2.Subset(iterset, [])

        wrapper_knl_args = tuple(a.global_kernel_arg for a in args)
        mod = op2.GlobalKernel(kinfo.kernel, wrapper_knl_args, subset=True)
        kernels.append(CompiledKernel(compile_global_kernel(mod, iterset.comm), kinfo))
    return cell_kernels, int_facet_kernels


def make_residual_wrapper(coeffs, maps, flops):
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
   PetscLogFlops({} * npoints);
   PetscFunctionReturn(0);
}}
""".format(struct_decl, pyop2_call, flops), struct


def make_c_arguments(form, kernel, state, integral_type, require_state=False,
                     require_facet_number=False):
    all_meshes = extract_domains(form)
    mesh = all_meshes[kernel.kinfo.domain_number]
    coeffs = []
    coeffs.extend([all_meshes[i].coordinates for i in kernel.kinfo.active_domain_numbers.coordinates])
    coeffs.extend([all_meshes[i].cell_orientations() for i in kernel.kinfo.active_domain_numbers.cell_orientations])
    coeffs.extend([all_meshes[i].cell_sizes for i in kernel.kinfo.active_domain_numbers.cell_sizes])
    for n, indices in kernel.kinfo.coefficient_numbers:
        c = form.coefficients()[n]
        if c is state:
            if indices != (0, ):
                raise ValueError(f"Active indices of state (dont_split) function must be (0, ), not {indices}")
            coeffs.append(c)
        else:
            coeffs.extend([c.subfunctions[ind] for ind in indices])
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
        map_ = get_map(c.function_space(), mesh, integral_type)
        if map_ is not None:
            for k in map_._kernel_args_:
                if k not in seen:
                    map_args.append(k)
                    seen.add(k)

    all_constants = extract_firedrake_constants(form)
    for constant_index in kernel.kinfo.constant_numbers:
        data_args.extend(all_constants[constant_index].dat._kernel_args_)

    if require_facet_number:
        data_args.extend(mesh.interior_facets.local_facet_dat._kernel_args_)
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
            assert Z.sub(idx).block_size == 1
        elif isinstance(Z.ufl_element(), MixedElement):
            if ghost:
                offset += sum(Z.sub(j).dof_count for j in range(idx))
            else:
                offset += sum(Z.sub(j).dof_dset.size * Z.sub(j).block_size for j in range(idx))
        else:
            raise NotImplementedError("How are you taking a .sub?")

        Z = Z.sub(idx)

    if Z.parent is not None and isinstance(Z.parent.ufl_element(), VectorElement):
        bs = Z.parent.block_size
        start = 0
        stop = 1
    else:
        bs = Z.block_size
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
        ndof = 0
        for p_ in closure_of_p:
            (dof, offset) = (coordinatesSection.getDof(p_), coordinatesSection.getOffset(p_))
            bary += data[offset:offset + dof].reshape(dof, gdim).sum(axis=0)
            ndof += dof
        bary /= ndof
        return bary

    def sort_entities(self, dm, axis, dir, ndiv=None, divisions=None):
        # compute
        # [(pStart, (x, y, z)), (pEnd, (x, y, z))]
        from firedrake.assemble import assemble

        if ndiv is None and divisions is None:
            raise RuntimeError("Must either set ndiv or divisions for PlaneSmoother!")

        mesh = dm.getAttr("__firedrake_mesh__")
        if len(set(mesh)) == 1:
            mesh_unique = mesh.unique()
        else:
            raise NotImplementedError("Not implemented for general mixed meshes")
        coordinates = mesh_unique.coordinates
        V = coordinates.function_space()
        if V.finat_element.is_dg():
            # We're using DG or DQ for our coordinates, so we got
            # a periodic mesh. We need to interpolate to CGk
            # with access descriptor MAX to define a consistent opinion
            # about where the vertices are.
            CGk = V.reconstruct(family="Lagrange")
            coordinates = assemble(interpolate(coordinates, CGk, access=op2.MAX))

        select = partial(select_entity, dm=dm, exclude="pyop2_ghost")
        entities = [(p, self.coords(dm, p, coordinates)) for p in
                    filter(select, range(*dm.getChart()))]

        if isinstance(axis, int):
            minx = min(entities, key=lambda z: z[1][axis])[1][axis]
            maxx = max(entities, key=lambda z: z[1][axis])[1][axis]

            def keyfunc(z):
                coords = tuple(z[1])
                return (coords[axis], ) + tuple(coords[:axis] + coords[axis+1:])
        else:
            minx = axis(min(entities, key=lambda z: axis(z[1]))[1])
            maxx = axis(max(entities, key=lambda z: axis(z[1]))[1])

            def keyfunc(z):
                coords = tuple(z[1])
                return (axis(coords), ) + coords

        s = sorted(entities, key=keyfunc, reverse=(dir == -1))
        (entities, coords) = zip(*s)
        if isinstance(axis, int):
            coords = [c[axis] for c in coords]
        else:
            coords = [axis(c) for c in coords]

        if divisions is None:
            divisions = numpy.linspace(minx, maxx, ndiv+1)
        if ndiv is None:
            ndiv = numpy.size(divisions)-1
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
        context = dm.getAttr("__firedrake_ctx__")
        prefix = pc.getOptionsPrefix() or ""
        sentinel = object()
        sweeps = PETSc.Options(prefix).getString("pc_patch_construct_ps_sweeps", default=sentinel)
        if sweeps == sentinel:
            raise ValueError("Must set %spc_patch_construct_ps_sweeps" % prefix)

        patches = []
        import re
        for sweep in sweeps.split(':'):
            sweep_split = re.split(r'([+-])', sweep)
            try:
                axis = int(sweep_split[0])
            except ValueError:
                try:
                    axis = context.appctx[sweep_split[0]]
                except KeyError:
                    raise KeyError("PlaneSmoother axis key %s not provided" % sweep_split[0])

            dir = {'+': +1, '-': -1}[sweep_split[1]]
            # Either use equispaced bins for relaxation or get from appctx
            try:
                ndiv = int(sweep_split[2])
                entities = self.sort_entities(dm, axis, dir, ndiv=ndiv)
            except ValueError:
                try:
                    divisions = context.appctx[sweep_split[2]]
                    entities = self.sort_entities(dm, axis, dir, divisions=divisions)
                except KeyError:
                    raise KeyError("PlaneSmoother division key %s not provided" % sweep_split[2:])

            for patch in entities:
                if not patch:
                    continue
                else:
                    iset = PETSc.IS().createGeneral(patch, comm=COMM_SELF)
                    patches.append(iset)

        iterationSet = PETSc.IS().createStride(size=len(patches), first=0, step=1, comm=COMM_SELF)
        return (patches, iterationSet)


class PatchBase(PCSNESBase):

    def initialize(self, obj):

        ctx = get_appctx(obj.getDM())
        if ctx is None:
            raise ValueError("No context found on form")
        if not isinstance(ctx, _SNESContext):
            raise ValueError("Don't know how to get form from %r" % ctx)

        J, bcs = self.form(obj)
        V = J.arguments()[0].function_space()
        mesh = V.mesh()
        if len(set(mesh)) == 1:
            mesh_unique = mesh.unique()
        else:
            raise NotImplementedError("Not implemented for general mixed meshes")
        self.plex = mesh_unique.topology_dm
        # We need to attach the mesh and appctx to the plex, so that
        # PlaneSmoothers (and any other user-customised patch
        # constructors) can use firedrake's opinion of what
        # the coordinates are, rather than plex's.
        self.plex.setAttr("__firedrake_mesh__", weakref.proxy(mesh))
        self.ctx = ctx
        self.plex.setAttr("__firedrake_ctx__", weakref.proxy(ctx))

        if mesh_unique.cell_set._extruded:
            raise NotImplementedError("Not implemented on extruded meshes")

        # Validate the mesh overlap
        prefix = (obj.getOptionsPrefix() or "") + "patch_"
        opts = PETSc.Options(prefix)
        petsc_prefix = self._petsc_prefix
        patch_type = opts.getString(f"{petsc_prefix}construct_type")
        patch_dim = opts.getInt(f"{petsc_prefix}construct_dim", -1)
        patch_codim = opts.getInt(f"{petsc_prefix}construct_codim", -1)
        if patch_dim != -1:
            assert patch_codim == -1, "Cannot set both dim and codim"
        elif patch_codim != -1:
            assert patch_dim == -1, "Cannot set both dim and codim"
            patch_dim = self.plex.getDimension() - patch_codim
        else:
            patch_dim = 0
        validate_overlap(mesh_unique, patch_dim, patch_type)

        patch = obj.__class__().create(comm=mesh.comm)
        patch.setOptionsPrefix(prefix)
        self.configure_patch(patch, obj)
        patch.setType("patch")

        if isinstance(obj, PETSc.SNES):
            Jstate = ctx._problem.u
            is_snes = True
        else:
            Jstate = None
            is_snes = False

        if len(bcs) > 0:
            ghost_bc_nodes = numpy.unique(
                numpy.concatenate([bcdofs(bc, ghost=True) for bc in bcs],
                                  dtype=PETSc.IntType)
            )
            global_bc_nodes = numpy.unique(
                numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs],
                                  dtype=PETSc.IntType))
        else:
            ghost_bc_nodes = numpy.empty(0, dtype=PETSc.IntType)
            global_bc_nodes = numpy.empty(0, dtype=PETSc.IntType)

        (
            jacobian_cell_callable,
            jacobian_interior_facet_callable,
        ) = make_jacobian_callables(J, Jstate)

        # self.Jop_struct = Jop_struct  # why?
        if is_snes:
            firedrake.cython.patchimpl.snespatch_set_compute_operator(
                patch,
                jacobian_cell_callable.ctypes_callable,
                jacobian_cell_callable.ctypes_struct_address,
            )
        else:
            firedrake.cython.patchimpl.pcpatch_set_compute_operator(
                patch,
                jacobian_cell_callable.ctypes_callable,
                jacobian_cell_callable.ctypes_struct_address,
            )

        if jacobian_interior_facet_callable is not None:
            raise NotImplementedError

        # Jcell_kernels, Jint_facet_kernels = matrix_funptr(J, Jstate)
        # Jcell_kernel, = Jcell_kernels
        # Jcell_flops = Jcell_kernel.kinfo.kernel.num_flops
        # # Jop_data_args, Jop_map_args = make_c_arguments(J, Jcell_kernel, Jstate,
        # #                                                operator.methodcaller("cell_node_map"))
        # code, Struct = make_jacobian_wrapper(Jcell_kernel, Jcell_flops)
        # Jop_function = load_c_function(code, "ComputeJacobian", mesh.comm)
        # Jop_struct = make_c_struct(Jcell_kernel.funptr, Struct)
        #
        # Jhas_int_facet_kernel = False
        # if len(Jint_facet_kernels) > 0:
        #     raise NotImplementedError
        #     Jint_facet_kernel, = Jint_facet_kernels
        #     Jhas_int_facet_kernel = True
        #     Jint_facet_flops = Jint_facet_kernel.kinfo.kernel.num_flops
        #     facet_Jop_data_args, facet_Jop_map_args = make_c_arguments(J, Jint_facet_kernel, Jstate,
        #                                                                "interior_facet",
        #                                                                require_facet_number=True)
        #     code, Struct = make_jacobian_wrapper(facet_Jop_data_args, facet_Jop_map_args, Jint_facet_flops)
        #     facet_Jop_function = load_c_function(code, "ComputeJacobian", mesh.comm)
        #     point2facet = mesh_unique.interior_facets.point2facetnumber.ctypes.data
        #     facet_Jop_struct = make_c_struct(facet_Jop_data_args, facet_Jop_map_args,
        #                                      Jint_facet_kernel.funptr, Struct,
        #                                      point2facet=point2facet)

        set_residual = hasattr(ctx, "F") and isinstance(obj, PETSc.SNES)
        if set_residual:
            raise NotImplementedError
            F = ctx.F
            Fstate = ctx._problem.u
            Fcell_kernels, Fint_facet_kernels = residual_funptr(F, Fstate)

            Fcell_kernel, = Fcell_kernels
            Fcell_flops = Fcell_kernel.kinfo.kernel.num_flops
            Fop_data_args, Fop_map_args = make_c_arguments(F, Fcell_kernel, Fstate,
                                                           "cell",
                                                           require_state=True)
            code, Struct = make_residual_wrapper(Fop_data_args, Fop_map_args, Fcell_flops)
            Fop_function = load_c_function(code, "ComputeResidual", mesh.comm)
            Fop_struct = make_c_struct(Fop_data_args, Fop_map_args, Fcell_kernel.funptr, Struct)

            Fhas_int_facet_kernel = False
            if len(Fint_facet_kernels) > 0:
                Fint_facet_kernel, = Fint_facet_kernels
                Fhas_int_facet_kernel = True
                Fint_facet_flops = Fint_facet_kernel.kinfo.kernel.num_flops
                facet_Fop_data_args, facet_Fop_map_args = make_c_arguments(F, Fint_facet_kernel, Fstate,
                                                                           "interior_facet",
                                                                           require_state=True,
                                                                           require_facet_number=True)
                code, Struct = make_jacobian_wrapper(facet_Fop_data_args, facet_Fop_map_args, Fint_facet_flops)
                facet_Fop_function = load_c_function(code, "ComputeResidual", mesh.comm)
                point2facet = extract_unique_domain(F).interior_facets.point2facetnumber.ctypes.data
                facet_Fop_struct = make_c_struct(facet_Fop_data_args, facet_Fop_map_args,
                                                 Fint_facet_kernel.funptr, Struct,
                                                 point2facet=point2facet)

        patch.setDM(self.plex)
        patch.setPatchCellNumbering(mesh_unique._cell_numbering)

        offsets = numpy.append([0], numpy.cumsum([W.dof_count
                                                  for W in V])).astype(PETSc.IntType)
        patch.setPatchDiscretisationInfo([W.dm for W in V],
                                         numpy.array([W.block_size for
                                                      W in V], dtype=PETSc.IntType),
                                         [W.cell_node_list for W in V],
                                         offsets,
                                         ghost_bc_nodes,
                                         global_bc_nodes)
        # if Jhas_int_facet_kernel:
        #     raise NotImplementedError
        #     self.facet_Jop_struct = facet_Jop_struct
        #     set_patch_jacobian(patch, ctypes.cast(facet_Jop_function, ctypes.c_voidp).value,
        #                        ctypes.addressof(facet_Jop_struct), is_snes=is_snes,
        #                        interior_facets=True)
        if set_residual:
            raise NotImplementedError
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
        self.patch = patch  # why?

    def destroy(self, obj):
        # In this destructor we clean up the __firedrake_mesh__ we set
        # on the plex and the context we set on the patch object.
        # We have to check if these attributes are available because
        # the destroy function will be called by petsc4py when
        # PCPythonSetContext is called (which occurs before
        # initialize).
        if hasattr(self, "plex"):
            d = self.plex.getDict()
            try:
                del d["__firedrake_mesh__"]
            except KeyError:
                pass
        if hasattr(self, "patch"):
            try:
                del self.patch.getDict()["ctx"]
            except KeyError:
                pass
            self.patch.destroy()

    def user_construction_op(self, obj, *args, **kwargs):
        prefix = obj.getOptionsPrefix() or ""
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

    _petsc_prefix = "pc_patch_"

    def configure_patch(self, patch, pc):
        (A, P) = pc.getOperators()
        patch.setOperators(A, P)

    def apply(self, pc, x, y):
        self.patch.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.patch.applyTranspose(x, y)


class PatchSNES(SNESBase, PatchBase):

    _petsc_prefix = "snes_patch_"

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

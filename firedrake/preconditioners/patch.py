from __future__ import annotations

import itertools
import textwrap
import typing
from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.preconditioners.asm import validate_overlap
from firedrake.petsc import PETSc
import firedrake.cython.patchimpl
from firedrake.solving_utils import _SNESContext
from firedrake.utils import complex_mode
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx
from firedrake.interpolation import interpolate
from firedrake.tsfc_interface import compile_form, KernelInfo
from firedrake.ufl_expr import extract_domains
from pyop2.datatypes import as_cstr

import loopy as lp

import ufl

import operator
from functools import cached_property, partial
import numpy
from finat.ufl import VectorElement, MixedElement
from tsfc.ufl_utils import extract_firedrake_constants
import weakref
import petsctools

import ctypes
import pyop2.compilation
from pyop2 import op2
import pyop2.types
from pyop2.mpi import COMM_SELF

if typing.TYPE_CHECKING:
    from firedrake import Function


__all__ = ("PatchPC", "PlaneSmoother", "PatchSNES")


class PatchCallable:
    """Class representing the evaluation of a patch operator or residual.

    When we set the callbacks for PCPatch/SNESPatch, we have to pass a function
    pointer that executes a patch-wise parloop, along with a struct that contains
    the additional coefficients and cell-node maps that the parloop needs. Given
    an input local kernel, this class coordinates the generation of these objects
    (called `ctypes_callable` and `ctypes_struct_address`).

    """
    def __init__(self, form: ufl.Form, kinfo: KernelInfo, state: Function):
        self.form = form
        self.kinfo = kinfo
        self.state = state

        args, names, state_index = self._set_up()
        self._args = args
        self._names = names
        self._state_index = state_index

    @cached_property
    def ctypes_callable(self):
        """Pointer to the compiled evaluation callback function.

        This is the function passed to 'PCPatchSetComputeOperator' and friends.

        """
        cppargs = petsctools.get_petsc_dirs(prefix="-I", subdir="include")
        ldargs = [
            *(petsctools.get_petsc_dirs(prefix="-L", subdir="lib")),
            *(petsctools.get_petsc_dirs(prefix="-Wl,-rpath,", subdir="lib")),
            "-lpetsc",
            "-lm",
        ]
        comm = self.form.arguments()[0].function_space().comm
        dll = pyop2.compilation.load(
            self._callback_code, "c", cppargs=cppargs, ldargs=ldargs, comm=comm
        )
        callback_name = "ComputeJacobian" if len(self.form.arguments()) == 2 else "ComputeResidual"
        fn = getattr(dll, callback_name)
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
        """Pointer to the 'context' struct passed to the callback function.

        This is passed to 'PCPatchSetComputeOperator' and friends.

        """
        return ctypes.addressof(self._ctypes_struct)

    def _set_up(self) -> tuple[
        list[tuple[op2.Dat, op2.Map | None] | op2.Global | op2.Constant],
        dict[op2.Dat | op2.Map | op2.Global | op2.Constant, str],
        int | None,
    ]:
        """Process ``form``, ``kinfo`` and ``state``.

        Returns
        -------
        args
            List of PyOP2 objects that are used in the wrapper kernel. The
            order matches the order that arguments are passed into the local
            kernel. The output tensor and optional state dat are not included.
            Dats are included as a 2-tuple of ``(dat, map)`` where ``map``
            can be `None`.
        names
            Mapping from PyOP2 objects to their names in the wrapper kernel.
        state_index
            Index of the state coefficient in the local kernel. `None` if
            state is not provided.

        """
        args: list[tuple[op2.Dat, op2.Map | None] | op2.Global | op2.Constant] = []
        names: dict[op2.Dat | op2.Map | op2.Global | op2.Constant, str] = {}
        state_index: int | None = None

        dat_name_counter = itertools.count()
        glob_name_counter = itertools.count()
        map_name_counter = itertools.count()

        def add_dat(dat, map_):
            if dat not in names:
                names[dat] = f"dat_{next(dat_name_counter)}"
            if map_ is not None and map_ not in names:
                names[map_] = f"map_{next(map_name_counter)}"
            args.append((dat, map_))

        def add_glob(glob):
            if glob not in names:
                names[glob] = f"glob_{next(glob_name_counter)}"
            args.append(glob)

        def add_coeff(coeff):
            add_dat(coeff.dat, self._get_map(coeff.function_space()))

        all_meshes = extract_domains(self.form)
        for domain_number in self.kinfo.active_domain_numbers.coordinates:
            add_coeff(all_meshes[domain_number].coordinates)

        for i in self.kinfo.active_domain_numbers.cell_orientations:
            add_coeff(all_meshes[i].cell_orientations())

        for i in self.kinfo.active_domain_numbers.cell_sizes:
            add_coeff(all_meshes[i].cell_sizes)

        for coeff_number, coeff_indices in self.kinfo.coefficient_numbers:
            coeff = self.form.coefficients()[coeff_number]
            if coeff is self.state:
                if coeff_indices != (0,):
                    raise ValueError(
                        f"Active indices of state function must be '(0,)', not '{coeff_indices}'"
                    )
                # state coefficient is provided separately but we have to
                # record its location so we know where to insert it in the
                # local kernel call instruction
                state_index = len(args) + 1  # add 1 because the output tensor is always first
                continue

            for coeff_index in coeff_indices:
                add_coeff(coeff.subfunctions[coeff_index])

        all_constants = extract_firedrake_constants(self.form)
        for constant_index in self.kinfo.constant_numbers:
            add_glob(all_constants[constant_index].dat)

        if self.kinfo.integral_type == "interior_facet":
            add_dat(self._mesh.interior_facets.local_facet_dat, None)
        elif self.kinfo.integral_type == "exterior_facet":
            add_dat(self._mesh.exterior_facets.local_facet_dat, None)

        return args, names, state_index

    @cached_property
    def _wrapper_kernel_args(self):
        """Arguments that are passed to the wrapper kernel.

        This function 'explodes' the ``_args`` attribute by placing the maps
        used by dats at the end.

        """
        flat_args = []
        maps = []
        for arg in self._args:
            if isinstance(arg, tuple):  # (dat, map)
                dat, map_ = arg
                flat_args.append(dat)
                if map_ is not None and map_ not in maps:
                    maps.append(map_)
            else:
                flat_args.append(arg)
        return tuple(flat_args + maps)

    @cached_property
    def _mesh(self):
        return extract_domains(self.form)[self.kinfo.domain_number]

    def _get_map(self, space):
        return space.entity_node_map(self._mesh.topological, self.kinfo.integral_type, None, None)

    @cached_property
    def _wrapper_kernel_code(self) -> str:
        temp_counter = itertools.count()

        temps: list[tuple[str, tuple[int, ...]]] = []
        pack_insns: list[str] = []
        local_kernel_args: list[str] = []

        # handle the output temporary
        temp_name = f"t_{next(temp_counter)}"
        spaces = map(operator.methodcaller("function_space"), self.form.arguments())
        sizes = []
        for space in spaces:
            map_ = self._get_map(space)
            size = sum(
                map_.arity*dset.cdim
                for map_, dset in zip(map_, space.dof_dset, strict=True)
            )
            sizes.append(size)
        if len(self.form.arguments()) == 2:
            row_size, column_size = sizes

            temps.append((temp_name, (row_size, column_size)))

            pack_insn = f"""\
for (int32_t k=0; k<{row_size}*{column_size}; k++)
  {temp_name}[k] = 0.0;"""
            pack_insns.append(pack_insn)

            local_kernel_args.append(temp_name)

            unpack_insn = (
                f"MatSetValues(J, {row_size}, &(activeDofsArray[{row_size}*i]), {column_size}, &(activeDofsArray[{column_size}*i]), {temp_name}, ADD_VALUES);"
            )
        else:
            size, = sizes

            temps.append((temp_name, (size,)))

            pack_insn = f"""\
for (int32_t k=0; k<{size}; k++)
  {temp_name}[k] = 0.0;"""
            pack_insns.append(pack_insn)

            local_kernel_args.append(temp_name)

            unpack_insn = f"""\
for (int32_t k=0; k<{size}; k++) {{
  if (activeDofsArray[{size}*i+k] >= 0)
    F[activeDofsArray[{size}*i+k]] += {temp_name}[k];
}}"""

        # now handle the other arguments
        for arg in self._args:
            if isinstance(arg, tuple):  # (dat, map)
                dat, map_ = arg
                assert isinstance(dat, op2.Dat)
                cdim = dat.dataset.cdim
                dat_name = self._names[dat]
                if map_ is None:
                    local_kernel_args.append(f"&({dat_name}[{cdim}*j])")
                else:
                    temp_name = f"t_{next(temp_counter)}"
                    map_name = self._names[map_]
                    arity = map_.arity
                    temps.append((temp_name, (arity, cdim)))

                    local_kernel_args.append(temp_name)

                    pack_insn = f"""\
for (int32_t k=0; k<{arity}; k++)
  for (int32_t l=0; l<{cdim}; l++)
    {temp_name}[{cdim}*k+l] = {dat_name}[{map_name}[j*{arity}+k]*{cdim}+l];"""
                    pack_insns.append(pack_insn)

            else:
                assert isinstance(arg, op2.Global | op2.Constant)
                local_kernel_args.append(self._names[arg])

        # optional state, can be any of the coefficients
        if self.state is not None:
            assert self._state_index is not None
            temp_name = f"t_{next(temp_counter)}"
            size = sizes[0]
            temps.append((temp_name, (size,)))

            local_kernel_args.insert(self._state_index, temp_name)

            pack_insn = f"""\
for (int32_t k=0; k<{size}; k++)
  {temp_name}[k] = state[dofArrayWithAll[i*{size}+k]];"""
            pack_insns.append(pack_insn)

        # generate the rest
        temp_decls = []
        for temp_name, temp_shape in temps:
            temp_decl = f"PetscScalar {temp_name}[{'*'.join(map(str, temp_shape))}];"
            temp_decls.append(temp_decl)
        temp_decls_str = "\n".join(temp_decls)
        pack_insns_str = "\n".join(pack_insns)
        local_kernel_call_insn = (
            f"{self.kinfo.kernel.name}({', '.join(local_kernel_args)});"
        )

        # wrapper kernel signature
        out_sig = "Mat J" if len(self.form.arguments()) == 2 else "PetscScalar *__restrict__ F"
        args_sig = f"PetscInt n, const PetscInt *__restrict__ subset, {out_sig}, const PetscInt *__restrict__ activeDofsArray"
        if self.state is not None:
            args_sig += ", const PetscScalar *__restrict__ state, const PetscInt *__restrict__ dofArrayWithAll"

        if self._wrapper_kernel_args:
            extra_sigs = ", ".join((
                f"const {as_cstr(arg.dtype)} *__restrict__ {self._names[arg]}"
                for arg in self._wrapper_kernel_args
            ))
            args_sig += f", {extra_sigs}"

        return f"""\
void wrapper_kernel({args_sig})
{{
{textwrap.indent(temp_decls_str, " "*2)}
  PetscInt j;

  for (int32_t i=0; i<n; i++)
  {{
    j = subset[i];
{textwrap.indent(pack_insns_str, " "*4)}

    {local_kernel_call_insn}

{textwrap.indent(unpack_insn, " "*4)}
  }}
}}
"""

    @cached_property
    def _callback_code(self) -> str:
        """Return the code that gets compiled and used for the PETSc callback."""
        if len(self.form.arguments()) == 2:
            return self._make_jacobian_callback_code()
        else:
            return self._make_residual_callback_code()

    def _make_jacobian_callback_code(self) -> str:
        return f"""\
#include <petsc.h>

{self._local_kernel_code}

{self._wrapper_kernel_code}

{self._struct_code}

PetscErrorCode ComputeJacobian(PC pc,
                               PetscInt point,
                               Vec x,
                               Mat J,
                               IS points,
                               PetscInt ndof,
                               const PetscInt *dofArray,
                               const PetscInt *dofArrayWithAll,
                               void *ctx_)
{{
  const PetscScalar *state           = NULL;
  const PetscInt    *whichPoints     = NULL;
  const PetscInt    *activeDofsArray = dofArray;
  UserCtx           *ctx             = (UserCtx *)ctx_;
  PetscInt           npoints;
  PetscInt          *filtpoints      = NULL;
  PetscInt          *filtdofs        = NULL;

  PetscFunctionBeginUser;
  PetscCall(ISGetSize(points, &npoints));
  if (!npoints) PetscFunctionReturn(PETSC_SUCCESS);
  if (x) PetscCall(VecGetArrayRead(x, &state));
  PetscCall(ISGetIndices(points, &whichPoints));
  if (ctx->point2facet) {{
    PetscInt nvalid = 0;
    PetscInt tDPP   = ndof / npoints;
    PetscCall(PetscMalloc1(npoints, &filtpoints));
    if (ndof > 0) PetscCall(PetscMalloc1(ndof, &filtdofs));
    for (PetscInt i = 0; i < npoints; i++) {{
      PetscInt fi = ctx->point2facet[whichPoints[i]];
      if (fi >= 0) {{
        filtpoints[nvalid] = fi;
        for (PetscInt d = 0; d < tDPP; d++)
          filtdofs[nvalid * tDPP + d] = dofArray[i * tDPP + d];
        nvalid++;
      }}
    }}
    PetscCall(ISRestoreIndices(points, &whichPoints));
    npoints         = nvalid;
    whichPoints     = filtpoints;
    activeDofsArray = filtdofs;
  }}

  if (npoints)
    {self._wrapper_kernel_call_insn};

  if (ctx->point2facet) {{
    PetscCall(PetscFree(filtpoints));
    PetscCall(PetscFree(filtdofs));
  }} else {{
    PetscCall(ISRestoreIndices(points, &whichPoints));
  }}
  if (x) PetscCall(VecRestoreArrayRead(x, &state));

  PetscCall(PetscLogFlops({self.kinfo.kernel.num_flops} * npoints));
  PetscFunctionReturn(PETSC_SUCCESS);
}}"""

    def _make_residual_callback_code(self) -> str:
        return f"""
#include <petsc.h>

{self._local_kernel_code}

{self._wrapper_kernel_code}

{self._struct_code}

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
  const PetscScalar *state           = NULL;
  const PetscInt    *whichPoints     = NULL;
  const PetscInt    *activeDofsArray = dofArray;
  PetscScalar       *Fdat            = NULL;
  UserCtx           *ctx             = (UserCtx *)ctx_;
  PetscInt           npoints;
  PetscInt          *filtpoints      = NULL;
  PetscInt          *filtdofs        = NULL;

  PetscFunctionBeginUser;
  PetscCall(ISGetSize(points, &npoints));
  if (!npoints) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(VecSet(F, 0.0));
  if (x) PetscCall(VecGetArrayRead(x, &state));
  PetscCall(VecGetArray(F, &Fdat));
  PetscCall(ISGetIndices(points, &whichPoints));
  if (ctx->point2facet) {{
    PetscInt nvalid = 0;
    PetscInt tDPP   = ndof / npoints;
    PetscCall(PetscMalloc1(npoints, &filtpoints));
    if (ndof > 0) PetscCall(PetscMalloc1(ndof, &filtdofs));
    for (PetscInt i = 0; i < npoints; i++) {{
      PetscInt fi = ctx->point2facet[whichPoints[i]];
      if (fi >= 0) {{
        filtpoints[nvalid] = fi;
        for (PetscInt d = 0; d < tDPP; d++)
          filtdofs[nvalid * tDPP + d] = dofArray[i * tDPP + d];
        nvalid++;
      }}
    }}
    PetscCall(ISRestoreIndices(points, &whichPoints));
    npoints         = nvalid;
    whichPoints     = filtpoints;
    activeDofsArray = filtdofs;
  }}

  if (npoints)
    {self._wrapper_kernel_call_insn};

  if (ctx->point2facet) {{
    PetscCall(PetscFree(filtpoints));
    PetscCall(PetscFree(filtdofs));
  }} else {{
    PetscCall(ISRestoreIndices(points, &whichPoints));
  }}
  PetscCall(VecRestoreArray(F, &Fdat));
  if (x) PetscCall(VecRestoreArrayRead(x, &state));

  PetscCall(PetscLogFlops({self.kinfo.kernel.num_flops} * npoints));
  PetscFunctionReturn(PETSC_SUCCESS);
}}
"""

    @cached_property
    def _local_kernel_code(self) -> str:
        return lp.generate_code_v2(self.kinfo.kernel.code).device_code()

    @property
    def _wrapper_kernel_call_insn(self) -> str:
        out_name = "J"if len(self.form.arguments()) == 2 else "Fdat"
        args = f"npoints, whichPoints, {out_name}, activeDofsArray"

        if self.state is not None:
            args += ", state, dofArrayWithAll"
        if self._wrapper_kernel_args:
            extra_args = ", ".join(
                (f"ctx->{self._names[arg]}" for arg in self._wrapper_kernel_args)
            )
            args += f", {extra_args}"

        return f"wrapper_kernel({args})"

    @cached_property
    def _ctypes_struct(self) -> ctypes.Structure:
        """Return a struct containing additional wrapper kernel arguments."""
        fields = [
            *((self._names[arg], ctypes.c_voidp) for arg in self._wrapper_kernel_args),
            ("point2facet", ctypes.c_voidp),
        ]

        class Struct(ctypes.Structure):
            _fields_ = fields

        if self.kinfo.integral_type == "cell":
            point2facet = 0
        elif self.kinfo.integral_type == "interior_facet":
            point2facet = self._mesh.interior_facets.point2facetnumber.ctypes.data
        else:
            assert self.kinfo.integral_type == "exterior_facet"
            point2facet = self._mesh.exterior_facets.point2facetnumber.ctypes.data

        struct_args = [
            *(karg for arg in self._wrapper_kernel_args for karg in arg._kernel_args_),
            point2facet,
        ]
        return Struct(*struct_args)

    @cached_property
    def _struct_code(self) -> str:
        decls = "\n".join((
            f"const {as_cstr(arg.dtype)} *{self._names[arg]};"
            for arg in self._wrapper_kernel_args
        ))

        return f"""\
typedef struct {{
{textwrap.indent(decls, "  ")}
  const PetscInt *point2facet;
}} UserCtx;"""


def make_patch_callables(form: ufl.Form, state: Function | None) -> tuple[
    PatchCallable | None, PatchCallable | None, PatchCallable | None
]:
    """Return 3 patch callables for cells and interior and exterior facets."""
    if state is not None:
        dont_split = (state,)
    else:
        dont_split = ()
    kernels = compile_form(form, "subspace_form", split=False, dont_split=dont_split)
    cell_callable = None
    interior_facet_callable = None
    exterior_facet_callable = None
    for kernel in kernels:
        kinfo = kernel.kinfo
        if kinfo.subdomain_id != ("otherwise",):
            raise NotImplementedError("Only for full domain integrals")
        if kinfo.integral_type not in {"cell", "interior_facet", "exterior_facet"}:
            raise NotImplementedError("Only for cell, interior facet, or exterior facet integrals")

        callable = PatchCallable(form, kinfo, state)
        if kinfo.integral_type == "cell":
            assert cell_callable is None, "Only a single cell callable allowed"
            cell_callable = callable
        elif kinfo.integral_type == "interior_facet":
            assert interior_facet_callable is None, "Only a single interior facet callable allowed"
            interior_facet_callable = callable
        else:
            assert kinfo.integral_type == "exterior_facet"
            assert exterior_facet_callable is None, "Only a single exterior facet callable allowed"
            exterior_facet_callable = callable
    return cell_callable, interior_facet_callable, exterior_facet_callable


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

        # We need to set C function pointer callbacks for PCPatch to work.
        # Although petsc4py provides a high-level Python wrapper for them,
        # this is very costly when going back and forth from C to Python only
        # to extract function pointers and send them straight back to C. Here,
        # since we know what the calling convention of the C function is, we
        # just wrap up everything as a C function pointer and use that
        # directly.

        is_snes = isinstance(obj, PETSc.SNES)
        state = ctx._problem.u if is_snes else None

        test, trial = map(operator.methodcaller("function_space"), J.arguments())
        if test != trial:
            raise NotImplementedError("Only for matching test and trial spaces")

        jacobian_patch_callables = make_patch_callables(J, state)
        # save a reference to prevent premature cleanup
        self._jacobian_patch_callables = jacobian_patch_callables
        (
            jacobian_cell_callable,
            jacobian_interior_facet_callable,
            jacobian_exterior_facet_callable,
        ) = jacobian_patch_callables

        if is_snes:
            if jacobian_cell_callable:
                firedrake.cython.patchimpl.snespatch_set_compute_operator(
                    patch,
                    jacobian_cell_callable.ctypes_callable,
                    jacobian_cell_callable.ctypes_struct_address,
                )
            if jacobian_interior_facet_callable:
                raise NotImplementedError("Interior facet operators not implemented for SNESPatch")
            if jacobian_exterior_facet_callable:
                raise NotImplementedError("Exterior facet operators not implemented for SNESPatch")

        else:
            if jacobian_cell_callable:
                firedrake.cython.patchimpl.pcpatch_set_compute_operator(
                    patch,
                    jacobian_cell_callable.ctypes_callable,
                    jacobian_cell_callable.ctypes_struct_address,
                )
            if jacobian_interior_facet_callable:
                firedrake.cython.patchimpl.pcpatch_set_compute_operator_interior_facets(
                    patch,
                    jacobian_interior_facet_callable.ctypes_callable,
                    jacobian_interior_facet_callable.ctypes_struct_address,
                )
            if jacobian_exterior_facet_callable:
                firedrake.cython.patchimpl.pcpatch_set_compute_operator_exterior_facets(
                    patch,
                    jacobian_exterior_facet_callable.ctypes_callable,
                    jacobian_exterior_facet_callable.ctypes_struct_address,
                )

        if is_snes and hasattr(ctx, "F"):
            test, = map(operator.methodcaller("function_space"), ctx.F.arguments())
            if state.function_space() != test:
                raise NotImplementedError("State and test space must be dual to one-another")

            residual_patch_callables = make_patch_callables(ctx.F, state)
            # save a reference to prevent premature cleanup
            self._residual_patch_callables = residual_patch_callables
            (
                residual_cell_callable,
                residual_interior_facet_callable,
                residual_exterior_facet_callable,
            ) = residual_patch_callables

            if is_snes:
                if residual_cell_callable:
                    firedrake.cython.patchimpl.snespatch_set_compute_function(
                        patch,
                        residual_cell_callable.ctypes_callable,
                        residual_cell_callable.ctypes_struct_address,
                    )
                if residual_interior_facet_callable:
                    raise NotImplementedError(
                        "Interior facet residual functions not implemented for SNESPatch"
                    )
                if residual_exterior_facet_callable:
                    raise NotImplementedError(
                        "Exterior facet residual functions not implemented for SNESPatch"
                    )

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
        patch.setPatchConstructType(PETSc.PC.PatchConstructType.PYTHON, operator=self.user_construction_op)
        patch.setAttr("ctx", ctx)
        patch.incrementTabLevel(1, parent=obj)
        patch.setFromOptions()
        patch.setUp()
        self.patch = patch

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

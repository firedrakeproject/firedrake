from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.petsc import PETSc
from firedrake.solving_utils import _SNESContext
from firedrake.utils import cached_property
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx

from collections import namedtuple
import operator
from functools import partial
import numpy
from ufl import VectorElement, MixedElement
from tsfc.kernel_interface.firedrake_loopy import make_builder

from pyop2 import op2
from pyop2 import base as pyop2
from pyop2 import sequential as seq
from pyop2.codegen.builder import Pack, MatPack, DatPack
from pyop2.codegen.representation import Comparison, Literal
from pyop2.codegen.rep2loopy import register_petsc_function
from pyop2.datatypes import IntType

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
    def coords(dm, p):
        coordsSection = dm.getCoordinateSection()
        coordsDM = dm.getCoordinateDM()
        dim = coordsDM.getDimension()
        coordsVec = dm.getCoordinatesLocal()
        return dm.getVecClosure(coordsSection, coordsVec, p).reshape(-1, dim).mean(axis=0)

    def sort_entities(self, dm, axis, dir, ndiv):
        # compute
        # [(pStart, (x, y, z)), (pEnd, (x, y, z))]
        select = partial(select_entity, dm=dm, exclude="pyop2_ghost")
        entities = [(p, self.coords(dm, p)) for p in
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
        self.plex = mesh._plex
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
        else:
            Jstate = None

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
        Jop_coeffs = [mesh.coordinates]
        Jcell_kernel, = Jcell_kernels
        if Jcell_kernel.kinfo.oriented:
            Jop_coeffs.append(J.ufl_domain().cell_orientations())
        for n in Jcell_kernel.kinfo.coefficient_map:
            Jop_coeffs.append(J.coefficients()[n])

        Jop_data_args = []
        Jop_map_args = []
        seen = set()
        Jop_state_data_slot = None
        Jop_state_map_slot = None
        for c in Jop_coeffs:
            if c is Jstate:
                Jop_state_data_slot = len(Jop_data_args)
                Jop_state_map_slot = len(Jop_map_args)
                Jop_data_args.append(None)
                Jop_map_args.append(None)
                continue
            Jop_data_args.extend(c.dat._kernel_args_)
            map_ = c.cell_node_map()
            if map_ is not None:
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    Jop_map_args.append(k)
                    seen.add(k)

        def Jop(obj, point, vec, mat, cellIS, cell_dofmap, cell_dofmapWithAll):
            cells = cellIS.indices
            ncell = len(cells)
            dofs = cell_dofmap.ctypes.data
            if cell_dofmapWithAll is not None:
                dofsWithAll = cell_dofmapWithAll.ctypes.data
            else:
                dofsWithAll = None
            if Jop_state_data_slot is not None:
                assert dofsWithAll is not None
                Jop_data_args[Jop_state_data_slot] = vec.array_r.ctypes.data
                Jop_map_args[Jop_state_map_slot] = dofsWithAll
            Jcell_kernel.funptr(0, ncell, cells.ctypes.data, mat.handle, *Jop_data_args,
                                dofs, *Jop_map_args)

        Jhas_int_facet_kernel = False
        if len(Jint_facet_kernels) > 0:
            Jint_facet_kernel, = Jint_facet_kernels
            Jhas_int_facet_kernel = True
            facet_Jop_coeffs = [mesh.coordinates]
            if Jint_facet_kernel.kinfo.oriented:
                facet_Jop_coeffs.append(J.ufl_domain().cell_orientations())
            for n in Jint_facet_kernel.kinfo.coefficient_map:
                facet_Jop_coeffs.append(J.coefficients()[n])

            facet_Jop_data_args = []
            facet_Jop_map_args = []
            facet_Jop_state_data_slot = None
            facet_Jop_state_map_slot = None
            seen = set()
            for c in facet_Jop_coeffs:
                if c is Jstate:
                    facet_Jop_state_data_slot = len(facet_Jop_data_args)
                    facet_Jop_state_map_slot = len(facet_Jop_map_args)
                    facet_Jop_data_args.append(None)
                    facet_Jop_map_args.append(None)
                    continue
                facet_Jop_data_args.extend(c.dat._kernel_args_)
                map_ = c.interior_facet_node_map()
                if map_ is not None:
                    for k in map_._kernel_args_:
                        if k in seen:
                            continue
                        facet_Jop_map_args.append(k)
                        seen.add(k)
            facet_Jop_data_args.extend(J.ufl_domain().interior_facets.local_facet_dat._kernel_args_)

            point2facetnumber = J.ufl_domain().interior_facets.point2facetnumber

            def Jfacet_op(pc, point, vec, mat, facetIS, facet_dofmap, facet_dofmapWithAll):
                facets = numpy.asarray(list(map(point2facetnumber.__getitem__, facetIS.indices)),
                                       dtype=IntType)
                nfacet = len(facets)
                dofs = facet_dofmap.ctypes.data
                if facet_Jop_state_data_slot is not None:
                    assert facet_dofmapWithAll is not None
                    facet_Jop_data_args[facet_Jop_state_data_slot] = vec.array_r.ctypes.data
                    facet_Jop_map_args[facet_Jop_state_map_slot] = facet_dofmapWithAll.ctypes.data
                Jint_facet_kernel.funptr(0, nfacet, facets.ctypes.data, mat.handle, *facet_Jop_data_args,
                                         dofs, *facet_Jop_map_args)

        set_residual = hasattr(ctx, "F") and isinstance(obj, PETSc.SNES)
        if set_residual:
            F = ctx.F
            Fstate = ctx._problem.u
            Fcell_kernels, Fint_facet_kernels = residual_funptr(F, Fstate)
            Fop_coeffs = [mesh.coordinates]
            Fcell_kernel, = Fcell_kernels
            if Fcell_kernel.kinfo.oriented:
                Fop_coeffs.append(F.ufl_domain().cell_orientations())
            for n in Fcell_kernel.kinfo.coefficient_map:
                Fop_coeffs.append(F.coefficients()[n])
            assert any(c is Fstate for c in Fop_coeffs), "Couldn't find state vector in F.coefficients()"

            Fop_data_args = []
            Fop_map_args = []
            seen = set()
            Fop_state_data_slot = None
            Fop_state_map_slot = None
            for c in Fop_coeffs:
                if c is Fstate:
                    Fop_state_data_slot = len(Fop_data_args)
                    Fop_state_map_slot = len(Fop_map_args)
                    Fop_data_args.append(None)
                    Fop_map_args.append(None)
                    continue
                Fop_data_args.extend(c.dat._kernel_args_)
                map_ = c.cell_node_map()
                if map_ is not None:
                    for k in map_._kernel_args_:
                        if k in seen:
                            continue
                        Fop_map_args.append(k)
                        seen.add(k)

            assert Fop_state_data_slot is not None

            def Fop(pc, point, vec, out, cellIS, cell_dofmap, cell_dofmapWithAll):
                cells = cellIS.indices
                ncell = len(cells)
                dofs = cell_dofmap.ctypes.data
                dofsWithAll = cell_dofmapWithAll.ctypes.data
                out.set(0)
                outdata = out.array
                Fop_data_args[Fop_state_data_slot] = vec.array_r.ctypes.data
                Fop_map_args[Fop_state_map_slot] = dofsWithAll
                Fcell_kernel.funptr(0, ncell, cells.ctypes.data, outdata.ctypes.data,
                                    *Fop_data_args, dofs, *Fop_map_args)

            Fhas_int_facet_kernel = False
            if len(Fint_facet_kernels) > 0:
                Fint_facet_kernel, = Fint_facet_kernels
                Fhas_int_facet_kernel = True
                facet_Fop_coeffs = [mesh.coordinates]
                if Fint_facet_kernel.kinfo.oriented:
                    facet_Fop_coeffs.append(F.ufl_domain().cell_orientations())
                for n in Fint_facet_kernel.kinfo.coefficient_map:
                    facet_Fop_coeffs.append(J.coefficients()[n])

                facet_Fop_data_args = []
                facet_Fop_map_args = []
                facet_Fop_state_data_slot = None
                facet_Fop_state_map_slot = None
                seen = set()
                for c in facet_Fop_coeffs:
                    if c is Fstate:
                        facet_Fop_state_data_slot = len(Fop_data_args)
                        facet_Fop_state_map_slot = len(Fop_map_args)
                        facet_Fop_data_args.append(None)
                        facet_Fop_map_args.append(None)
                        continue
                    facet_Fop_data_args.extend(c.dat._kernel_args_)
                    map_ = c.interior_facet_node_map()
                    if map_ is not None:
                        for k in map_._kernel_args_:
                            if k in seen:
                                continue
                            facet_Fop_map_args.append(k)
                            seen.add(k)
                facet_Fop_data_args.extend(F.ufl_domain().interior_facets.local_facet_dat._kernel_args_)

                point2facetnumber = F.ufl_domain().interior_facets.point2facetnumber

                def Ffacet_op(pc, point, vec, mat, facetIS, facet_dofmap, facet_dofmapWithAll):
                    facets = numpy.asarray(list(map(point2facetnumber.__getitem__, facetIS.indices)),
                                           dtype=IntType)
                    nfacet = len(facets)
                    dofs = facet_dofmap.ctypes.data
                    if facet_Fop_state_data_slot is not None:
                        assert facet_dofmapWithAll is not None
                        facet_Fop_data_args[facet_Fop_state_data_slot] = vec.array_r.ctypes.data
                        facet_Fop_map_args[facet_Fop_state_map_slot] = facet_dofmapWithAll.ctypes.data
                    Fint_facet_kernel.funptr(0, nfacet, facets.ctypes.data, mat.handle, *facet_Fop_data_args,
                                             dofs, *facet_Fop_map_args)

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
        patch.setPatchComputeOperator(Jop)
        if Jhas_int_facet_kernel:
            patch.setPatchComputeOperatorInteriorFacets(Jfacet_op)
        if set_residual:
            patch.setPatchComputeFunction(Fop)
            if Fhas_int_facet_kernel:
                patch.setPatchComputeFunctionInteriorFacets(Ffacet_op)

        patch.setPatchConstructType(PETSc.PC.PatchConstructType.PYTHON, operator=self.user_construction_op)
        patch.setAttr("ctx", ctx)
        patch.incrementTabLevel(1, parent=obj)
        patch.setFromOptions()
        patch.setUp()
        self.patch = patch

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

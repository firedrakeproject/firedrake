from firedrake.preconditioners.base import PCBase, SNESBase, PCSNESBase
from firedrake.petsc import PETSc
from firedrake.solving_utils import _SNESContext
from firedrake.matrix_free.operators import ImplicitMatrixContext
from firedrake.dmhooks import get_appctx, push_appctx, pop_appctx

from collections import namedtuple
import operator
from functools import partial
import numpy
from ufl import VectorElement, MixedElement
from tsfc.kernel_interface.firedrake import make_builder

from pyop2 import op2
from pyop2 import base as pyop2
from pyop2 import sequential as seq
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


class MatArg(seq.Arg):
    def c_addto(self, i, j, buf_name, tmp_name, tmp_decl,
                extruded=None, is_facet=False, applied_blas=False):
        # Override global c_addto to index the map locally rather than globally.
        # Replaces MatSetValuesLocal with MatSetValues
        from pyop2.utils import as_tuple
        rmap, cmap = as_tuple(self.map, op2.Map)
        rset, cset = self.data.sparsity.dsets
        nrows = sum(m.arity*s.cdim for m, s in zip(rmap, rset))
        ncols = sum(m.arity*s.cdim for m, s in zip(cmap, cset))
        rows_str = "%s + n * %s" % (self.c_map_name(0, i), nrows)
        cols_str = "%s + n * %s" % (self.c_map_name(1, j), ncols)

        if extruded is not None:
            raise NotImplementedError("Not for extruded right now")

        if is_facet:
            raise NotImplementedError("Not for interior facets and extruded")

        ret = []
        addto_name = buf_name
        if rmap.vector_index is not None or cmap.vector_index is not None:
            raise NotImplementedError
        ret.append("""MatSetValues(%(mat)s, %(nrows)s, %(rows)s,
                                         %(ncols)s, %(cols)s,
                                         (const PetscScalar *)%(vals)s,
                                         %(insert)s);""" %
                   {'mat': self.c_arg_name(i, j),
                    'vals': addto_name,
                    'nrows': nrows,
                    'ncols': ncols,
                    'rows': rows_str,
                    'cols': cols_str,
                    'insert': "INSERT_VALUES" if self.access == op2.WRITE else "ADD_VALUES"})
        return "\n".join(ret)


class DenseMat(pyop2.Mat):
    def __init__(self, dset):
        self._sparsity = DenseSparsity(dset, dset)
        self.dtype = numpy.dtype(PETSc.ScalarType)

    def __call__(self, access, path):
        path_maps = [arg.map for arg in path]
        path_idxs = [arg.idx for arg in path]
        return MatArg(self, path_maps, path_idxs, access)


class DatArg(seq.Arg):

    def c_buffer_gather(self, size, idx, buf_name, extruded=False):
        dim = self.data.cdim
        val = ";\n".join(["%(name)s[i_0*%(dim)d%(ofs)s] = *(%(ind)s%(ofs)s);\n" %
                          {"name": buf_name,
                           "dim": dim,
                           "ind": self.c_kernel_arg(idx, extruded=extruded),
                           "ofs": " + %s" % j if j else ""} for j in range(dim)])
        val = val.replace("[i *", "[n *")
        return val

    def c_buffer_scatter_vec(self, count, i, j, mxofs, buf_name, extruded=False):
        dim = self.data.split[i].cdim
        ind = self.c_kernel_arg(count, i, j, extruded=extruded)
        ind = ind.replace("[i *", "[n *")
        map_val = "%(map_name)s[%(var)s * %(arity)s + %(idx)s]" % \
            {'name': self.c_arg_name(i),
             'map_name': self.c_map_name(i, 0),
             'var': "n",
             'arity': self.map.split[i].arity,
             'idx': "i_%d" % self.idx.index}

        val = "\n".join(["if (%(map_val)s >= 0) {*(%(ind)s%(nfofs)s) %(op)s %(name)s[i_0*%(dim)d%(nfofs)s%(mxofs)s];}" %
                         {"ind": ind,
                          "op": "=" if self.access == op2.WRITE else "+=",
                          "name": buf_name,
                          "dim": dim,
                          "nfofs": " + %d" % o if o else "",
                          "mxofs": " + %d" % (mxofs[0] * dim) if mxofs else "",
                          "map_val": map_val}
                         for o in range(dim)])
        return val


class DenseDat(pyop2.Dat):
    def __init__(self, dset):
        self._dataset = dset
        self.dtype = numpy.dtype(PETSc.ScalarType)
        self._soa = False

    def __call__(self, access, path):
        return DatArg(self, map=path.map, idx=path.idx, access=access)


class JITModule(seq.JITModule):
    @classmethod
    def _cache_key(cls, *args, **kwargs):
        # No caching
        return None


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
        mat = DenseMat(dofset)

        arg = mat(op2.INC, (entity_node_map[op2.i[0]],
                            entity_node_map[op2.i[1]]))
        arg.position = 0
        args.append(arg)
        statedat = DenseDat(dofset)
        statearg = statedat(op2.READ, entity_node_map[op2.i[0]])

        mesh = form.ufl_domains()[kinfo.domain_number]
        arg = mesh.coordinates.dat(op2.READ, get_map(mesh.coordinates)[op2.i[0]])
        arg.position = 1
        args.append(arg)
        for n in kinfo.coefficient_map:
            c = form.coefficients()[n]
            if c is state:
                statearg.position = len(args)
                args.append(statearg)
                continue
            for (i, c_) in enumerate(c.split()):
                map_ = get_map(c_)
                if map_ is not None:
                    map_ = map_[op2.i[0]]
                arg = c_.dat(op2.READ, map_)
                arg.position = len(args)
                args.append(arg)

        if kinfo.integral_type == "interior_facet":
            arg = test.ufl_domain().interior_facets.local_facet_dat(op2.READ)
            arg.position = len(args)
            args.append(arg)
        iterset = op2.Subset(iterset, [0])
        mod = JITModule(kinfo.kernel, iterset, *args)
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
        dat = DenseDat(dofset)

        statedat = DenseDat(dofset)
        statearg = statedat(op2.READ, entity_node_map[op2.i[0]])

        arg = dat(op2.INC, entity_node_map[op2.i[0]])
        arg.position = 0
        args.append(arg)

        mesh = form.ufl_domains()[kinfo.domain_number]
        arg = mesh.coordinates.dat(op2.READ, get_map(mesh.coordinates)[op2.i[0]])
        arg.position = 1
        args.append(arg)
        for n in kinfo.coefficient_map:
            c = form.coefficients()[n]
            if c is state:
                statearg.position = len(args)
                args.append(statearg)
                continue
            for (i, c_) in enumerate(c.split()):
                map_ = get_map(c_)
                if map_ is not None:
                    map_ = map_[op2.i[0]]
                arg = c_.dat(op2.READ, map_)
                arg.position = len(args)
                args.append(arg)

        if kinfo.integral_type == "interior_facet":
            arg = test.ufl_domain().interior_facets.local_facet_dat(op2.READ)
            arg.position = len(args)
            args.append(arg)
        iterset = op2.Subset(iterset, [0])
        mod = JITModule(kinfo.kernel, iterset, *args)
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
        for n in Jcell_kernel.kinfo.coefficient_map:
            Jop_coeffs.append(J.coefficients()[n])

        Jop_args = []
        Jop_state_slot = None
        for c in Jop_coeffs:
            if c is Jstate:
                Jop_state_slot = len(Jop_args)
                Jop_args.append(None)
                Jop_args.append(None)
                continue
            for c_ in c.split():
                Jop_args.append(c_.dat._data.ctypes.data)
                c_map = c_.cell_node_map()
                if c_map is not None:
                    Jop_args.append(c_map._values.ctypes.data)

        def Jop(obj, point, vec, mat, cellIS, cell_dofmap, cell_dofmapWithAll):
            cells = cellIS.indices
            ncell = len(cells)
            dofs = cell_dofmap.ctypes.data
            if cell_dofmapWithAll is not None:
                dofsWithAll = cell_dofmapWithAll.ctypes.data
            else:
                dofsWithAll = None
            if Jop_state_slot is not None:
                assert dofsWithAll is not None
                Jop_args[Jop_state_slot] = vec.array_r.ctypes.data
                Jop_args[Jop_state_slot + 1] = dofsWithAll
            Jcell_kernel.funptr(0, ncell, cells.ctypes.data, mat.handle,
                                dofs, dofs, *Jop_args)

        Jhas_int_facet_kernel = False
        if len(Jint_facet_kernels) > 0:
            Jint_facet_kernel, = Jint_facet_kernels
            Jhas_int_facet_kernel = True
            facet_Jop_coeffs = [mesh.coordinates]
            for n in Jint_facet_kernel.kinfo.coefficient_map:
                facet_Jop_coeffs.append(J.coefficients()[n])

            facet_Jop_args = []
            for c in facet_Jop_coeffs:
                for c_ in c.split():
                    facet_Jop_args.append(c_.dat._data.ctypes.data)
                    c_map = c_.interior_facet_node_map()
                    if c_map is not None:
                        facet_Jop_args.append(c_map._values.ctypes.data)
            facet_Jop_args.append(J.ufl_domain().interior_facets.local_facet_dat._data.ctypes.data)

            point2facetnumber = J.ufl_domain().interior_facets.point2facetnumber

            def Jfacet_op(pc, point, vec, mat, facetIS, facet_dofmap, facet_dofmapWithAll):
                facets = numpy.asarray(list(map(point2facetnumber.__getitem__, facetIS.indices)),
                                       dtype=IntType)
                nfacet = len(facets)
                dofs = facet_dofmap.ctypes.data
                Jint_facet_kernel.funptr(0, nfacet, facets.ctypes.data, mat.handle,
                                         dofs, dofs, *facet_Jop_args)

        set_residual = hasattr(ctx, "F") and isinstance(obj, PETSc.SNES)
        if set_residual:
            F = ctx.F
            Fstate = ctx._problem.u
            Fcell_kernels, Fint_facet_kernels = residual_funptr(F, Fstate)
            Fop_coeffs = [mesh.coordinates]
            Fcell_kernel, = Fcell_kernels
            for n in Fcell_kernel.kinfo.coefficient_map:
                Fop_coeffs.append(F.coefficients()[n])
            assert any(c is Fstate for c in Fop_coeffs), "Couldn't find state vector in F.coefficients()"

            Fop_args = []
            Fop_state_slot = None
            for c in Fop_coeffs:
                if c is Fstate:
                    Fop_state_slot = len(Fop_args)
                    Fop_args.append(None)
                    Fop_args.append(None)
                    continue
                for c_ in c.split():
                    Fop_args.append(c_.dat._data.ctypes.data)
                    c_map = c_.cell_node_map()
                    if c_map is not None:
                        Fop_args.append(c_map._values.ctypes.data)

            assert Fop_state_slot is not None

            def Fop(pc, point, vec, out, cellIS, cell_dofmap, cell_dofmapWithAll):
                cells = cellIS.indices
                ncell = len(cells)
                dofs = cell_dofmap.ctypes.data
                dofsWithAll = cell_dofmapWithAll.ctypes.data
                out.set(0)
                outdata = out.array
                Fop_args[Fop_state_slot] = vec.array_r.ctypes.data
                Fop_args[Fop_state_slot + 1] = dofsWithAll
                Fcell_kernel.funptr(0, ncell, cells.ctypes.data, outdata.ctypes.data,
                                    dofs, *Fop_args)

            Fhas_int_facet_kernel = False
            if len(Fint_facet_kernels) > 0:
                Fint_facet_kernel, = Fint_facet_kernels
                Fhas_int_facet_kernel = True
                facet_Fop_coeffs = [mesh.coordinates]
                for n in Fint_facet_kernel.kinfo.coefficient_map:
                    facet_Fop_coeffs.append(J.coefficients()[n])

                facet_Fop_args = []
                for c in facet_Fop_coeffs:
                    for c_ in c.split():
                        facet_Fop_args.append(c_.dat._data.ctypes.data)
                        c_map = c_.interior_facet_node_map()
                        if c_map is not None:
                            facet_Fop_args.append(c_map._values.ctypes.data)
                facet_Fop_args.append(F.ufl_domain().interior_facets.local_facet_dat._data.ctypes.data)

                point2facetnumber = F.ufl_domain().interior_facets.point2facetnumber

                def Ffacet_op(pc, point, vec, mat, facetIS, facet_dofmap, facet_dofmapWithAll):
                    facets = numpy.asarray(list(map(point2facetnumber.__getitem__, facetIS.indices)),
                                           dtype=IntType)
                    nfacet = len(facets)
                    dofs = facet_dofmap.ctypes.data
                    Fint_facet_kernel.funptr(0, nfacet, facets.ctypes.data, mat.handle,
                                             dofs, dofs, *facet_Fop_args)

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

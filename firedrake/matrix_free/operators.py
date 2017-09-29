
from ufl import action

from firedrake.ufl_expr import adjoint
from firedrake.formmanipulation import ExtractSubBlock

from firedrake.function import Function

from firedrake.petsc import PETSc

import six
import numpy as np

__all__ = ("ImplicitMatrixContext", "LoopyImplicitMatrixContext")


def find_sub_block(iset, ises):
    """Determine if iset comes from a concatenation of some subset of
    ises.

    :arg iset: a PETSc IS to find in ``ises``.
    :arg ises: An iterable of PETSc ISes.

    :returns: The indices into ``ises`` that when concatenated
        together produces ``iset``.

    :raises LookupError: if ``iset`` could not be found in
        ``ises``.
    """
    found = []
    sfound = set()
    comm = iset.comm
    while True:
        match = False
        for i, iset_ in enumerate(ises):
            if i in sfound:
                continue
            lsize = iset_.getLocalSize()
            if lsize > iset.getLocalSize():
                continue
            indices = iset.indices
            tmp = PETSc.IS().createGeneral(indices[:lsize], comm=comm)
            if tmp.equal(iset_):
                found.append(i)
                sfound.add(i)
                iset = PETSc.IS().createGeneral(indices[lsize:], comm=comm)
                match = True
                continue
        if not match:
            break
    if iset.getSize() > 0:
        raise LookupError("Unable to find %s in %s" % (iset, ises))
    return found


class ImplicitMatrixContext(object):
    # By default, these matrices will represent diagonal blocks (the
    # (0,0) block of a 1x1 block matrix is on the diagonal).
    on_diag = True

    """This class gives the Python context for a PETSc Python matrix.

    :arg a: The bilinear form defining the matrix

    :arg row_bcs: An iterable of the :class.`.DirichletBC`s that are
      imposed on the test space.  We distinguish between row and
      column boundary conditions in the case of submatrices off of the
      diagonal.

    :arg col_bcs: An iterable of the :class.`.DirichletBC`s that are
       imposed on the trial space.

    :arg fcparams: A dictionary of parameters to pass on to the form
       compiler.

    :arg appctx: Any extra user-supplied context, available to
       preconditioners and the like.

    """
    def __init__(self, a, row_bcs=[], col_bcs=[],
                 fc_params=None, appctx=None):
        self.a = a
        self.aT = adjoint(a)
        self.fc_params = fc_params
        self.appctx = appctx

        self.row_bcs = row_bcs
        self.col_bcs = col_bcs

        # create functions from test and trial space to help
        # with 1-form assembly
        test_space, trial_space = [
            a.arguments()[i].function_space() for i in (0, 1)
        ]

        self._y = Function(test_space)
        self._x = Function(trial_space)

        # These are temporary storage for holding the BC
        # values during matvec application.  _xbc is for
        # the action and ._ybc is for transpose.
        if len(self.row_bcs) > 0:
            self._xbc = Function(trial_space)
        if len(self.col_bcs) > 0:
            self._ybc = Function(test_space)

        # Get size information from template vecs on test and trial spaces
        trial_vec = trial_space.dof_dset.layout_vec
        test_vec = test_space.dof_dset.layout_vec
        self.col_sizes = trial_vec.getSizes()
        self.row_sizes = test_vec.getSizes()

        self.block_size = (test_vec.getBlockSize(), trial_vec.getBlockSize())

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

        from firedrake.assemble import create_assembly_callable
        self._assemble_action = create_assembly_callable(self.action, tensor=self._y,
                                                         form_compiler_parameters=self.fc_params)

        self._assemble_actionT = create_assembly_callable(self.actionT, tensor=self._x,
                                                          form_compiler_parameters=self.fc_params)

    def mult(self, mat, X, Y):
        with self._x.dat.vec_wo as v:
            X.copy(v)

        # if we are a block on the diagonal, then the matrix has an
        # identity block corresponding to the Dirichlet boundary conditions.
        # our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying
        # [ A_II 0 ; 0 I ] where A_II is the block corresponding only to
        # non-fixed dofs and I is the identity block on the fixed dofs.

        # If we are not, then the matrix just has 0s in the rows and columns.

        for bc in self.col_bcs:
            bc.zero(self._x)

        self._assemble_action()

        # This sets the essential boundary condition values on the
        # result.
        if self.on_diag:
            if len(self.row_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._xbc.dat.vec_wo as v:
                    X.copy(v)
            for bc in self.row_bcs:
                bc.set(self._y, self._xbc)
        else:
            for bc in self.row_bcs:
                bc.zero(self._y)

        with self._y.dat.vec_ro as v:
            v.copy(Y)

    def multTranspose(self, mat, Y, X):
        # As for mult, just everything swapped round.
        with self._y.dat.vec_wo as v:
            Y.copy(v)

        for bc in self.row_bcs:
            bc.zero(self._y)

        self._assemble_actionT()

        if self.on_diag:
            if len(self.col_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._ybc.dat.vec_wo as v:
                    Y.copy(v)
            for bc in self.col_bcs:
                bc.set(self._x, self._ybc)
        else:
            for bc in self.col_bcs:
                bc.zero(self._x)

        with self._x.dat.vec_ro as v:
            v.copy(X)

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake matrix-free operator %s\n" %
                           type(self).__name__)

    def getInfo(self, mat, info=None):
        from mpi4py import MPI
        memory = self._x.dat.nbytes + self._y.dat.nbytes
        if hasattr(self, "_xbc"):
            memory += self._xbc.dat.nbytes
        if hasattr(self, "_ybc"):
            memory += self._ybc.dat.nbytes
        if info is None:
            info = PETSc.Mat.InfoType.GLOBAL_SUM
        if info == PETSc.Mat.InfoType.LOCAL:
            return {"memory": memory}
        elif info == PETSc.Mat.InfoType.GLOBAL_SUM:
            gmem = mat.comm.tompi4py().allreduce(memory, op=MPI.SUM)
            return {"memory": gmem}
        elif info == PETSc.Mat.InfoType.GLOBAL_MAX:
            gmem = mat.comm.tompi4py().allreduce(memory, op=MPI.MAX)
            return {"memory": gmem}
        else:
            raise ValueError("Unknown info type %s" % info)

    # Now, to enable fieldsplit preconditioners, we need to enable submatrix
    # extraction for our custom matrix type.  Note that we are splitting UFL
    # and index sets rather than an assembled matrix, keeping matrix
    # assembly deferred as long as possible.
    def createSubMatrix(self, mat, row_is, col_is, target=None):
        if target is not None:
            # Repeat call, just return the matrix, since we don't
            # actually assemble in here.
            target.assemble()
            return target
        from firedrake import DirichletBC

        # These are the sets of ISes of which the the row and column
        # space consist.
        row_ises = self._y.function_space().dof_dset.field_ises
        col_ises = self._x.function_space().dof_dset.field_ises

        row_inds = find_sub_block(row_is, row_ises)
        if row_is == col_is and row_ises == col_ises:
            col_inds = row_inds
        else:
            col_inds = find_sub_block(col_is, col_ises)

        asub = ExtractSubBlock().split(self.a,
                                       argument_indices=(row_inds, col_inds))
        Wrow = asub.arguments()[0].function_space()
        Wcol = asub.arguments()[1].function_space()

        row_bcs = []
        col_bcs = []

        for bc in self.row_bcs:
            for i, r in enumerate(row_inds):
                if bc.function_space().index == r:
                    row_bcs.append(DirichletBC(Wrow.split()[i],
                                               bc.function_arg,
                                               bc.sub_domain,
                                               method=bc.method))

        if Wrow == Wcol and row_inds == col_inds and self.row_bcs == self.col_bcs:
            col_bcs = row_bcs
        else:
            for bc in self.col_bcs:
                for i, c in enumerate(col_inds):
                    if bc.function_space().index == c:
                        col_bcs.append(DirichletBC(Wcol.split()[i],
                                                   bc.function_arg,
                                                   bc.sub_domain,
                                                   method=bc.method))
        submat_ctx = ImplicitMatrixContext(asub,
                                           row_bcs=row_bcs,
                                           col_bcs=col_bcs,
                                           fc_params=self.fc_params,
                                           appctx=self.appctx)
        submat_ctx.on_diag = self.on_diag and row_inds == col_inds
        submat = PETSc.Mat().create(comm=mat.comm)
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes),
                        bsize=submat_ctx.block_size)
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat


class LoopyImplicitMatrixContext(object):
    # By default, these matrices will represent diagonal blocks (the
    # (0,0) block of a 1x1 block matrix is on the diagonal).
    on_diag = True

    """This class gives the Python context for a PETSc Python matrix.

    :arg a: The bilinear form defining the matrix

    :arg row_bcs: An iterable of the :class.`.DirichletBC`s that are
      imposed on the test space.  We distinguish between row and
      column boundary conditions in the case of submatrices off of the
      diagonal.

    :arg col_bcs: An iterable of the :class.`.DirichletBC`s that are
       imposed on the trial space.

    :arg fcparams: A dictionary of parameters to pass on to the form
       compiler.

    :arg context: Any extra user-supplied context, available to
       preconditioners and the like.

    """
    def __init__(self, a, row_bcs=[], col_bcs=[],
                 fc_params=None, appctx=None):
        self.a = a
        self.aT = adjoint(a)
        self.fc_params = fc_params
        self.appctx = appctx

        self.row_bcs = row_bcs
        self.col_bcs = col_bcs

        # create functions from test and trial space to help
        # with 1-form assembly
        test_space, trial_space = [
            a.arguments()[i].function_space() for i in (0, 1)
        ]
        from firedrake import function

        self._y = function.Function(test_space)
        self._x = function.Function(trial_space)

        # These are temporary storage for holding the BC
        # values during matvec application.  _xbc is for
        # the action and ._ybc is for transpose.
        self._xbc = function.Function(trial_space)
        self._ybc = function.Function(test_space)

        with self._x.dat.vec_ro as xx:
            self.col_sizes = xx.getSizes()
        with self._y.dat.vec_ro as yy:
            self.row_sizes = yy.getSizes()

        if len(test_space) == 1:
            rbsize = test_space.value_size
        else:
            rbsize = 1
        if len(trial_space) == 1:
            cbsize = trial_space.value_size
        else:
            cbsize = 1

        self.block_size = (rbsize, cbsize)
        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

        # Now we build loopy kernels for the action
        from tsfc import compile_form, tsfc_to_loopy
        import loopy as lp
        import numpy as np

        # FIXME: Assumes we just get one kernel back.  No boundary
        # terms yet...
        kernel, = compile_form(self.action)
        knl = tsfc_to_loopy(kernel._ir, kernel._argument_ordering)

        coeffs = self.action.coefficients()
        fcoeffs = [x for c in coeffs for x in c.split()]
        ws_labels = ["w_" + str(i) for i, _ in enumerate(fcoeffs)]
        As_labels = [aa.name for aa in knl.args if aa.name[0] == 'A']

        things_to_batch = tuple(As_labels + ws_labels + ["coords"])
        type_dict = dict((x, np.float64) for x in things_to_batch)

        knl = lp.add_and_infer_dtypes(knl, type_dict)

        knl = lp.to_batched(knl, "nelements", things_to_batch,
                            batch_iname_prefix="iel")
        for rule in list(six.itervalues(knl.substitutions)):
            knl = lp.precompute(knl, rule.name, rule.arguments)

        coords = test_space.mesh().coordinates
        coord_fs = coords.function_space()

        # I want to get all of the local-to-global mappings needed for
        # scatter and gather in place.
        # This means:
        # 1.) a master list of all the function spaces
        # 2.) a list for the things I'm scattering
        # 3.) another list for the things I'm gathering.

        fspace_to_number = {}
        fspaces_in_form = []
        fspace_to_number[coord_fs] = 0
        fspaces_in_form.append(coord_fs)

        for coeff in fcoeffs:
            fs = coeff.function_space()
            if fs not in fspace_to_number:
                fspace_to_number[fs] = len(fspace_to_number)
                fspaces_in_form.append(fs)

        # the bits of the test space are also function spaces that
        # might or might not appear among the coefficients.
        # They are definitely needed for the output arguments.
        for ts in test_space.split():
            if ts not in fspace_to_number:
                fspace_to_number[ts] = len(fspace_to_number)
                fspaces_in_form.append(ts)

        # we need to handle scatter kernels for the coordinates
        # and coefficients.
        things_to_scatter = (
            [("coords", coord_fs)]
            + [("w_%d" % i, coeff.function_space())
               for i, coeff in enumerate(fcoeffs)])

        kernel_args = {}

        for i, (thing, fspace) in enumerate(things_to_scatter):
            thing_global = thing + "_global"
            fspace_nr = fspace_to_number[fspace]
            ltg_cur = fspace.cell_node_map().values
            ibf = "ibf_scat_%d" % i
            idim = "idim_scat_%d" % i
            ltgi = "ltg_%d" % fspace_nr
            if fspace.value_size == 1:
                tg_shape = (thing_global+"_len",)
                scatter_rule = """
                {thing}[iel, {ibf}, {idim}] = \
                {thing_global}[{ltgi}[iel, {ibf}]]
                """.format(
                    thing=thing, thing_global=thing_global,
                    ltgi=ltgi,
                    fnr=fspace_nr,
                    idim=idim,
                    ibf=ibf,
                    dim=fspace.value_size),
            else:
                tg_shape = (thing_global+"_len", fspace.value_size)
                scatter_rule = """
                {thing}[iel, {ibf}, {idim}] = \
                {thing_global}[{ltgi}[iel, {ibf}], {idim}]
                """.format(
                    thing=thing, thing_global=thing_global,
                    ltgi=ltgi,
                    fnr=fspace_nr,
                    idim=idim,
                    ibf=ibf,
                    dim=fspace.value_size),

            scatter_knl = lp.make_kernel(
                "{{ [iel, {ibf}, {idim}]:"
                "    0 <= iel < nelements "
                "and 0 <= {ibf} < {nbf} "
                "and 0 <= {idim} < {dim}}}"
                .format(
                    fnr=fspace_nr,
                    nbf=ltg_cur.shape[1],
                    dim=fspace.value_size,
                    ibf=ibf,
                    idim=idim,
                ),
                scatter_rule,
                [lp.ValueArg(thing_global+"_len", np.int32),
                 lp.GlobalArg(thing_global, np.float64,
                              shape=tg_shape), "..."]
            )

            scatter_knl = lp.add_dtypes(
                scatter_knl, {
                    "nelements": np.int32,
                    ltgi: np.int32,
                    thing: np.float64,
                })

            knl = lp.fuse_kernels(
                (scatter_knl, knl),
                data_flow=[(thing, 0, 1)])

            knl = lp.assignment_to_subst(knl, thing)

        knl = lp.make_reduction_inames_unique(knl)

        
        # Now, for each bit of the test space/resulting tensor,
        # We need to initialize space to store it,
        # and gather into it.
        # We are going to have one output value for each field
        # so that we can copy into a dat with ghost values
        # and reverse the halo exchange before copying into the
        # output PETSc vector.

        gather_inames = [("iel", "0", "nelements")]
        gather_insns = []
        gather_args = []
        
        for i, ts in enumerate(test_space.split()):
            tssize = "A%d_size" % i
            aiglobal = "A%d_global" % i
            init_i = "i_init_%d" % i
            init_id = "init_%d" % i

           
            gather_inames.append(
                (init_i, "0", tssize))

            if ts.value_size > 1:
                init_dim = "dim_init_%d" % i
                
                gather_inames.append(
                    (init_dim, "0", str(ts.value_size))
                )
                gather_insns.append(
                    """{aiglobal}[{init_i}, {init_dim}] = 0.0 {{id={init_id}, atomic}}""".format(
                    aiglobal=aiglobal,
                    init_i=init_i,
                    init_dim=init_dim,
                    init_id=init_id))
            else:
                gather_insns.append(
                    """{aiglobal}[{init_i}] = 0.0 {{id={init_id}, atomic}}""".format(
                        aiglobal=aiglobal,
                        init_i=init_i,
                        init_id=init_id))

        # gather instructions themselves.  This is a bit trickier
        # since I have one instruction per bit of each vector space
        # so I need to label the results of the element kernel and
        # the global outputs differently...

        el_tensor_count = 0
        ts_count = 0

        for ts in test_space.split():
            fspace_nr = fspace_to_number[ts]
            ltg_cur = ts.cell_node_map().values_with_halo
            ats_global = "A%d_global" % ts_count
            ltgts = "ltg_%d" % fspace_nr
            
            if ts.value_size == 1:
                ibf = "ibf_gather_%d" % el_tensor_count
                nbf = ltg_cur.shape[1]
                init_id = "init_%d" % ts_count
                aeltc = "A_%d" % el_tensor_count
                gather_inames.append(
                    (ibf, "0", str(nbf))
                )
                gather_insns.append(
                    """{atsgl}[{ltg}[iel,{ibf}]] = (
                       {atsgl}[{ltg}[iel,{ibf}]]
                         + {aeltc}[iel, {ibf}] ) {{dep={init}, atomic}}"""
                    .format(
                        atsgl=ats_global,
                        ibf=ibf, init=init_id,
                        ltg=ltgts,
                        aeltc=aeltc))
                el_tensor_count += 1
            else:
                for d in range(ts.value_size):
                    ibf = "ibf_gather_%d" % el_tensor_count
                    nbf = ltg_cur.shape[1]
                    init_id = "init_%d" % ts_count
                    aeltc = "A_%d" % el_tensor_count
                    gather_inames.append(
                        (ibf, "0", str(nbf))
                    )
                    gather_insns.append(
                        """{atsgl}[{ltg}[iel,{ibf}], {dim}] = (
                        {atsgl}[{ltg}[iel,{ibf}], {dim}]
                         + {aeltc}[iel, {ibf}] ) {{dep={init}, atomic}}"""
                    .format(
                        atsgl=ats_global, dim=d,
                        ibf=ibf, init=init_id,
                        ltg=ltgts,
                        aeltc=aeltc))
                    el_tensor_count += 1

            ts_count += 1

        # now create a gather full gather kernel out of this.
        g_dom = str.join(', ', [g[0] for g in gather_inames])

        g_bounds = ["{b} <= {a} < {c}".format(a=a, b=b, c=c)
                    for (a, b, c) in gather_inames]

        g_bd = str.join(" and ", g_bounds)

        g_insns = str.join("\n", gather_insns)

        # collect kernel arguments
        g_args = [lp.ValueArg("nelements", np.int32)]

        eltc = 0
        for i, ts in enumerate(test_space.split()):
            tssize = "A%d_size" % i
            aiglobal = "A%d_global" % i
            ltg = "ltg_%d" % fspace_to_number[ts]
            nbf = ts.cell_node_map().values_with_halo.shape[1]
            if ts.value_size == 1:
                shp = (tssize,)
            else:
                shp = (tssize, ts.value_size)

            g_args.append(lp.GlobalArg(
                ltg, np.int32, shape=lp.auto))
            if ts.value_size == 1:
                shp = (tssize,)
            else:
                shp = (tssize, ts.value_size)
            g_args.append(
                lp.GlobalArg(
                    aiglobal, np.float64, shape=shp, for_atomic=True))

            g_args.append(lp.ValueArg(tssize, np.int32))

            # the element kernels
            for d in range(ts.value_size):
                ai = "A_%d" % eltc
                g_args.append(
                    lp.GlobalArg(
                        ai, np.float64, shape=('nelements', nbf)))
                eltc += 1

        g_args.append("...")

        gather_knl = lp.make_kernel(
            "{{ [{gdom}]: {gbd}}}".format(gdom=g_dom, gbd=g_bd),
            g_insns,
            g_args)

        # fuse in gather kernel now
        ais = []
        eltc = 0
        for i, ts in enumerate(test_space.split()):
            for d in range(ts.value_size):
                ais.append("A_%d" % eltc)
                eltc += 1

        data_flow = [(ai, 0, 1) for ai in ais]

        knl = lp.fuse_kernels(
            (knl, gather_knl),
            data_flow=data_flow)

        for ai in ais:
            knl = lp.assignment_to_subst(knl, ai)
            
        knl = lp.infer_unknown_types(knl)
        knl = lp.make_reduction_inames_unique(knl)
        self.knl = knl

        # Set up arguments for the kernel
        kernel_args["coords_global"] = coords.dat._data
        kernel_args["coords_global_len"] = coords.dat._data.shape[0]

        for i, coeff in enumerate(fcoeffs):
            kernel_args["w_"+str(i)+"_global"] = coeff.dat._data
            kernel_args["w_"+str(i)+"_global_len"] = coeff.dat._data.shape[0]

        for i, fspace in enumerate(fspaces_in_form):
            kernel_args["ltg_"+str(i)] = fspace.cell_node_map().values_with_halo

        for i, yi in enumerate(self._y.split()):
            tssize = "A%d_size" % i
            if len(yi.dat._data.shape) > 2:
                1/0
            kernel_args[tssize] = yi.dat._data.shape[0]

        self.kernel_args = kernel_args

        # Now get device/queue set up for mat-vec
        import pyopencl as cl
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

    def mult(self, mat, X, Y):

        with self._x.dat.vec as v:
            X.copy(v)

        # if we are a block on the diagonal, then the matrix has an
        # identity block corresponding to the Dirichlet boundary conditions.
        # our algorithm in this case is to save the BC values, zero them
        # out before computing the action so that they don't pollute
        # anything, and then set the values into the result.
        # This has the effect of applying
        # [ A_II 0 ; 0 I ] where A_II is the block corresponding only to
        # non-fixed dofs and I is the identity block on the fixed dofs.

        # If we are not, then the matrix just has 0s in the rows and columns.

        if self.on_diag:  # stash BC values for later
            with self._xbc.dat.vec as v:
                X.copy(v)

        for bc in self.col_bcs:
            bc.zero(self._x)

        # from firedrake.parloops import READ, INC
        # self._x.dat.global_to_local_begin(READ)
        # self._x.dat.global_to_local_end(READ)            

        # This loopy kernel does the scatter plus element integration.
        evt, As = self.knl(self.queue, **self.kernel_args)

        for Ai, y in zip(As, self._y.split()):
            y.dat._data[:] = np.reshape(Ai, y.dat._data.shape)

        # self._y.dat.local_to_global_begin(INC)
        # self._y.dat.local_to_global_end(INC)

        # This sets the essential boundary condition values on the
        # result.
        if self.on_diag:
            for bc in self.row_bcs:
                bc.set(self._y, self._xbc)
        else:
            for bc in self.row_bcs:
                bc.zero(self._y)

        with self._y.dat.vec_ro as v:
            v.copy(Y)

    def multTranspose(self, mat, Y, X):
        # As for mult, just everything swapped round.
        from firedrake.assemble import assemble

        with self._y.dat.vec as v:
            Y.copy(v)
        if self.on_diag:  # stash BC values for later
            with self._ybc.dat.vec as v:
                Y.copy(v)

        for bc in self.row_bcs:
            bc.zero(self._y)

        assemble(self.actionT, self._x,
                 form_compiler_parameters=self.fc_params)

        if self.on_diag:
            for bc in self.col_bcs:
                bc.set(self._x, self._ybc)
        else:
            for bc in self.col_bcs:
                bc.zero(self._x)

        with self._x.dat.vec_ro as v:
            v.copy(X)

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        typ = viewer.getType()
        if typ != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII("Firedrake matrix-free operator %s\n" %
                           type(self).__name__)

    # Now, to enable fieldsplit preconditioners, we need to enable submatrix
    # extraction for our custom matrix type.  Note that we are splitting UFL
    # and index sets rather than an assembled matrix, keeping matrix
    # assembly deferred as long as possible.
    def getSubMatrix(self, mat, row_is, col_is, target=None):
        if target is not None:
            # Repeat call, just return the matrix, since we don't
            # actually assemble in here.
            target.assemble()
            return target
        from firedrake import DirichletBC

        # These are the sets of ISes of which the the row and column
        # space consist.
        row_ises = self._y.function_space().dof_dset.field_ises
        col_ises = self._x.function_space().dof_dset.field_ises

        row_inds = find_sub_block(row_is, row_ises)
        if row_is == col_is and row_ises == col_ises:
            col_inds = row_inds
        else:
            col_inds = find_sub_block(col_is, col_ises)

        asub = ExtractSubBlock().split(self.a,
                                       argument_indices=(row_inds, col_inds))
        Wrow = asub.arguments()[0].function_space()
        Wcol = asub.arguments()[1].function_space()

        row_bcs = []
        col_bcs = []

        for bc in self.row_bcs:
            for i, r in enumerate(row_inds):
                if bc.function_space().index == r:
                    row_bcs.append(DirichletBC(Wrow.split()[i],
                                               bc.function_arg,
                                               bc.sub_domain,
                                               method=bc.method))

        if Wrow == Wcol and row_inds == col_inds and self.row_bcs == self.col_bcs:
            col_bcs = row_bcs
        else:
            for bc in self.col_bcs:
                for i, c in enumerate(col_inds):
                    if bc.function_space().index == c:
                        col_bcs.append(DirichletBC(Wcol.split()[i],
                                                   bc.function_arg,
                                                   bc.sub_domain,
                                                   method=bc.method))
        submat_ctx = ImplicitMatrixContext(asub,
                                           row_bcs=row_bcs,
                                           col_bcs=col_bcs,
                                           fc_params=self.fc_params,
                                           context=self.context)
        submat_ctx.on_diag = self.on_diag and row_inds == col_inds
        submat = PETSc.Mat().create()
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes),
                        bsize=submat_ctx.block_size)
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat


def compile_form_loopy(form):
    from firedrake.formmanipulation import split_form
    from tsfc import compile_form, tsfc_to_loopy

    tsfc_kernels = [krnl for (idx, f) in split_form(form)
                    for krnl in compile_form(f)]

    for k in tsfc_kernels:
        print(k.ast)

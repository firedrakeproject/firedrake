from __future__ import absolute_import

from ufl import action

from firedrake.ufl_expr import adjoint
from firedrake.formmanipulation import ExtractSubBlock

from firedrake import Function
from loopy.symbolic import IdentityMapper

from firedrake.petsc import PETSc

import six

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
            self._xbc = function.Function(trial_space)
        if len(self.col_bcs) > 0:
            self._ybc = function.Function(test_space)

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

        for bc in self.col_bcs:
            bc.zero(self._x)

        self._assemble_action()

        # This sets the essential boundary condition values on the
        # result.
        if self.on_diag:
            if len(self.row_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._xbc.dat.vec as v:
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
        with self._y.dat.vec as v:
            Y.copy(v)

        for bc in self.row_bcs:
            bc.zero(self._y)

        self._assemble_actionT()

        if self.on_diag:
            if len(self.col_bcs) > 0:
                # TODO, can we avoid the copy?
                with self._ybc.dat.vec as v:
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
                                           appctx=self.appctx)
        submat_ctx.on_diag = self.on_diag and row_inds == col_inds
        submat = PETSc.Mat().create(comm=mat.comm)
        submat.setType("python")
        submat.setSizes((submat_ctx.row_sizes, submat_ctx.col_sizes),
                        bsize=submat_ctx.block_size)
        submat.setPythonContext(submat_ctx)
        submat.setUp()

        return submat


# Loopy business
class TsfcCoefficientIndexUnflattener(IdentityMapper):
    def __init__(self, variable_to_dim_nbf):
        self.variable_to_dim_nbf = variable_to_dim_nbf

    def map_subscript(self, expr):
        if expr.aggregate.name in self.variable_to_dim_nbf:
            assert len(expr.index_tuple) == 3
            assert expr.index_tuple[2] == 0

            lin_nbf_dim = expr.index_tuple[1]

            # print type(lin_nbf_dim)
            # Seems true, but safe to delete
            # assert isinstance(lin_nbf_dim, int)

            dim, nbf = self.variable_to_dim_nbf[expr.aggregate.name]

            inbf = lin_nbf_dim % nbf
            idim = lin_nbf_dim // nbf

            return expr.aggregate[expr.index[0], inbf, idim]
        else:
            return super(TsfcCoefficientIndexUnflattener, self).map_subscript(expr)


def adjust_tsfc_shapes(knl, variable_to_dim_nbf):
    import loopy as lp

    args = []
    for arg in knl.args:
        if arg.name in variable_to_dim_nbf:
            dim, nbf = variable_to_dim_nbf[arg.name]
            assert arg.shape[1] == dim*nbf
            assert len(arg.shape) == 2 or arg.shape[2] == 1

            args.append(lp.GlobalArg(
                arg.name,
                shape=(arg.shape[0], nbf, dim),
                order="C"))
        else:
            args.append(arg)

    return knl.copy(args=args)


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
            rbsize = test_space.dim
        else:
            rbsize = 1
        if len(trial_space) == 1:
            cbsize = trial_space.dim
        else:
            cbsize = 1

        self.block_size = (rbsize, cbsize)

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

        # Now we build loopy kernels for the action
        from tsfc import compile_form, tsfc_to_loopy
        import loopy as lp
        import numpy as np
        import string
        kernel, = compile_form(self.action)

        # A bit complicated since we are allowing a general number of
        # coefficients
        coeffs = self.action.coefficients()
        ws_labels = ["w_" + str(i) for i, _ in enumerate(coeffs)]
        things_to_batch = tuple(["A0"] + ws_labels + ["coords"])
        things_to_infer = string.join(things_to_batch, ",")
        knl = tsfc_to_loopy(kernel._ir)

        knl = lp.add_and_infer_dtypes(knl, {things_to_infer: np.float64})
        knl = lp.to_batched(knl, "nelements", things_to_batch,
                            batch_iname_prefix="iel")
        for rule in list(six.itervalues(knl.substitutions)):
            knl = lp.precompute(knl, rule.name, rule.arguments)

        mesh = test_space.mesh()
        coords = mesh.coordinates
        coord_fs = coords.function_space()

        variable_to_dim_nbf = {
            "coords": (coord_fs.dim,
                       coord_fs.cell_node_map().values.shape[1]),
        }
        for i, coeff in enumerate(coeffs):
            nbf = coeff.function_space().cell_node_map().values.shape[1]
            dim = coeff.function_space().dim
            variable_to_dim_nbf["w_"+str(i)] = (dim, nbf)

        unflatter = TsfcCoefficientIndexUnflattener(variable_to_dim_nbf)

        knl = knl.copy(
            instructions=[
                insn.with_transformed_expressions(unflatter) for insn in knl.instructions])

        # variable_to_dim_nbf["A0"] = (test_space.dim,
        #                              test_space.cell_node_map().values.shape[1])
        knl = adjust_tsfc_shapes(knl, variable_to_dim_nbf)

        # now set up space to hold scattered coefficients

        import pyopencl as cl
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

        # Create scatter kernels for each coefficient, then fuse

        # for coords and each coeff, fuse in a copy of scatter_knl,
        # then bind kernel_args to the non-scattered vectors.

        fspace_to_number = {}
        fspaces_in_form = []
        fspace_to_number[coord_fs] = 0
        fspaces_in_form.append(coord_fs)
        for coeff in coeffs:
            fs = coeff.function_space()
            if fs not in fspace_to_number:
                fspace_to_number[fs] = len(fspace_to_number)
                fspaces_in_form.append(fs)

        things_with_fspaces = (
            [("coords", coord_fs)]
            + [("w_%d" % i, coeff.function_space())
               for i, coeff in enumerate(coeffs)])
        for thing, fspace in things_with_fspaces:
            fspace_nr = fspace_to_number[fspace]
            ltg_cur = fspace.cell_node_map().values
            ibf = "ibf%d" % fspace_nr
            idim = "idim%d" % fspace_nr

            scatter_knl = lp.make_kernel(
                "{{ [iel, {ibf}, {idim}]:"
                "    0 <= iel<nelements "
                "and 0 <= {ibf} < {nbf} "
                "and 0 <= {idim} < {dim}}}"
                .format(
                    fnr=fspace_nr,
                    nbf=ltg_cur.shape[1],
                    dim=fspace.dim,
                    ibf=ibf,
                    idim=idim,
                ),
                """
                {thing}[iel, {ibf}, {idim}] = \
                    {thing}_global[{dim} * ltg{fnr}[iel, {ibf}] + {idim}]
                """.format(
                    thing=thing,
                    fnr=fspace_nr,
                    idim=idim,
                    ibf=ibf,
                    dim=fspace.dim)
            )

            scatter_knl = lp.add_dtypes(
                scatter_knl, {
                    "nelements": np.int32,
                    "ltg"+str(fspace_nr): np.int32,
                    thing+"_global": np.float64,
                })

            knl = lp.fuse_kernels(
                (scatter_knl, knl),
                data_flow=[(thing, 0, 1)])

            knl = lp.assignment_to_subst(knl, thing)

        knl = lp.add_dtypes(knl,
                            {"A0": np.float64})

        gather_knl = lp.make_kernel(
            "{{ [iel, ibf_A0, idim_A0]: 0 <= iel < nelements and 0 <= ibf_A0 < {nbf} and 0 <= idim_A0 < {dim} }}"
            .format(
                nbf=test_space.cell_node_map().values.shape[1],
                dim=test_space.dim
            ),
            """
            A0_global[{dim} * ltg_A0[iel, ibf_A0] + idim_A0] = (
                A0_global[{dim} * ltg_A0[iel, ibf_A0] + idim_A0]
                + A0[iel, ibf_A0 + {nbf}*idim_A0])  {{atomic}}
            """
            .format(
                dim=test_space.dim,
                nbf=test_space.cell_node_map().values.shape[1]
            ),
            [
                lp.GlobalArg("A0_global", np.float64, shape=(self._y.vector().size(),), for_atomic=True),
                lp.GlobalArg("ltg_A0", np.int32, shape=lp.auto),
                "..."])
        # A0writes = [ins for ins in knl.instructions
        #             if ins.assignee.aggregate.name == "A0"]
        # assert len(A0writes) == 1
        # # snip off the x = x + y and make it x = y.
        # insn = A0writes[0]
        # rvalue = insn.expression
        # newrvalue = rvalue.children[1]
        # insn.expression = newrvalue

        gather_knl = lp.add_dtypes(gather_knl, {
            "nelements": np.int32,
            "A0": np.float64,
        })
        knl = lp.fuse_kernels(
            (knl, gather_knl),
            data_flow=[("A0", 0, 1)])

        knl = lp.assignment_to_subst(knl, "A0")
        knl = lp.infer_unknown_types(knl)

        knl = lp.make_reduction_inames_unique(knl)

        self.knl = knl

        # Set up arguments for the kernel
        kernel_args = {}
        with coords.dat.vec as v:
            kernel_args["coords_global"] = v.array
        for i, coeff in enumerate(coeffs):
            with coeff.dat.vec as v:
                kernel_args["w_"+str(i)+"_global"] = v.array

        for i, fspace in enumerate(fspaces_in_form):
            kernel_args["ltg"+str(i)] = fspace.cell_node_map().values

        kernel_args["ltg_A0"] = test_space.cell_node_map().values

        self.kernel_args = kernel_args

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

        # This loopy kernel does the scatter plus element integration.

        evt, (A0, ) = self.knl(self.queue, **self.kernel_args)

        # now we need to copy A0's values into y
        with self._y.dat.vec as ycur:
            ycur[:] = A0[:]

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


# Loopy business
class TsfcCoefficientIndexUnflattener(IdentityMapper):
    def __init__(self, variable_to_dim_nbf):
        self.variable_to_dim_nbf = variable_to_dim_nbf

    def map_subscript(self, expr):
        if expr.aggregate.name in self.variable_to_dim_nbf:
            assert len(expr.index_tuple) == 3
            assert expr.index_tuple[2] == 0

            lin_nbf_dim = expr.index_tuple[1]

            # print type(lin_nbf_dim)
            # Seems true, but safe to delete
            # assert isinstance(lin_nbf_dim, int)

            dim, nbf = self.variable_to_dim_nbf[expr.aggregate.name]

            inbf = lin_nbf_dim % nbf
            idim = lin_nbf_dim // nbf

            return expr.aggregate[expr.index[0], inbf, idim]
        else:
            return super(TsfcCoefficientIndexUnflattener, self).map_subscript(expr)


def adjust_tsfc_shapes(knl, variable_to_dim_nbf):
    import loopy as lp

    args = []
    for arg in knl.args:
        if arg.name in variable_to_dim_nbf:
            dim, nbf = variable_to_dim_nbf[arg.name]
            assert arg.shape[1] == dim*nbf
            assert arg.shape[2] == 1

            args.append(lp.GlobalArg(
                arg.name,
                shape=(arg.shape[0], nbf, dim),
                order="C"))
        else:
            args.append(arg)

    return knl.copy(args=args)


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
            rbsize = test_space.dim
        else:
            rbsize = 1
        if len(trial_space) == 1:
            cbsize = trial_space.dim
        else:
            cbsize = 1

        self.block_size = (rbsize, cbsize)

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

        # Now we build loopy kernels for the action
        from tsfc import compile_form, tsfc_to_loopy
        import loopy as lp
        import numpy as np
        import string
        kernel, = compile_form(self.action)

        # A bit complicated since we are allowing a general number of
        # coefficients
        coeffs = self.action.coefficients()
        ws_labels = ["w_" + str(i) for i, _ in enumerate(coeffs)]
        things_to_batch = tuple(["A0"] + ws_labels + ["coords"])
        things_to_infer = string.join(things_to_batch, ",")
        knl = tsfc_to_loopy(kernel._ir)

        knl = lp.add_and_infer_dtypes(knl, {things_to_infer: np.float64})
        knl = lp.to_batched(knl, "nelements", things_to_batch,
                            batch_iname_prefix="iel")
        for rule in list(six.itervalues(knl.substitutions)):
            knl = lp.precompute(knl, rule.name, rule.arguments)

        # FIXME: this needs to be general!
        variable_to_dim_nbf = {
            "coords": (2, 3)
            }
        for i, coeff in enumerate(coeffs):
            nbf = coeff.function_space().cell_node_map().values.shape[1]
            dim = coeff.function_space().dim
            variable_to_dim_nbf["w_"+str(i)] = (dim, nbf)

        unflatter = TsfcCoefficientIndexUnflattener(variable_to_dim_nbf)
        knl = knl.copy(
            instructions=[
            insn.with_transformed_expressions(unflatter) for insn in knl.instructions])

        knl = adjust_tsfc_shapes(knl, variable_to_dim_nbf)

        # now set up space to hold scattered coefficients
        # FIXME: check for VectorElements.
        def c_to_shp(c):
            return tuple(
                list(c.function_space().cell_node_map().values.shape)+[1])

        self.scattered_coeffs = [np.zeros(c_to_shp(c))
                                 for c in self.action.coefficients()]

        # also need scattered coordinate fields.
        mesh = test_space.mesh()
        coords = mesh.coordinates
        coord_fs = coords.function_space()

        import pyopencl as cl
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

        # Create scatter kernels for each coefficient, then fuse

        # for coords and each coeff, fuse in a copy of scatter_knl,
        # then bind kernel_args to the non-scattered vectors.

        fspace_to_number = {}
        fspaces_in_form = []
        fspace_to_number[coord_fs] = 0
        fspaces_in_form.append(coord_fs)
        for coeff in coeffs:
            fs = coeff.function_space()
            if fs not in fspace_to_number:
                fspace_to_number[fs] = len(fspace_to_number)
                fspaces_in_form.append(fs)

        things_with_fspaces = (
            [("coords", coord_fs)]
            + [("w_%d" % i, coeff.function_space())
               for i, coeff in enumerate(coeffs)])
        for thing, fspace in things_with_fspaces:
            fspace_nr = fspace_to_number[fspace]
            ltg_cur = fspace.cell_node_map().values
            ibf = "ibf%d" % fspace_nr
            idim = "idim%d" % fspace_nr

            scatter_knl = lp.make_kernel(
                "{{ [iel, {ibf}, {idim}]:"
                "    0 <= iel<nelements "
                "and 0 <= {ibf} < {nbf} "
                "and 0 <= {idim} < {dim}}}"
                .format(
                    fnr=fspace_nr,
                    nbf=ltg_cur.shape[1],
                    dim=fspace.dim,
                    ibf=ibf,
                    idim=idim,
                ),
                """
                {thing}[iel, {ibf}, {idim}] = \
                    {thing}_global[{dim} * ltg{fnr}[iel, {ibf}] + {idim}]
                """.format(
                    thing=thing,
                    fnr=fspace_nr,
                    idim=idim,
                    ibf=ibf,
                    dim=fspace.dim)
            )

            scatter_knl = lp.add_dtypes(
                scatter_knl, {
                    "nelements": np.int32,
                    "ltg"+str(fspace_nr): np.int32,
                    thing+"_global": np.float64,
                    })


            knl = lp.fuse_kernels(
                (scatter_knl, knl),
                data_flow=[(thing, 0, 1)])

            knl = lp.assignment_to_subst(knl, thing)

        # FIXME A0 not yet vector-ready
        gather_knl = lp.make_kernel(
            "{{ [iel, ibf_A0, idim_A0]: 0 <= iel < nelements and 0 <= ibf_A0 < {nbf} and 0 <= idim_A0 < {dim} }}"
            .format(
                nbf=test_space.cell_node_map().values.shape[1],
                dim=test_space.dim
                ),
            """
            A0_global[{dim} * ltg_A0[iel, ibf_A0] + idim_A0] = (
                A0_global[{dim} * ltg_A0[iel, ibf_A0] + idim_A0]
                + A0[iel, ibf_A0])  {{atomic}}
            """
            .format(
                dim=test_space.dim
                ),
            [
                lp.GlobalArg("A0_global", np.float64, shape=Function(test_space).dat.shape, for_atomic=True),
                lp.GlobalArg("ltg_A0", np.int32, shape=lp.auto),
                "..."])
        # A0writes = [ins for ins in knl.instructions
        #             if ins.assignee.aggregate.name == "A0"]
        # assert len(A0writes) == 1
        # # snip off the x = x + y and make it x = y.
        # insn = A0writes[0]
        # rvalue = insn.expression
        # newrvalue = rvalue.children[1]
        # insn.expression = newrvalue

        gather_knl = lp.add_dtypes(gather_knl, {
            "nelements": np.int32,
            "A0": np.float64,
            })
        knl = lp.fuse_kernels(
            (knl, gather_knl),
            data_flow=[("A0", 0, 1)])

        knl = lp.assignment_to_subst(knl, "A0")
        knl = lp.infer_unknown_types(knl)

        knl = lp.make_reduction_inames_unique(knl)
        

        self.knl = knl

        # Set up arguments for the kernel
        kernel_args = {}
        with coords.dat.vec as v:
            kernel_args["coords_global"] = v.array
        for i, coeff in enumerate(coeffs):
            with coeff.dat.vec as v:
                kernel_args["w_"+str(i)+"_global"] = v.array

        for i, fspace in enumerate(fspaces_in_form):
            kernel_args["ltg"+str(i)] = fspace.cell_node_map().values

        kernel_args["ltg_A0"] = test_space.cell_node_map().values

        self.kernel_args = kernel_args

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

        # This loopy kernel does the scatter plus element integration.

        evt, (A0, ) = self.knl(self.queue, **self.kernel_args)

        # now we need to copy A0's values into y
        with self._y.dat.vec as ycur:
            ycur[:] = A0[:]

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


# Loopy business
class TsfcCoefficientIndexUnflattener(IdentityMapper):
    def __init__(self, variable_to_dim_nbf):
        self.variable_to_dim_nbf = variable_to_dim_nbf

    def map_subscript(self, expr):
        if expr.aggregate.name in self.variable_to_dim_nbf:
            assert len(expr.index_tuple) == 3
            assert expr.index_tuple[2] == 0

            lin_nbf_dim = expr.index_tuple[1]

            # print type(lin_nbf_dim)
            # Seems true, but safe to delete
            # assert isinstance(lin_nbf_dim, int)

            dim, nbf = self.variable_to_dim_nbf[expr.aggregate.name]

            inbf = lin_nbf_dim % nbf
            idim = lin_nbf_dim // nbf

            return expr.aggregate[expr.index[0], inbf, idim]
        else:
            return super(TsfcCoefficientIndexUnflattener, self).map_subscript(expr)


def adjust_tsfc_shapes(knl, variable_to_dim_nbf):
    import loopy as lp

    args = []
    for arg in knl.args:
        if arg.name in variable_to_dim_nbf:
            dim, nbf = variable_to_dim_nbf[arg.name]
            assert arg.shape[1] == dim*nbf
            assert arg.shape[2] == 1

            args.append(lp.GlobalArg(
                arg.name,
                shape=(arg.shape[0], nbf, dim),
                order="C"))
        else:
            args.append(arg)

    return knl.copy(args=args)


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
            rbsize = test_space.dim
        else:
            rbsize = 1
        if len(trial_space) == 1:
            cbsize = trial_space.dim
        else:
            cbsize = 1

        self.block_size = (rbsize, cbsize)

        self.action = action(self.a, self._x)
        self.actionT = action(self.aT, self._y)

        # Now we build loopy kernels for the action
        from tsfc import compile_form, tsfc_to_loopy
        import loopy as lp
        import numpy as np
        import string
        kernel, = compile_form(self.action)

        # A bit complicated since we are allowing a general number of
        # coefficients
        coeffs = self.action.coefficients()
        ws_labels = ["w_" + str(i) for i, _ in enumerate(coeffs)]
        things_to_batch = tuple(["A0"] + ws_labels + ["coords"])
        things_to_infer = string.join(things_to_batch, ",")
        knl = tsfc_to_loopy(kernel._ir)

        knl = lp.add_and_infer_dtypes(knl, {things_to_infer: np.float64})
        knl = lp.to_batched(knl, "nelements", things_to_batch,
                            batch_iname_prefix="iel")
        for rule in list(six.itervalues(knl.substitutions)):
            knl = lp.precompute(knl, rule.name, rule.arguments)

        # FIXME: this needs to be general!
        variable_to_dim_nbf = {
            "coords": (2, 3)
            }
        for i, coeff in enumerate(coeffs):
            nbf = coeff.function_space().cell_node_map().values.shape[1]
            dim = coeff.function_space().dim
            variable_to_dim_nbf["w_"+str(i)] = (dim, nbf)

        unflatter = TsfcCoefficientIndexUnflattener(variable_to_dim_nbf)
        knl = knl.copy(
            instructions=[
            insn.with_transformed_expressions(unflatter) for insn in knl.instructions])

        knl = adjust_tsfc_shapes(knl, variable_to_dim_nbf)

        # now set up space to hold scattered coefficients
        # FIXME: check for VectorElements.
        def c_to_shp(c):
            return tuple(
                list(c.function_space().cell_node_map().values.shape)+[1])

        self.scattered_coeffs = [np.zeros(c_to_shp(c))
                                 for c in self.action.coefficients()]

        # also need scattered coordinate fields.
        mesh = test_space.mesh()
        coords = mesh.coordinates
        coord_fs = coords.function_space()

        import pyopencl as cl
        self.ctx = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.ctx)

        # Create scatter kernels for each coefficient, then fuse

        # for coords and each coeff, fuse in a copy of scatter_knl,
        # then bind kernel_args to the non-scattered vectors.

        fspace_to_number = {}
        fspaces_in_form = []
        fspace_to_number[coord_fs] = 0
        fspaces_in_form.append(coord_fs)
        for coeff in coeffs:
            fs = coeff.function_space()
            if fs not in fspace_to_number:
                fspace_to_number[fs] = len(fspace_to_number)
                fspaces_in_form.append(fs)

        things_with_fspaces = (
            [("coords", coord_fs)]
            + [("w_%d" % i, coeff.function_space())
               for i, coeff in enumerate(coeffs)])
        for thing, fspace in things_with_fspaces:
            fspace_nr = fspace_to_number[fspace]
            ltg_cur = fspace.cell_node_map().values
            ibf = "ibf%d" % fspace_nr
            idim = "idim%d" % fspace_nr

            scatter_knl = lp.make_kernel(
                "{{ [iel, {ibf}, {idim}]:"
                "    0 <= iel<nelements "
                "and 0 <= {ibf} < {nbf} "
                "and 0 <= {idim} < {dim}}}"
                .format(
                    fnr=fspace_nr,
                    nbf=ltg_cur.shape[1],
                    dim=fspace.dim,
                    ibf=ibf,
                    idim=idim,
                ),
                """
                {thing}[iel, {ibf}, {idim}] = \
                    {thing}_global[{dim} * ltg{fnr}[iel, {ibf}] + {idim}]
                """.format(
                    thing=thing,
                    fnr=fspace_nr,
                    idim=idim,
                    ibf=ibf,
                    dim=fspace.dim)
            )

            scatter_knl = lp.add_dtypes(
                scatter_knl, {
                    "nelements": np.int32,
                    "ltg"+str(fspace_nr): np.int32,
                    thing+"_global": np.float64,
                    })


            knl = lp.fuse_kernels(
                (scatter_knl, knl),
                data_flow=[(thing, 0, 1)])

            knl = lp.assignment_to_subst(knl, thing)

        # FIXME A0 not yet vector-ready
        gather_knl = lp.make_kernel(
            "{{ [iel, ibf_A0, idim_A0]: 0 <= iel < nelements and 0 <= ibf_A0 < {nbf} and 0 <= idim_A0 < {dim} }}"
            .format(
                nbf=test_space.cell_node_map().values.shape[1],
                dim=test_space.dim
                ),
            """
            A0_global[{dim} * ltg_A0[iel, ibf_A0] + idim_A0] = (
                A0_global[{dim} * ltg_A0[iel, ibf_A0] + idim_A0]
                + A0[iel, ibf_A0])  {{atomic}}
            """
            .format(
                dim=test_space.dim
                ),
            [
                lp.GlobalArg("A0_global", np.float64, shape=Function(test_space).dat.shape, for_atomic=True),
                lp.GlobalArg("ltg_A0", np.int32, shape=lp.auto),
                "..."])
        # A0writes = [ins for ins in knl.instructions
        #             if ins.assignee.aggregate.name == "A0"]
        # assert len(A0writes) == 1
        # # snip off the x = x + y and make it x = y.
        # insn = A0writes[0]
        # rvalue = insn.expression
        # newrvalue = rvalue.children[1]
        # insn.expression = newrvalue

        gather_knl = lp.add_dtypes(gather_knl, {
            "nelements": np.int32,
            "A0": np.float64,
            })
        knl = lp.fuse_kernels(
            (knl, gather_knl),
            data_flow=[("A0", 0, 1)])

        knl = lp.assignment_to_subst(knl, "A0")
        knl = lp.infer_unknown_types(knl)

        knl = lp.make_reduction_inames_unique(knl)
        

        self.knl = knl

        # Set up arguments for the kernel
        kernel_args = {}
        with coords.dat.vec as v:
            kernel_args["coords_global"] = v.array
        for i, coeff in enumerate(coeffs):
            with coeff.dat.vec as v:
                kernel_args["w_"+str(i)+"_global"] = v.array

        for i, fspace in enumerate(fspaces_in_form):
            kernel_args["ltg"+str(i)] = fspace.cell_node_map().values

        kernel_args["ltg_A0"] = test_space.cell_node_map().values

        self.kernel_args = kernel_args

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

        # This loopy kernel does the scatter plus element integration.

        evt, (A0, ) = self.knl(self.queue, **self.kernel_args)

        # now we need to copy A0's values into y
        with self._y.dat.vec as ycur:
            ycur[:] = A0[:]

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

from firedrake.petsc import PETSc
from firedrake.ensemble.ensemble_function import EnsembleFunction
from firedrake.ensemble.ensemble_functionspace import EnsembleFunctionSpaceBase


class EnsembleMatBase:
    def __init__(self, row_space, col_space):
        name = type(self).__name__
        if not isinstance(row_space, EnsembleFunctionSpaceBase):
            raise ValueError(
                f"{name} row_space must be EnsembleFunctionSpace not {type(row_space).__name__}")
        if not isinstance(col_space, EnsembleFunctionSpaceBase):
            raise ValueError(
                f"{name} col_space must be EnsembleFunctionSpace not {type(col_space).__name__}")

        if row_space.ensemble != col_space.ensemble:
            raise ValueError(
                f"{name} row and column spaces must have the same Ensemble")

        self.ensemble = row_space.ensemble
        self.row_space = row_space
        self.col_space = col_space

        # input/output Vecs will be copied in/out of these
        # so that base classes can implement mult only in
        # terms of Ensemble objects not Vecs.
        self.x = EnsembleFunction(self.row_space)
        self.y = EnsembleFunction(self.col_space)

    def mult(self, A, x, y):
        with self.x.vec_wo() as xvec:
            x.copy(result=xvec)

        self.mult_impl(A, self.x, self.y)

        with self.y.vec_ro() as yvec:
            yvec.copy(result=y)


class EnsembleBlockDiagonalMat(EnsembleMatBase):
    def __init__(self, block_mats, row_space, col_space):
        super().__init__(row_space, col_space)
        self.block_mats = block_mats

        if self.row_space.nlocal_spaces != self.col_space.nlocal_spaces:
            raise ValueError(
                "EnsembleBlockDiagonalMat row and col spaces must be the same length,"
                f" not {row_space.nlocal_spaces=} and {col_space.nlocal_spaces=}")

        if len(self.block_mats) != self.row_space.nlocal_spaces:
            raise ValueError(
                f"EnsembleBlockDiagonalMat requires one submatrix for each of the"
                f" {self.row_space.nlocal_spaces} local subfunctions of the EnsembleFunctionSpace,"
                f" but only {len(self.block_mats)} provided.")

        for i, (Vrow, Vcol, block) in enumerate(zip(self.row_space.local_spaces,
                                                    self.col_space.local_spaces,
                                                    self.block_mats)):
            # number of columns is row length, and vice-versa
            vc_sizes = Vrow.dof_dset.layout_vec.sizes
            vr_sizes = Vcol.dof_dset.layout_vec.sizes
            mr_sizes, mc_sizes = block.sizes
            if (vr_sizes[0] != mr_sizes[0]) or (vr_sizes[1] != mr_sizes[1]):
                raise ValueError(
                    f"Row sizes {mr_sizes} of block {i} and {vr_sizes} of row_space {i} of EnsembleBlockDiagonalMat must match.")
            if (vc_sizes[0] != mc_sizes[0]) or (vc_sizes[1] != mc_sizes[1]):
                raise ValueError(
                    f"Col sizes of block {i} and col_space {i} of EnsembleBlockDiagonalMat must match.")

    def mult_impl(self, A, x, y):
        for block, xsub, ysub in zip(self.block_mats,
                                     x.subfunctions,
                                     y.subfunctions):
            with xsub.dat.vec_ro as xvec, ysub.dat.vec_wo as yvec:
                block.mult(xvec, yvec)

    def setUp(self, mat):
        for bmat in self.block_mats:
            bmat.setUp()

    def view(self, mat, viewer=None):
        if viewer is None:
            return
        if viewer.getType() != PETSc.Viewer.Type.ASCII:
            return
        viewer.printfASCII(f"  firedrake block diagonal Ensemble matrix: {type(self).__name__}\n")
        viewer.printfASCII(f"  Number of blocks = {self.col_space.nglobal_spaces}, Number of ensemble ranks = {self.ensemble.ensemble_size}\n")

        if viewer.getFormat() != PETSc.Viewer.Format.ASCII_INFO_DETAIL:
            viewer.printfASCII("  Local information for first block is in the following Mat objects on rank 0:\n")
            prefix = mat.getOptionsPrefix() or ""
            viewer.printfASCII(f"  Use -{prefix}ksp_view ::ascii_info_detail to display information for all blocks\n")
            subviewer = viewer.getSubViewer(self.ensemble.comm)
            if self.ensemble.ensemble_rank == 0:
                subviewer.pushASCIITab()
                self.block_mats[0].view(subviewer)
                subviewer.popASCIITab()
            viewer.restoreSubViewer(subviewer)
            # Comment taken from PCView_BJacobi in https://petsc.org/release/src/ksp/pc/impls/bjacobi/bjacobi.c.html#PCBJACOBI
            # extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer()
            viewer.popASCIISynchronized()

        else:
            viewer.pushASCIISynchronized()
            viewer.printfASCII("  Local information for each block is in the following Mat objects:\n")
            viewer.pushASCIITab()
            subviewer = viewer.getSubViewer(self.ensemble.comm)
            r = self.ensemble.ensemble_rank
            offset = self.col_space.global_spaces_offset
            subviewer.printfASCII(f"[{r}] number of local blocks = {self.col_space.nlocal_spaces}, first local block number = {offset}\n")
            for i, submat in enumerate(self.block_mats):
                subviewer.printfASCII(f"[{r}] local block number {i}, global block number {offset + i}\n")
                submat.view(subviewer)
                subviewer.printfASCII("- - - - - - - - - - - - - - - - - -\n")
            viewer.restoreSubViewer(subviewer)
            viewer.popASCIITab()
            viewer.popASCIISynchronized()


def EnsembleBlockDiagonalMatrix(block_mats, row_space, col_space):
    ctx = EnsembleBlockDiagonalMat(block_mats, row_space, col_space)

    # number of columns is row length, and vice-versa
    ncols = ctx.col_space.layout_vec.getSizes()
    nrows = ctx.row_space.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (ncols, nrows), ctx,
        comm=ctx.ensemble.global_comm)
    mat.setUp()
    mat.assemble()
    return mat

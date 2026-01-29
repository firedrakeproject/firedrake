from typing import Iterable
from firedrake.petsc import PETSc
from firedrake.ensemble.ensemble_function import EnsembleFunction, EnsembleFunctionBase
from firedrake.ensemble.ensemble_functionspace import EnsembleFunctionSpaceBase


class EnsembleMatCtxBase:
    """
    Base class for python type Mats defined over an :class:`~.ensemble.Ensemble`.

    Parameters
    ----------
    row_space :
        The function space that the matrix acts on.
        Must have the same number of subspaces on each ensemble rank as col_space.
    col_space :
        The function space for the result of the matrix action.
        Must have the same number of subspaces on each ensemble rank as row_space.

    Notes
    -----
    The main use of this base class is to enable users to implement the matrix
    action as acting on and resulting in an :class:`~.ensemble_function.EnsembleFunction`.
    This is done by implementing the ``mult_impl`` method.

    See Also
    --------
    .ensemble_pc.EnsemblePCBase
    """
    def __init__(self, row_space: EnsembleFunctionSpaceBase,
                 col_space: EnsembleFunctionSpaceBase):
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
        """Apply the action of the matrix to x, putting the result in y.

        This method will be called by PETSc with x and y as Vecs, and acts
        as a wrapper around the ``mult_impl`` method which has x and y as
        EnsembleFunction for convenience.
        y is not guaranteed to be zero on entry.

        Parameters
        ----------
        A : PETSc.Mat
            The PETSc matrix that self is the python context of.
        x : PETSc.Vec
            The vector acted on by the matrix.
        y : PETSc.Vec
            The result of the matrix action.

        See Also
        --------
        EnsembleMatCtxBase.mult_impl
        """
        with self.x.vec_wo() as xvec:
            x.copy(result=xvec)

        self.mult_impl(A, self.x, self.y)

        with self.y.vec_ro() as yvec:
            yvec.copy(result=y)

    def mult_impl(self, A, x: EnsembleFunctionBase, y: EnsembleFunctionBase):
        """Apply the action of the matrix to x, putting the result in y.

        y is not guaranteed to be zero on entry.
        This is a convenience method allowing the matrix action to be
        implemented in terms of EnsembleFunction input and outputs by
        inheriting classes.

        Parameters
        ----------
        A : PETSc.Mat
            The PETSc matrix that self is the python context of.
        x :
            The vector acted on by the matrix.
        y :
            The result of the matrix action.

        See Also
        --------
        EnsembleMatCtxBase.mult
        """
        raise NotImplementedError


class EnsembleBlockDiagonalMatCtx(EnsembleMatCtxBase):
    """
    A python Mat context for a block diagonal matrix defined over an :class:`~.ensemble.Ensemble`.
    Each block acts on a single subspace of an :class:`~.ensemble_functionspace.EnsembleFunctionSpace`.

    Parameters
    ----------
    block_mats : Iterable[PETSc.Mat]
        The PETSc Mats for each block. On each ensemble rank there must be as many
        Mats as there are local subspaces of ``row_space`` and ``col_space``, and
        the Mat sizes must match the sizes of the corresponding subspaces.
    row_space :
        The function space that the matrix acts on.
        Must have the same number of subspaces on each ensemble rank as col_space.
    col_space :
        The function space for the result of the matrix action.
        Must have the same number of subspaces on each ensemble rank as row_space.

    Notes
    -----
    This is a python context, not an actual PETSc.Mat. To create the corresponding
    PETSc.Mat users should call :func:`~.EnsembleBlockDiagonalMat`.

    See Also
    --------
    EnsembleBlockDiagonalMat
    ~.ensemble_pc.EnsembleBJacobiPC
    """
    def __init__(self, block_mats: Iterable,
                 row_space: EnsembleFunctionSpaceBase,
                 col_space: EnsembleFunctionSpaceBase):
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
            if not isinstance(block, PETSc.Mat):
                raise TypeError(
                    f"Block {i} must be a PETSc.Mat not a {type(block).__name__}.\n"
                    "Did you mean to use assemble(block).petscmat instead?")
            # number of columns is row length, and vice-versa
            vr_sizes = Vrow.template_vec.sizes
            vc_sizes = Vcol.template_vec.sizes
            mc_sizes, mr_sizes = block.sizes
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


def EnsembleBlockDiagonalMat(block_mats: Iterable,
                             row_space: EnsembleFunctionSpaceBase,
                             col_space: EnsembleFunctionSpaceBase):
    """
    A Mat for a block diagonal matrix defined over an :class:`~.ensemble.Ensemble`.
    Each block acts on a single subspace of an :class:`~.ensemble_functionspace.EnsembleFunctionSpace`.
    This is a convenience function to create a PETSc.Mat with a :class:`.EnsembleBlockDiagonalMatCtx` Python context.

    Parameters
    ----------
    block_mats : Iterable[PETSc.Mat]
        The PETSc Mats for each block. On each ensemble rank there must be as many
        Mats as there are local subspaces of ``row_space`` and ``col_space``, and
        the Mat sizes must match the sizes of the corresponding subspaces.
    row_space :
        The function space that the matrix acts on.
        Must have the same number of subspaces on each ensemble rank as col_space.
    col_space :
        The function space for the result of the matrix action.
        Must have the same number of subspaces on each ensemble rank as row_space.

    Returns
    -------
    PETSc.Mat :
        The PETSc.Mat with an :class:`.EnsembleBlockDiagonalMatCtx` Python context.

    See Also
    --------
    EnsembleBlockDiagonalMatCtx
    ~.ensemble_pc.EnsembleBJacobiPC
    """
    ctx = EnsembleBlockDiagonalMatCtx(block_mats, row_space, col_space)

    # number of columns is row length, and vice-versa
    ncols = ctx.col_space.layout_vec.getSizes()
    nrows = ctx.row_space.layout_vec.getSizes()

    mat = PETSc.Mat().createPython(
        (ncols, nrows), ctx,
        comm=ctx.ensemble.global_comm)
    mat.setUp()
    mat.assemble()
    return mat

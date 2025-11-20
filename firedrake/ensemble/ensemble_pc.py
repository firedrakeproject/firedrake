import petsctools
from firedrake.petsc import PETSc
from firedrake.ensemble.ensemble_function import EnsembleFunction
from firedrake.ensemble.ensemble_mat import EnsembleMatBase, EnsembleBlockDiagonalMat

__all__ = (
    "EnsembleBJacobiPC",
)


def obj_name(obj):
    return f"{type(obj).__module__}.{type(obj).__name__}"


class EnsemblePCBase(petsctools.PCBase):
    """
    Base class for python type PCs defined over an :class:`~.ensemble.Ensemble`.

    The pc operators must be python Mats with :class:`~.ensemble_mat.EnsembleMatBase`.

    Notes
    -----
    The main use of this base class is to enable users to implement the preconditioner
    action as acting on and resulting in an :class:`~.ensemble_function.EnsembleFunction`.
    This is done by implementing the ``apply_impl`` method.

    See Also
    --------
    ~.ensemble_mat.EnsembleMatBase
    """
    needs_python_pmat = True

    def initialize(self, pc):
        super().initialize(pc)

        if not isinstance(self.pmat, EnsembleMatBase):
            pcname = obj_name(self)
            pmatname = obj_name(self.pmat)
            raise TypeError(
                f"PC {pcname} needs an EnsembleMatBase pmat, but it is a {pmatname}")

        self.ensemble = self.pmat.ensemble

        self.row_space = self.pmat.row_space.dual()
        self.col_space = self.pmat.col_space.dual()

        self.x = EnsembleFunction(self.row_space)
        self.y = EnsembleFunction(self.col_space)

    def apply(self, pc, x, y):
        with self.x.vec_wo() as v:
            x.copy(result=v)

        self.apply_impl(pc, self.x, self.y)

        with self.y.vec_ro() as v:
            v.copy(result=y)

    def apply_impl(self, pc, x, y):
        raise NotImplementedError


class EnsembleBJacobiPC(EnsemblePCBase):
    """
    A python PC context for a block Jacobi method defined over an :class:`~.ensemble.Ensemble`.
    Each block acts on a single subspace of an :class:`~.ensemble_functionspace.EnsembleFunctionSpace`
    and is (approximately) solved with its own KSP, which defaults to -ksp_type preonly.

    Available options:

    * ``-pc_use_amat`` - use Amat to apply block of operator in inner Krylov method
    * ``-sub_%d`` - set options for the ``%d``'th block, numbered from ensemble rank 0.
    * ``-sub_`` - set default options for all blocks.

    Notes
    -----
    Currently this is only implemented for :class:`~.ensemble_mat.EnsembleBlockDiagonalMat` matrices.

    See Also
    --------
    ~.ensemble_mat.EnsembleBlockDiagonalMatrix
    ~.ensemble_mat.EnsembleBlockDiagonalMat
    """
    prefix = "ebjacobi_"

    def initialize(self, pc):
        super().initialize(pc)

        use_amat_prefix = self.parent_prefix + "pc_use_amat"
        self.use_amat = PETSc.Options().getBool(use_amat_prefix, False)

        if not isinstance(self.pmat, EnsembleBlockDiagonalMat):
            pcname = obj_name(self)
            matname = obj_name(self.pmat)
            raise TypeError(
                f"PC {pcname} needs an EnsembleBlockDiagonalMat pmat, but it is a {matname}")

        if self.use_amat:
            if not isinstance(self.amat, EnsembleBlockDiagonalMat):
                pcname = obj_name(self)
                matname = obj_name(self.amat)
                raise TypeError(
                    f"PC {pcname} needs an EnsembleBlockDiagonalMat amat, but it is a {matname}")

        # # default to behaving like a PC
        # default_options = {'ksp_type': 'preonly'}

        # default_sub_prefix = self.parent_prefix + "sub_"
        # default_sub_options = get_default_options(
        #     default_sub_prefix, range(self.col_space.nglobal_spaces))
        # default_options.update(default_sub_options)

        default_sub_prefix = self.parent_prefix + "sub_"

        default_options = petsctools.DefaultOptionSet(
            base_prefix=default_sub_prefix,
            custom_prefix_endings=range(self.col_space.nglobal_spaces))

        block_offset = self.col_space.global_spaces_offset

        sub_ksps = []
        for i in range(len(self.pmat.block_mats)):
            sub_ksp = PETSc.KSP().create(
                comm=self.ensemble.comm)

            if self.use_amat:
                sub_amat = self.amat.block_mats[i]
            else:
                sub_amat = self.pmat.block_mats[i]
            sub_pmat = self.pmat.block_mats[i]
            sub_ksp.setOperators(sub_amat, sub_pmat)

            sub_prefix = default_sub_prefix + str(block_offset + i)

            petsctools.attach_options(
                sub_ksp, parameters={},
                options_prefix=sub_prefix,
                default_options_set=default_options)

            # default to behaving like a PC
            petsctools.set_default_parameter(
                sub_ksp, "ksp_type", "preonly")

            petsctools.set_from_options(sub_ksp)

            sub_ksp.incrementTabLevel(1, parent=pc)
            sub_ksp.pc.incrementTabLevel(1, parent=pc)

            sub_ksps.append(sub_ksp)

        self.sub_ksps = tuple(sub_ksps)

    def apply_impl(self, pc, x, y):
        sub_vecs = zip(self.x.subfunctions, self.y.subfunctions)
        for sub_ksp, (subx, suby) in zip(self.sub_ksps, sub_vecs):
            with subx.dat.vec_ro as rhs, suby.dat.vec_wo as sol:
                with petsctools.inserted_options(sub_ksp):
                    sub_ksp.solve(rhs, sol)

    def update(self, pc):
        for sub_ksp in self.sub_ksps:
            sub_ksp.setUp()

    def view(self, pc, viewer=None):
        super().view(pc, viewer=viewer)
        viewer.printfASCII("  firedrake block Jacobi preconditioner for ensemble Mats\n")
        if self.use_amat:
            viewer.printfASCII("  using Amat local matrix\n")
        viewer.printfASCII(f"  Number of blocks = {self.col_space.nglobal_spaces}, Number of ensemble ranks = {self.ensemble.ensemble_size}\n")

        if viewer.getFormat() != PETSc.Viewer.Format.ASCII_INFO_DETAIL:
            viewer.printfASCII("  Local solver information for first block is in the following KSP and PC objects on rank 0:\n")
            prefix = self.parent_prefix
            viewer.printfASCII(f"  Use -{prefix}ksp_view ::ascii_info_detail to display information for all blocks\n")
            subviewer = viewer.getSubViewer(self.ensemble.comm)
            if self.ensemble.ensemble_rank == 0:
                subviewer.pushASCIITab()
                self.sub_ksps[0].view(subviewer)
                subviewer.popASCIITab()
            viewer.restoreSubViewer(subviewer)
            # Comment taken from PCView_BJacobi in https://petsc.org/release/src/ksp/pc/impls/bjacobi/bjacobi.c.html#PCBJACOBI
            # extra call needed because of the two calls to PetscViewerASCIIPushSynchronized() in PetscViewerGetSubViewer()
            viewer.popASCIISynchronized()

        else:
            viewer.pushASCIISynchronized()
            viewer.printfASCII("  Local solver information for each block is in the following KSP and PC objects:\n")
            viewer.pushASCIITab()
            subviewer = viewer.getSubViewer(self.ensemble.comm)
            r = self.ensemble.ensemble_rank
            offset = self.col_space.global_spaces_offset
            subviewer.printfASCII(f"[{r}] number of local blocks = {self.col_space.nlocal_spaces}, first local block number = {offset}\n")
            for i, subksp in enumerate(self.sub_ksps):
                subviewer.printfASCII(f"[{r}] local block number {i}, global block number {offset + i}\n")
                subksp.view(subviewer)
                subviewer.printfASCII("- - - - - - - - - - - - - - - - - -\n")
            viewer.restoreSubViewer(subviewer)
            viewer.popASCIITab()
            viewer.popASCIISynchronized()

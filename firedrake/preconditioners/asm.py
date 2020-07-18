import abc

from pyop2.mpi import COMM_SELF
from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space
import numpy

__all__ = ("ASMPatchPC", "ASMLinesmoothPC")


class ASMPatchPC(PCBase):
    ''' PC for PETSc PCASM

    should implement:
    - :meth:`get_patches`
    '''

    @property
    @abc.abstractmethod
    def _prefix(self):
        "Options prefix for the solver"

    def initialize(self, pc):
        # Get context from pc
        _, P = pc.getOperators()
        dm = pc.getDM()
        self.prefix = pc.getOptionsPrefix() + self._prefix

        # Extract function space and mesh to obtain plex and indexing functions
        V = get_function_space(dm)

        # Obtain patches from user defined funtion
        ises = self.get_patches(V)

        # Create new PC object as ASM type and set index sets for patches
        asmpc = PETSc.PC().create(comm=pc.comm)
        asmpc.incrementTabLevel(1, parent=pc)
        asmpc.setOptionsPrefix(self.prefix + "_sub")
        asmpc.setOperators(*pc.getOperators())
        asmpc.setType(asmpc.Type.ASM)
        asmpc.setASMLocalSubdomains(len(ises), ises)
        asmpc.setFromOptions()
        self.asmpc = asmpc

    @abc.abstractmethod
    def get_patches(self, V):
        ''' Get the patches used for PETSc PSASM

        :param  V: the :class:`~.FunctionSpace`.

        returns a list of index sets defining the ASM patches.
        '''
        pass

    def view(self, pc):
        self.ampc.view()

    def update(self, pc):
        self.asmpc.setUp()

    def apply(self, pc, x, y):
        self.asmpc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.asmpc.applyTranspose(x, y)


class ASMLinesmoothPC(ASMPatchPC):
    '''Linesmoother PC for extruded meshes implemented as an
    :class:`ASMPatchPC`.

    ASMLinesmoothPC is an additive Schwarz preconditioner where each
    patch consists of all dofs associated with a vertical column (and
    hence extruded meshes are necessary). Three types of columns are
    possible: columns of horizontal faces (each column built over a
    face of the base mesh), columns of vertical faces (each column
    built over an edge of the base mesh), and columns of vertical
    edges (each column built over a vertex of the base mesh).

    To select the column type or types for the patches, use
    'pc_linesmooth_codims' to set integers giving the codimension of
    the base mesh entities for the columns. For example,
    'pc_linesmooth_codims 0,1' creates patches for each cell and each
    facet of the base mesh.
    '''

    _prefix = "pc_linesmooth_"

    def get_patches(self, V):
        mesh = V._mesh
        assert mesh.cell_set._extruded
        dm = mesh._topology_dm
        section = V.dm.getDefaultSection()
        lgmap = V.dof_dset.lgmap

        # Obtain the codimensions to loop over from options, if present
        codim_list = PETSc.Options().getString(self.prefix+"codims", "0, 1")
        codim_list = [int(ii) for ii in codim_list.split(",")]

        # Build index sets for the patches
        ises = []
        for codim in codim_list:
            for p in range(*dm.getHeightStratum(codim)):
                # Only want to build patches over owned faces
                if dm.getLabelValue("pyop2_ghost", p) != -1:
                    continue
                dof = section.getDof(p)
                if dof <= 0:
                    continue
                off = section.getOffset(p)
                indices = numpy.arange(off*V.value_size, V.value_size * (off + dof), dtype='int32')
                # Map local indices into global indices and create the IS for PCASM
                global_indices = lgmap.apply(indices)
                iset = PETSc.IS().createGeneral(global_indices, comm=COMM_SELF)
                ises.append(iset)

        return ises

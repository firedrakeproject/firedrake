import abc
import warnings

from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space
from firedrake.mesh import DistributedMeshOverlapType
from firedrake.logging import warning
from firedrake.exceptions import NonUniqueMeshSequenceError
from tinyasm import _tinyasm as tinyasm
from mpi4py import MPI
import numpy
import firedrake.exceptions
from firedrake import utils


__all__ = ("ASMPatchPC", "ASMStarPC", "ASMVankaPC", "ASMLinesmoothPC", "ASMExtrudedStarPC")


class ASMPatchPC(PCBase):
    ''' PC for PETSc PCASM

    should implement:
    - :meth:`get_patches`
    '''

    @property
    @abc.abstractmethod
    def _prefix(self):
        "Options prefix for the solver (should end in an underscore)"

    def initialize(self, pc):
        # Get context from pc
        _, P = pc.getOperators()
        dm = pc.getDM()
        self.prefix = (pc.getOptionsPrefix() or "") + self._prefix
        opts = PETSc.Options(self.prefix)

        # Extract function space and mesh to obtain plex and indexing functions
        V = get_function_space(dm)

        # Obtain patches from user defined function
        ises = self.get_patches(V)
        # PCASM expects at least one patch, so we define an empty one on idle processes
        if len(ises) == 0:
            ises = [PETSc.IS().createGeneral(numpy.empty(0, dtype=utils.IntType), comm=PETSc.COMM_SELF)]

        # Create new PC object as ASM type and set index sets for patches
        asmpc = PETSc.PC().create(comm=pc.comm)
        asmpc.incrementTabLevel(1, parent=pc)
        asmpc.setOptionsPrefix(self.prefix + "sub_")
        asmpc.setOperators(*pc.getOperators())

        backend = opts.getString("backend", default="petscasm").lower()
        # Either use PETSc's ASM PC or use TinyASM (as simple ASM
        # implementation designed to be fast for small block sizes).
        if backend == "petscasm":
            asmpc.setType(asmpc.Type.ASM)
            # Set default solver parameters
            asmpc.setASMType(PETSc.PC.ASMType.BASIC)
            sub_opts = PETSc.Options(asmpc.getOptionsPrefix())
            if "sub_pc_type" not in sub_opts:
                sub_opts["sub_pc_type"] = "lu"
            if "sub_pc_factor_mat_ordering_type" not in sub_opts:
                # Preserve the natural ordering to avoid zero pivots in saddle-point problems
                sub_opts["sub_pc_factor_mat_ordering_type"] = "natural"

            # If an ordering type is provided, PCASM should not sort patch indices, otherwise it can.
            mat_type = P.getType()
            if not mat_type.endswith("sbaij"):
                sentinel = object()
                ordering = opts.getString("mat_ordering_type", default=sentinel)
                asmpc.setASMSortIndices(ordering is sentinel)

            lgmap = V._lgmap
            # Translate to global numbers
            ises = tuple(lgmap.applyIS(iset) for iset in ises)
            asmpc.setASMLocalSubdomains(len(ises), ises)
        elif backend == "tinyasm":
            _, P = asmpc.getOperators()
            lgmap = V._lgmap
            P.setLGMap(rmap=lgmap, cmap=lgmap)

            asmpc.setType("tinyasm")
            # TinyASM wants local numbers, no need to translate
            tinyasm.SetASMLocalSubdomains(
                asmpc, ises,
                [W.dm.getDefaultSF() for W in V],
                [W.block_size for W in V],
                sum(W.block_size * W.axes.local_size for W in V))
            asmpc.setUp()
        else:
            raise ValueError(f"Unknown backend type {backend}")

        asmpc.setFromOptions()
        self.asmpc = asmpc

        self._patch_statistics = []
        if opts.getBool("view_patch_sizes", default=False):
            # Compute and stash patch statistics
            mpi_comm = pc.comm.tompi4py()
            max_local_patch = max(is_.getSize() for is_ in ises)
            min_local_patch = min(is_.getSize() for is_ in ises)
            sum_local_patch = sum(is_.getSize() for is_ in ises)
            max_global_patch = mpi_comm.allreduce(max_local_patch, op=MPI.MAX)
            min_global_patch = mpi_comm.allreduce(min_local_patch, op=MPI.MIN)
            sum_global_patch = mpi_comm.allreduce(sum_local_patch, op=MPI.SUM)
            avg_global_patch = sum_global_patch / mpi_comm.allreduce(len(ises) if sum_local_patch > 0 else 0, op=MPI.SUM)
            msg = f"Minimum / average / maximum patch sizes : {min_global_patch} / {avg_global_patch} / {max_global_patch}\n"
            self._patch_statistics.append(msg)

    @abc.abstractmethod
    def get_patches(self, V, *, columns: bool):
        ''' Get the patches used for PETSc PCASM

        :param  V: the :class:`~.FunctionSpace`.

        :returns: a list of index sets defining the ASM patches in local
            numbering (before lgmap.apply has been called).
        '''
        pass

    def view(self, pc, viewer=None):
        self.asmpc.view(viewer=viewer)
        if viewer is not None:
            for msg in self._patch_statistics:
                viewer.printfASCII(msg)

    def update(self, pc):
        # This is required to update an inplace ILU factorization
        if self.asmpc.getType() == "asm":
            for sub in self.asmpc.getASMSubKSP():
                sub.getOperators()[0].setUnfactored()

    def apply(self, pc, x, y):
        self.asmpc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.asmpc.applyTranspose(x, y)

    def destroy(self, pc):
        if hasattr(self, "asmpc"):
            self.asmpc.destroy()


class ASMStarPC(ASMPatchPC):
    '''Patch-based PC using Star of mesh entities implemented as an
    :class:`ASMPatchPC`.

    ASMStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the topological star of the mesh entity
    specified by `pc_star_construct_dim`.

    Non-overlapping patches may be optionally grouped together via a
    coloring of the mesh entities. This is specified via the option
    `pc_star_use_coloring`.

    The mesh entities in the patches may be reordered by applying a
    matrix reordering to the connectivity graph with the option
    `pc_star_mat_ordering_type`.
    '''

    _prefix = "pc_star_"

    def get_patches(self, V):
        try:
            mesh = V.mesh().unique()
        except NonUniqueMeshSequenceError:
            raise NotImplementedError("Not implemented for general mixed meshes")

        # Obtain the topological entities to use to construct the stars
        opts = PETSc.Options(self.prefix)
        depth = opts.getInt("construct_dim", default=0)
        validate_overlap(mesh, depth, "star")

        use_coloring = opts.getBool("use_coloring", default=False)
        ordering = opts.getString("mat_ordering_type", default="natural")

        if _get_columns_option(opts, mesh):
            mesh_dm = mesh._base_mesh.topology_dm
            sections = [Vsub._base_mesh_section for Vsub in V]
        else:
            mesh_dm = mesh.topology_dm
            sections = [Vsub.local_section for Vsub in V]

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = get_local_ises_indices(V)

        # Build index sets for the patches
        colors = get_colors(mesh_dm, use_coloring, depth, distance=1)
        ises = [build_star_indices(sections, V_local_ises_indices, mesh_dm, ordering, self.prefix, color)
                for color in colors]
        return ises


class ASMVankaPC(ASMPatchPC):
    '''Patch-based PC using closure of star of mesh entities implmented as an
    :class:`ASMPatchPC`.

    ASMVankaPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the closure of the star of the mesh entity
    specified by `pc_vanka_construct_dim` (or codim).

    Non-overlapping patches may be optionally grouped together via a
    coloring of the mesh entities. This is specified via the option
    `pc_vanka_use_coloring`.

    The mesh entities in the patches may be reordered by applying a
    matrix reordering to the connectivity graph with the option
    `pc_vanka_mat_ordering_type`.
   '''

    _prefix = "pc_vanka_"

    def get_patches(self, V):
        try:
            mesh = V.mesh().unique()
        except NonUniqueMeshSequenceError:
            raise NotImplementedError("Not implemented for general mixed meshes")

        if _get_columns_option(opts, mesh):
            mesh_dm = mesh._base_mesh.topology_dm
            sections = [Vsub._base_mesh_section for Vsub in V]
        else:
            mesh_dm = mesh.topology_dm
            sections = [Vsub.local_section for Vsub in V]

        # Obtain the topological entities to use to construct the stars
        opts = PETSc.Options(self.prefix)
        depth = opts.getInt("construct_dim", default=-1)
        height = opts.getInt("construct_codim", default=-1)
        if (depth == -1 and height == -1) or (depth != -1 and height != -1):
            raise ValueError(f"Must set exactly one of {self.prefix}construct_dim or {self.prefix}construct_codim")
        if depth == -1:
            depth = mesh_dm.getDimension() - height
        validate_overlap(mesh, depth, "vanka")

        exclude_subspaces = opts.getIntArray("exclude_subspaces", default=[])
        include_subspaces = [i for i in range(len(V)) if i not in exclude_subspaces]
        include_type = opts.getString("include_type", default="star").lower()
        if include_type not in ["star", "entity"]:
            raise ValueError(f"{self.prefix}include_type must be either 'star' or 'entity', not {include_type}")
        include_star = include_type == "star"

        use_coloring = opts.getBool("use_coloring", default=False)
        ordering = opts.getString("mat_ordering_type", default="natural")

        def splitting(V):
            return (tuple(V[i] for i in include_subspaces), tuple(V[i] for i in exclude_subspaces))

        Z = splitting(V)

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = get_local_ises_indices(V)
        Z_local_ises_indices = splitting(V_local_ises_indices)

        # Build index sets for the patches
        colors = get_colors(mesh_dm, use_coloring, depth, distance=3)
        ises = [build_vanka_indices(sections, Z_local_ises_indices, mesh_dm, ordering, self.prefix,
                                    include_star, color) for color in colors]
        return ises


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
        try:
            mesh = V.mesh().unique()
        except NonUniqueMeshSequenceError:
            raise NotImplementedError("Not implemented for general mixed meshes")
        assert mesh.extruded

        base_dm = mesh._base_mesh.topology_dm
        section = V._base_mesh_section
        # Obtain the codimensions to loop over from options, if present
        opts = PETSc.Options(self.prefix)
        codim_list = list(map(int, opts.getString("codims", "0, 1").split(",")))

        if len(V) > 1:
            raise NotImplementedError("Not implemeneted for mixed spaces")
        V_local_ises_indices = get_local_ises_indices(V)

        # Build index sets for the patches
        ises = []
        for codim in codim_list:
            for base_p in range(*base_dm.getHeightStratum(codim)):
                # Only want to build patches over owned faces
                if base_dm.getLabelValue("firedrake_is_ghost", base_p) != -1:
                    continue

                dof = section.getDof(base_p)
                if dof <= 0:
                    continue
                off = section.getOffset(base_p)
                zlice = slice(off, off+dof, dtype=utils.IntType)

                indices = V_local_ises_indices[0][zlice]
                indices = indices[indices >= 0]
                iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
                ises.append(iset)

        return ises


def order_points(mesh_dm, points, ordering_type, prefix):
    '''Order the points (topological entities) of a patch based
    on the adjacency graph of the mesh.

    :arg mesh_dm: the `mesh.topology_dm`
    :arg points: array with point indices forming the patch
    :arg ordering_type: a `PETSc.Mat.OrderingType`
    :arg prefix: the prefix associated with additional ordering options

    :returns: the permuted array of points
    '''
    # Order points by decreasing topological dimension (interiors, faces, edges, vertices)
    points = points[::-1]
    if ordering_type == "natural":
        return points
    subgraph = [numpy.intersect1d(points, mesh_dm.getAdjacency(p), return_indices=True)[1] for p in points]
    ia = numpy.cumsum([0] + [len(neigh) for neigh in subgraph]).astype(PETSc.IntType)
    ja = numpy.concatenate(subgraph).astype(PETSc.IntType)
    A = PETSc.Mat().createAIJ((len(points), )*2, csr=(ia, ja, numpy.ones(ja.shape, PETSc.RealType)), comm=PETSc.COMM_SELF)
    A.setOptionsPrefix(prefix)
    rperm, cperm = A.getOrdering(ordering_type)
    indices = points[rperm.getIndices()]
    A.destroy()
    rperm.destroy()
    cperm.destroy()
    return indices


def get_basemesh_nodes(W):
    raise NotImplementedError
    pstart, pend = W.mesh().topology_dm.getChart()
    section = W.dm.getDefaultSection()
    # location of first dof on an entity
    basemeshoff = numpy.empty(pend - pstart, dtype=utils.IntType)
    # number of dofs on this entity
    basemeshdof = numpy.empty(pend - pstart, dtype=utils.IntType)
    # number of dofs stacked on this entity in each cell
    basemeshlayeroffset = numpy.empty(pend - pstart, dtype=utils.IntType)

    # For every base mesh entity, what's the layer offset?
    layer_offsets = numpy.full(W.nodes.local_size, -1, dtype=utils.IntType)
    layer_offsets[W.cell_node_map().values_with_halo] = W.cell_node_map().offset
    nlayers = W.mesh().layers

    for p in range(pstart, pend):
        dof = section.getDof(p)
        off = section.getOffset(p)
        if dof == 0:
            dof_per_layer = 0
            layer_offset = 0
        else:
            layer_offset = layer_offsets[off]
            assert layer_offset >= 0
            dof_per_layer = dof - (nlayers - 1) * layer_offset

        basemeshoff[p - pstart] = off
        basemeshdof[p - pstart] = dof_per_layer
        basemeshlayeroffset[p - pstart] = layer_offset

    if W.mesh().extruded_periodic:
        # Account for missing dofs from the top layer
        for dim in range(W.mesh().topological_dimension):
            qstart, qend = W.mesh().topology_dm.getDepthStratum(dim)
            quotient = len(W.finat_element.entity_dofs()[(dim, 0)][0])
            basemeshdof[qstart-pstart:qend-pstart] += quotient

    return basemeshoff, basemeshdof, basemeshlayeroffset


class ASMExtrudedStarPC(ASMStarPC):
    '''Patch-based PC using Star of mesh entities implmented as an
    :class:`ASMPatchPC`.

    This class is deprecated. You should use ASMStarPC passing the option
    column = 0 instead.

    ASMExtrudedStarPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the topological star of the mesh entity
    specified by `pc_star_construct_dim`.

    Non-overlapping patches may be optionally grouped together via a
    coloring of the mesh entities. This is specified via the option
    `pc_star_use_coloring`.

    The mesh entities in the patches may be reordered by applying a
    matrix reordering to the connectivity graph with the option
    `pc_star_mat_ordering_type`.
    '''

    def __init__(self, *args, **kwargs):
        # make sure not passing columns...
        warnings.warn(
            "ASMExtrudedStarPC is deprecated. Please use ASMStarPC instead. "
            "You will have to specify column=0.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)


def get_local_ises_indices(V):
    """Return the local indices of each subspace of V.
    The restricted DOFs will be masked for a RestrictedFunctionSpace.
    """
    V_local_ises_indices = tuple(iset.indices for iset in V.local_ises)
    for Vi, indices in zip(V, V_local_ises_indices):
        if Vi.boundary_set:
            indices[Vi.lgmap().indices < 0] = -1
    return V_local_ises_indices


def validate_overlap(mesh, patch_dim, patch_type):
    if patch_type == "python":
        return
    patch_depth = {"pardecomp": 0, "star": 1, "vanka": 2}[patch_type]

    tdim = mesh.topology_dm.getDimension()
    overlap_entity, overlap_depth = mesh._distribution_parameters["overlap_type"]
    overlap_dim = {
        DistributedMeshOverlapType.VERTEX: 0,
        DistributedMeshOverlapType.FACET: tdim-1,
        DistributedMeshOverlapType.NONE: tdim,
    }[overlap_entity]

    if mesh.comm.size > 1:
        if overlap_dim > patch_dim:
            patch_entity = {0: "vertex", 1: "edge", 2: "face", tdim: "cell"}[patch_dim]
            warning(f"{overlap_entity} does not support {patch_entity}-patches. "
                    "Did you forget to set overlap_type in your mesh's distribution_parameters?")
        if overlap_depth < patch_depth:
            warning(f"Mesh overlap depth of {overlap_depth} does not support {patch_type}-patches. "
                    "Did you forget to set overlap_type in your mesh's distribution_parameters?")


# TODO: This function only exists to be a central place to catch deprecated behaviour.
# We should be able to remove it when the deprecation cycle is complete.
def _get_columns_option(opts, mesh):
    # NOTE: What if we extrude multiple times? How can we specify different column types?
    # Or is that just a bad idea?

    if opts.hasName("column"):
        columns = opts.getBool("column")
        if columns and not mesh.extruded:
            raise ValueError("Can only pass 'columns' on an extruded mesh")
    else:
        if mesh.extruded:
            warnings.warn(
                f"""\
**IMPORTANT**

You are using {type(self).__name__} on an extruded mesh without specifying
the 'columns' option. The current behaviour is for the patch to be over the
base mesh and covering the full column. THIS IS GOING TO CHANGE AS THE DEFAULT
BEHAVIOUR. In future releases of Firedrake the patches will by default only
cover the DoFs immediately surrounding the vertex.

To continue to keep this behaviour you have to pass the option 'PREFIX + column = 1'.""",
                FutureWarning,
            )
            columns = True
        else:
            columns = False
    return columns


def get_colors(mesh_dm, use_coloring, depth, distance=1):
    """Returns a coloring of the mesh entities.

    :arg mesh_dm: the DMPlex
    :arg use_coloring: if True computes the coloring,
        otherwise each entity gets its own color
    :arg depth: the entity dimension
    :arg distance: the coloring distance

    :returns: an iterable of PETSc.IS or int defining each color
    """
    if use_coloring:
        opts_modified = False
        opts = PETSc.Options()
        if "mat_coloring_type" not in opts:
            opts_modified = True
            coloring_type = "power" if distance > 2 else "greedy"
            opts.setValue("mat_coloring_type", coloring_type)
        colors = mesh_dm.createColoring(depth=depth, distance=distance)
        if opts_modified:
            opts.delValue("mat_coloring_type")
    else:
        colors = range(*mesh_dm.getDepthStratum(depth))
    return colors


def get_entity_dofs(sections, V_local_ises_indices, points):
    """Return degrees of freedom associated with mesh entities (points of the DMPlex).

    :arg V: the FunctionSpace to extract DOFs from
    :arg V_local_ises_indices: V.local_ises.indices
    :points: an iterable of mesh entities

    :returns: a list with the DOFs of V associated with the mesh entities
    """
    indices = []
    for i, section in enumerate(sections):
        for p in points:
            dof = section.getDof(p)
            if dof <= 0:
                continue
            off = section.getOffset(p)
            # Local indices within W
            indices.extend(V_local_ises_indices[i][off:off+dof])
    return indices


def get_star_points(mesh_dm, ordering, prefix, seed_points):
    """Get DMPlex points in the star of each point in seed_points.

    :arg mesh_dm: the DMPlex
    :arg ordering: a Mat.OrderingType indicating the ordering type
    :arg prefix: the PETSc.Options prefix to further specify the ordering
    :seed_points: an iterable point indices to compute the star for

    :returns: A list of the points in the star
    """
    if isinstance(seed_points, PETSc.IS):
        seed_points = seed_points.indices
    elif numpy.isscalar(seed_points):
        seed_points = (seed_points,)
    points = []
    for seed in seed_points:
        # Only build patches over owned DoFs
        if mesh_dm.getLabelValue("firedrake_is_ghost", seed) != -1:
            continue
        # Create point list from mesh DM
        star, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
        star = order_points(mesh_dm, star, ordering, prefix)
        points.extend(star)
    return points


def build_star_indices(sections, V_local_ises_indices, mesh_dm, ordering, prefix, seed_points):
    """Return DOFs in the star of each point in seed_points.

    :arg V: the FunctionSpace to extract DOFs from
    :arg V_local_ises_indices: V.local_ises.indices
    :arg mesh_dm: the DMPlex
    :arg ordering: a Mat.OrderingType indicating the ordering type
    :arg prefix: the PETSc.Options prefix to further specify the ordering
    :seed_points: an iterable of point indices to construct the star patches

    :returns: A PETSc.IS with the degrees of freedom in the star patches
    """
    points = get_star_points(mesh_dm, ordering, prefix, seed_points)
    indices = get_entity_dofs(sections, V_local_ises_indices, points)
    indices = numpy.array(indices, dtype=PETSc.IntType)
    indices = indices[indices >= 0]
    iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
    return iset


def build_vanka_indices(Z_sections, Z_local_ises_indices, mesh_dm, ordering, prefix, include_star, seed_points):
    """Return DOFs in the Vanka patches constructed at each point in seed_points.

    :arg Z: a tuple of the included/excluded FunctionSpaces to extract DOFs from
    :arg Z_local_ises_indices: (Z[0].local_ises.indices, Z[1].local_ises.indices)
    :arg mesh_dm: the DMPlex
    :arg ordering: a Mat.OrderingType indicating the ordering type
    :arg prefix: the PETSc.Options prefix to further specify the ordering
    :arg include_star: whether to include DOFs of Z[1] in the star or just the entity
    :seed_points: an iterable of point indices to construct the Vanka patches

    :returns: A PETSc.IS with the degrees of freedom in the Vanka patches
    """
    if isinstance(seed_points, PETSc.IS):
        seed_points = seed_points.indices
    elif numpy.isscalar(seed_points):
        seed_points = (seed_points,)
    indices = []
    for seed in seed_points:
        V_points = []
        Q_points = []
        # Only build patches over owned DoFs
        if mesh_dm.getLabelValue("firedrake_is_ghost", seed) != -1:
            continue
        # Create point list from mesh DM
        star, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
        star = order_points(mesh_dm, star, ordering, prefix)
        if include_star:
            Q_points.extend(star)
        else:
            Q_points.append(seed)
        closure = []
        for s in reversed(star):
            cs, _ = mesh_dm.getTransitiveClosure(s, useCone=True)
            closure.extend(cs)
        # Grab unique points with stable ordering
        closure = reversed(dict.fromkeys(closure))
        V_points.extend(closure)
        indices.extend(get_entity_dofs([Z_sections[0]], Z_local_ises_indices[0], V_points))
        indices.extend(get_entity_dofs([Z_sections[1]], Z_local_ises_indices[1], Q_points))

    indices = numpy.array(indices, dtype=PETSc.IntType)
    indices = indices[indices >= 0]
    iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
    return iset

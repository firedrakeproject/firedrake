import abc

from pyop2.datatypes import IntType
from firedrake.cython import patchimpl
from firedrake.preconditioners.base import PCBase
from firedrake.petsc import PETSc
from firedrake.adapt import mark_refined_entities
from firedrake.dmhooks import get_function_space
from firedrake.mesh import DistributedMeshOverlapType, MeshGeometry
from firedrake.logging import warning
from firedrake.exceptions import NonUniqueMeshSequenceError
from tinyasm import _tinyasm as tinyasm
from mpi4py import MPI
import numpy


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

        # Extract function space and mesh to obtain plex and indexing functions
        V = get_function_space(dm)

        # Obtain patches from user defined function
        ises = self.get_patches(V)
        # PCASM expects at least one patch, so we define an empty one on idle processes
        if len(ises) == 0:
            ises = [PETSc.IS().createGeneral(numpy.empty(0, dtype=IntType), comm=PETSc.COMM_SELF)]

        # Create new PC object as ASM type and set index sets for patches
        asmpc = PETSc.PC().create(comm=pc.comm)
        asmpc.incrementTabLevel(1, parent=pc)
        asmpc.setOptionsPrefix(self.prefix + "sub_")
        asmpc.setOperators(*pc.getOperators())

        opts = PETSc.Options(self.prefix)
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

            lgmap = V.dof_dset.lgmap
            # Translate to global numbers
            ises = tuple(lgmap.applyIS(iset) for iset in ises)
            asmpc.setASMLocalSubdomains(len(ises), ises)
        elif backend == "tinyasm":
            _, P = asmpc.getOperators()
            lgmap = V.dof_dset.lgmap
            P.setLGMap(rmap=lgmap, cmap=lgmap)

            asmpc.setType("tinyasm")
            # TinyASM wants local numbers, no need to translate
            tinyasm.SetASMLocalSubdomains(
                asmpc, ises,
                [W.dm.getDefaultSF() for W in V],
                [W.block_size for W in V],
                sum(W.block_size * W.dof_dset.total_size for W in V))
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
    def get_patches(self, V):
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

    The mesh entities that get a patch may be restricted to those marked
    by a `PETSc.DMLabel` on the mesh's DMPlex, named by the option
    `pc_star_construct_label` and holding them in the stratum
    `pc_star_construct_label_value`. The option `pc_star_adaptive` builds
    that label with `~firedrake.adapt.mark_refined_entities`, so that a
    smoother on an adaptively refined level only relaxes the entities whose
    star meets the refined region. The coloring, if requested, then colors
    only the marked entities.

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
        mesh_dm = mesh.topology_dm
        if mesh.cell_set._extruded:
            warning("applying ASMStarPC on an extruded mesh")

        # Obtain the topological entities to use to construct the stars
        opts = PETSc.Options(self.prefix)
        depth = opts.getInt("construct_dim", default=0)
        validate_overlap(mesh, depth, "star")

        use_coloring = opts.getBool("use_coloring", default=False)
        ordering = opts.getString("mat_ordering_type", default="natural")
        label, value = get_construct_label(mesh, self.prefix)

        # Take the star of every seed point, one patch at a time
        seeds, seed_offsets = get_seeds(mesh_dm, use_coloring, depth, 1, label, value)
        points, star_offsets, patch_offsets, _ = patchimpl.create_star_points(mesh_dm, seeds, seed_offsets)
        if ordering != "natural":
            # Each star is reordered on its own, since it is a patch of its own
            # unless a coloring grouped it with others
            for s in range(len(star_offsets) - 1):
                star = slice(star_offsets[s], star_offsets[s+1])
                points[star] = order_points(mesh_dm, points[star], ordering, self.prefix)

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = get_local_ises_indices(V)

        # Build index sets for the patches, every subspace reading the same points
        return patchimpl.create_patch_ises([points] * len(V),
                                           [star_offsets[patch_offsets]] * len(V),
                                           [W.dm.getLocalSection() for W in V],
                                           [W.block_size for W in V],
                                           V_local_ises_indices)


class ASMVankaPC(ASMPatchPC):
    '''Patch-based PC using closure of star of mesh entities implmented as an
    :class:`ASMPatchPC`.

    ASMVankaPC is an additive Schwarz preconditioner where each patch
    consists of all DoFs on the closure of the star of the mesh entity
    specified by `pc_vanka_construct_dim` (or codim).

    Non-overlapping patches may be optionally grouped together via a
    coloring of the mesh entities. This is specified via the option
    `pc_vanka_use_coloring`.

    The mesh entities that get a patch may be restricted to those marked
    by a `PETSc.DMLabel` on the mesh's DMPlex, named by the option
    `pc_vanka_construct_label` and holding them in the stratum
    `pc_vanka_construct_label_value`. The option `pc_vanka_adaptive` builds
    that label with `~firedrake.adapt.mark_refined_entities`, so that a
    smoother on an adaptively refined level only relaxes the entities whose
    patch meets the refined region. The coloring, if requested, then colors
    only the marked entities.

    The subspaces listed in `pc_vanka_exclude_subspaces` contribute only the
    DoFs on the star of the mesh entity, or on the entity alone when
    `pc_vanka_include_type` is `entity` rather than the default `star`.

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
        mesh_dm = mesh.topology_dm
        if mesh.layers:
            warning("applying ASMVankaPC on an extruded mesh")

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
        include_type = opts.getString("include_type", default="star").lower()
        if include_type not in ["star", "entity"]:
            raise ValueError(f"{self.prefix}include_type must be either 'star' or 'entity', not {include_type}")

        use_coloring = opts.getBool("use_coloring", default=False)
        ordering = opts.getString("mat_ordering_type", default="natural")
        label, value = get_construct_label(mesh, self.prefix)

        # Take the closure of the star of every seed point, one patch at a time
        seeds, seed_offsets = get_seeds(mesh_dm, use_coloring, depth, 3, label, value)
        star, star_offsets, patch_offsets, seeds = patchimpl.create_star_points(mesh_dm, seeds, seed_offsets)
        if ordering != "natural":
            # Each star is reordered on its own, since it is a patch of its own
            # unless a coloring grouped it with others, and the closure follows it
            for s in range(len(star_offsets) - 1):
                points = slice(star_offsets[s], star_offsets[s+1])
                star[points] = order_points(mesh_dm, star[points], ordering, self.prefix)
        closure, closure_offsets = patchimpl.create_closure_points(mesh_dm, star, star_offsets)

        # The included subspaces span the closure of the star, the excluded ones
        # only the star, or the entity it was built around
        included = (closure, closure_offsets[patch_offsets])
        excluded = (star, star_offsets[patch_offsets]) if include_type == "star" else (seeds, patch_offsets)
        groups = [excluded if i in exclude_subspaces else included for i in range(len(V))]

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_local_ises_indices = get_local_ises_indices(V)

        # Build index sets for the patches
        return patchimpl.create_patch_ises([points for points, _ in groups],
                                           [offsets for _, offsets in groups],
                                           [W.dm.getLocalSection() for W in V],
                                           [W.block_size for W in V],
                                           V_local_ises_indices)


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
        assert mesh.cell_set._extruded
        dm = mesh.topology_dm
        section = V.dm.getDefaultSection()
        # Obtain the codimensions to loop over from options, if present
        opts = PETSc.Options(self.prefix)
        codim_list = list(map(int, opts.getString("codims", "0, 1").split(",")))

        if len(V) > 1:
            raise NotImplementedError("Not implemeneted for mixed spaces")
        V_local_ises_indices = get_local_ises_indices(V)

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
                zlice = slice(off*V.block_size, V.block_size * (off + dof))

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
    pstart, pend = W.mesh().topology_dm.getChart()
    section = W.dm.getDefaultSection()
    # location of first dof on an entity
    basemeshoff = numpy.empty(pend - pstart, dtype=IntType)
    # number of dofs on this entity
    basemeshdof = numpy.empty(pend - pstart, dtype=IntType)
    # number of dofs stacked on this entity in each cell
    basemeshlayeroffset = numpy.empty(pend - pstart, dtype=IntType)

    # For every base mesh entity, what's the layer offset?
    layer_offsets = numpy.full(W.node_set.total_size, -1, dtype=IntType)
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

    _prefix = 'pc_star_'

    def get_patches(self, V):
        try:
            mesh = V.mesh().unique()
        except NonUniqueMeshSequenceError:
            raise NotImplementedError("Not implemented for general mixed meshes")
        mesh_dm = mesh.topology_dm
        nlayers = mesh.layers
        if not mesh.cell_set._extruded:
            return super(ASMExtrudedStarPC, self).get_patches(V)
        periodic = mesh.extruded_periodic

        # Obtain the topological entities to use to construct the stars
        opts = PETSc.Options(self.prefix)
        depth = opts.getInt("construct_dim", default=0)
        ordering = opts.getString("mat_ordering_type", default="natural")
        use_coloring = opts.getBool("use_coloring", default=False)

        # Accessing .indices causes the allocation of a global array,
        # so we need to cache these for efficiency
        V_ises = get_local_ises_indices(V)
        basemeshoff = []
        basemeshdof = []
        basemeshlayeroffsets = []
        for (i, W) in enumerate(V):
            boff, bdof, blayer_offsets = get_basemesh_nodes(W)
            basemeshoff.append(boff)
            basemeshdof.append(bdof)
            basemeshlayeroffsets.append(blayer_offsets)

        # Build index sets for the patches
        ises = []
        # Build a base_depth-star on the base mesh and extrude it by an
        # interval_depth-star on the interval mesh such that the depths sum to depth
        # and 0 <= interval_depth <= 1.
        #
        # Vertex-stars: depth = 0 = 0 + 0.
        # 0 + 0 -> vertex-star = (2D vertex-star) x (1D vertex-star)
        #
        # Edge-stars: depth = 1 = 1 + 0 = 0 + 1.
        # 1 + 0 -> horizontal edge-star = (2D edge-star) x (1D vertex-star)
        # 0 + 1 -> vertical edge-star = (2D vertex-star) x (1D interior)
        #
        # Face-stars: depth = 2 = 2 + 0 = 1 + 1.
        # 2 + 0 -> horizontal face-star = (2D interior) x (1D vertex-star)
        # 1 + 1 -> vertical face-star = (2D edge-star) x (1D interior)
        pstart, _ = mesh_dm.getChart()
        for base_depth in range(depth+1):
            interval_depth = depth - base_depth
            if interval_depth == 0:
                # extrude by 1D vertex-star
                layer_entities = [(1, 1), (1, 0), (0, 0)]
            elif interval_depth == 1:
                # extrude by 1D interior
                layer_entities = [(1, 0)]
            else:
                continue

            validate_overlap(mesh, base_depth, "star")

            num_layer_seeds = nlayers-1 if (periodic or interval_depth > 0) else nlayers
            # In the extruded direction we only need two colors (even/odd coloring)
            num_layer_colors = 2 if use_coloring else num_layer_seeds

            # Loop through the coloring of the base mesh
            colors = get_colors(mesh_dm, use_coloring, base_depth, distance=1)
            for color in colors:
                points = get_star_points(mesh_dm, ordering, self.prefix, color)
                if len(points) == 0:
                    continue
                points = numpy.asarray(points)
                points -= pstart  # offset by chart start

                # Loop through the coloring of the extruded direction
                for layer_color in range(num_layer_colors):
                    indices = []
                    # offset by the layer color
                    # loop until you reach the last point
                    # stride by the number of colors
                    for layer_seed in range(layer_color, num_layer_seeds, num_layer_colors):
                        # Get DoF indices for patch
                        for i, W in enumerate(V):
                            iset = V_ises[i]
                            for layer_dim, layer_shift in layer_entities:
                                layer = layer_seed - layer_shift
                                if periodic:
                                    # Handle periodic case
                                    layer = layer % (nlayers-1)
                                elif layer < 0 or (layer + layer_dim) >= nlayers:
                                    # We are out of bounds
                                    continue

                                for p in points:
                                    # How to walk up one layer
                                    blayer_offset = basemeshlayeroffsets[i][p]
                                    if blayer_offset <= 0:
                                        # In this case we don't have any dofs on
                                        # this entity.
                                        continue
                                    # Offset in the global array for the bottom of
                                    # the column
                                    off = basemeshoff[i][p]
                                    # Number of dofs in the interior of the
                                    # vertical interval cell on top of this base
                                    # entity
                                    dof = basemeshdof[i][p]
                                    # Hard-code taking the star
                                    if layer_dim == 0:
                                        begin = off + layer * blayer_offset
                                        end = off + layer * blayer_offset + dof
                                    else:
                                        begin = off + layer * blayer_offset + dof
                                        end = off + (layer + 1) * blayer_offset
                                    zlice = slice(W.block_size * begin, W.block_size * end)
                                    indices.extend(iset[zlice])

                    indices = numpy.array(indices, dtype=PETSc.IntType)
                    indices = indices[indices >= 0]
                    iset = PETSc.IS().createGeneral(indices, comm=PETSc.COMM_SELF)
                    ises.append(iset)
        return ises


def get_local_ises_indices(V):
    """Return the local indices of each subspace of V.
    The restricted DOFs will be masked for a RestrictedFunctionSpace.
    """
    V_local_ises_indices = tuple(iset.indices for iset in V.dof_dset.local_ises)
    for Vi, indices in zip(V, V_local_ises_indices):
        if Vi.boundary_set:
            indices[Vi.dof_dset.lgmap.indices < 0] = -1
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


def get_construct_label(mesh: MeshGeometry, prefix: str) -> tuple[PETSc.DMLabel | None, int]:
    """Return the label restricting the mesh entities that get a patch.

    Parameters
    ----------
    mesh
        The mesh the patches are built on.
    prefix
        The `PETSc.Options` prefix of the preconditioner.

    Returns
    -------
    tuple of PETSc.DMLabel or None, and int
        The label marking the entities and the stratum value holding them, or
        ``(None, 0)`` if the options select no label, in which case every
        entity of the stratum gets a patch.

    Raises
    ------
    ValueError
        If the options name a label the mesh does not have.

    """
    opts = PETSc.Options(prefix)
    if opts.getBool("adaptive", default=False):
        return mark_refined_entities(mesh), 1
    name = opts.getString("construct_label", default="")
    if not name:
        return None, 0
    mesh_dm = mesh.topology_dm
    if not mesh_dm.hasLabel(name):
        raise ValueError(f"{prefix}construct_label names the label {name!r}, "
                         "which the mesh does not have")
    return mesh_dm.getLabel(name), opts.getInt("construct_label_value", default=1)


def get_colors(mesh_dm: PETSc.DMPlex, use_coloring: bool, depth: int, distance: int = 1,
               label: PETSc.DMLabel | None = None, value: int = 0):
    """Return a coloring of the mesh entities.

    Parameters
    ----------
    mesh_dm
        The mesh topology.
    use_coloring
        If `True` compute the coloring, otherwise give each entity its own color.
    depth
        The topological dimension of the entities.
    distance
        How far through the mesh an entity reaches, 1 to separate star patches
        and 3 to separate Vanka patches.
    label
        A label selecting the entities, or `None` for all of them.
    value
        The stratum of ``label`` holding the entities.

    Returns
    -------
    iterable of PETSc.IS, or iterable of int
        The entities of each color.

    """
    if use_coloring:
        # Colors the subgraph the selected entities induce, so restricting
        # them can only merge colors, never split one
        colors = mesh_dm.createColoringLabel(depth=depth, distance=distance,
                                             label=label, value=value)
    elif label is None:
        colors = range(*mesh_dm.getDepthStratum(depth))
    else:
        # A completed label spans every stratum, so keep only the entities we seed with
        pstart, pend = mesh_dm.getDepthStratum(depth)
        points = numpy.empty(0, dtype=IntType)
        if label.getStratumSize(value) > 0:
            points = label.getStratumIS(value).indices
        colors = points[(points >= pstart) & (points < pend)]
    return colors


def get_seeds(mesh_dm: PETSc.DMPlex, use_coloring: bool, depth: int, distance: int = 1,
              label: PETSc.DMLabel | None = None,
              value: int = 0) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Return the mesh entities that each patch is built around.

    Parameters
    ----------
    mesh_dm
        The mesh topology.
    use_coloring
        If `True` group the entities by a coloring, otherwise give each entity
        its own patch.
    depth
        The topological dimension of the entities.
    distance
        How far through the mesh an entity reaches, 1 to separate star patches
        and 3 to separate Vanka patches.
    label
        A label selecting the entities, or `None` for all of them.
    value
        The stratum of ``label`` holding the entities.

    Returns
    -------
    tuple of numpy.ndarray
        The entities patch by patch, and the offsets saying where each patch
        starts in them.

    """
    colors = get_colors(mesh_dm, use_coloring, depth, distance, label, value)
    if isinstance(colors, range):
        seeds = numpy.arange(colors.start, colors.stop, dtype=IntType)
    elif use_coloring:
        indices = [numpy.asarray(color.indices, dtype=IntType) for color in colors]
        offsets = numpy.cumsum([0, *(len(i) for i in indices)], dtype=IntType)
        if len(indices) == 0:
            return numpy.empty(0, dtype=IntType), offsets
        return numpy.concatenate(indices), offsets
    else:
        seeds = numpy.asarray(colors, dtype=IntType)
    return seeds, numpy.arange(len(seeds) + 1, dtype=IntType)


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
        if mesh_dm.getLabelValue("pyop2_ghost", seed) != -1:
            continue
        # Create point list from mesh DM, by decreasing topological dimension
        # (interiors, faces, edges, vertices)
        star, _ = mesh_dm.getTransitiveClosure(seed, useCone=False)
        star = order_points(mesh_dm, star[::-1], ordering, prefix)
        points.extend(star)
    return points

from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.facet_split import get_restriction_indices
from firedrake.petsc import PETSc, DEFAULT_PARTITIONER
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.preconditioners.fdm import broken_function, tabulate_exterior_derivative
from firedrake.preconditioners.hiptmair import curl_to_grad
from firedrake.subspace import subspace, complete_strata, parent_local_to_global_map
from firedrake.utils import IntType
from firedrake.cython import dmcommon
from functools import cached_property

from firedrake.parloops import par_loop, INC, READ
from firedrake.bcs import DirichletBC
from pyop2.mpi import COMM_SELF
from ufl import H1, H2, HCurl, curl, div, dx, inner, replace
from finat.ufl import TensorElement, VectorElement
from pyop2.utils import as_tuple
import numpy

__all__ = ("BDDCPC",)


class BDDCPC(PCBase):
    """PC for PETSc PCBDDC (Balancing Domain Decomposition by Constraints).
    This is a domain decomposition method using subdomains defined by the
    blocks in a Mat of type IS.

    Internally, this PC creates a PETSc PCBDDC object that can be controlled by
    the options:
    - ``'bddc_cellwise'`` to set up a MatIS on cellwise subdomains if P.type == python,
    - ``'bddc_subdomain_size'`` to divide the cells of each process into subdomains
    of the given size in cells, if P.type == python,
    - ``'bddc_pc_bddc_neumann'`` to set sub-KSPs on subdomains excluding corners,
    - ``'bddc_pc_bddc_dirichlet'`` to set sub-KSPs on subdomain interiors,
    - ``'bddc_pc_bddc_coarse'`` to set the coarse solver KSP.

    This PC also inspects optional callbacks supplied in the application context:
    - ``'get_discrete_gradient'`` for 3D problems in H(curl), this is a callable that
    provide the arguments (a Mat tabulating the gradient of the auxiliary H1 space) and
    keyword arguments supplied to ``PETSc.PC.setBDDCDiscreteGradient``.
    - ``'get_divergence_mat'`` for problems in H(div) (resp. 2D H(curl)), this is
    provide the arguments (a Mat with the assembled bilinear form testing the divergence
    (curl) against an L2 space) and keyword arguments supplied to ``PETSc.PC.setDivergenceMat``.
    - ``'primal_markers'`` a Function marking degrees of freedom of the solution space to be included in the
    coarse space. Any nonzero value is counted as a marked degree of freedom.
    If a DG(0) Function is provided, then all degrees of freedom on the cell are marked.
    Alternatively, ``'primal_markers'`` can be a list of the global degrees of freedom to
    be supplied directly to ``PETSc.PC.setBDDCPrimalVerticesIS``.
    """

    _prefix = "bddc_"

    def initialize(self, pc):
        prefix = (pc.getOptionsPrefix() or "") + self._prefix

        dm = pc.getDM()
        V = get_function_space(dm).collapse()

        # Create new PC object as BDDC type
        bddcpc = PETSc.PC().create(comm=pc.comm)
        bddcpc.incrementTabLevel(1, parent=pc)
        bddcpc.setOptionsPrefix(prefix)
        bddcpc.setType(PETSc.PC.Type.BDDC)

        opts = PETSc.Options(bddcpc.getOptionsPrefix())

        # Set operators
        assemblers = []
        cellwise = opts.getBool("cellwise", False)
        subdomain_size = opts.getInt("subdomain_size", 0)
        if cellwise and subdomain_size > 0:
            raise ValueError(
                "Set either 'bddc_cellwise' or 'bddc_subdomain_size', not both. "
                "Cellwise subdomains are a 'bddc_subdomain_size' of one.")
        label = partition_cells(V.mesh().unique(), subdomain_size) if subdomain_size > 0 else None

        A, P = pc.getOperators()
        if P.type == "python":
            # Reconstruct P as a MatIS on the subspaces of the subdomains
            ctx = P.getPythonContext()
            v_arg, u_arg = ctx.a.arguments()
            parents = (v_arg.function_space(), u_arg.function_space())
            Wv, Wu = (subspace(Vsub, cellwise=cellwise, label=label) for Vsub in parents)
            form = replace(ctx.a, {v_arg: TestFunction(Wv), u_arg: TrialFunction(Wu)})
            if cellwise:
                # A broken space carries no dofs on a facet, so the closure of
                # the marked facets finds none of them
                bcs = tuple(BrokenDirichletBC(bc.reconstruct(g=0)) for bc in ctx.bcs)
            else:
                bcs = tuple(bc.reconstruct(V=Wv, g=0) for bc in ctx.bcs)
            P, assembleP = assemble_matis(form, bcs, parents)
            assemblers.append(assembleP)

        if P.type != "is":
            raise ValueError(f"Expecting P to be either 'matfree' or 'is', not {P.type}.")

        bddcpc.setOperators(A, P)
        self.assemblers = assemblers

        # we may inject some options, we remove them after calling setFromOptions
        rem_opts = []

        # Do not use CSR of local matrix to define dofs connectivity unless requested
        # Using the CSR only makes sense for H1/H2 problems
        is_h1h2 = V.ufl_element().sobolev_space in {H1, H2}
        if "pc_bddc_use_local_mat_graph" not in opts and (not is_h1h2 or not V.finat_element.has_pointwise_dual_basis):
            opts["pc_bddc_use_local_mat_graph"] = False
            rem_opts.append("pc_bddc_use_local_mat_graph")

        # The local matrix is block diagonal across the strata, so its
        # components are the subdomains
        if label is not None and "pc_bddc_detect_disconnected" not in opts:
            opts["pc_bddc_detect_disconnected"] = True
            rem_opts.append("pc_bddc_detect_disconnected")

        # Get context from DM
        ctx = get_appctx(dm)

        # Handle boundary dofs
        bcs = tuple(ctx._problem.dirichlet_bcs())
        mesh = V.mesh().unique()
        if mesh.extruded and not mesh.extruded_periodic:
            boundary_nodes = numpy.unique(numpy.concatenate(list(map(V.boundary_nodes, ("on_boundary", "top", "bottom")))))
        else:
            boundary_nodes = V.boundary_nodes("on_boundary")
        if len(bcs) == 0:
            dir_nodes = numpy.empty(0, dtype=boundary_nodes.dtype)
        else:
            dir_nodes = numpy.unique(numpy.concatenate([bcdofs(bc, ghost=False) for bc in bcs]))
        neu_nodes = numpy.setdiff1d(boundary_nodes, dir_nodes)

        dir_nodes = V.dof_dset.lgmap.apply(dir_nodes)
        dir_bndr = PETSc.IS().createGeneral(dir_nodes, comm=pc.comm)
        bddcpc.setBDDCDirichletBoundaries(dir_bndr)

        neu_nodes = V.dof_dset.lgmap.apply(neu_nodes)
        neu_bndr = PETSc.IS().createGeneral(neu_nodes, comm=pc.comm)
        bddcpc.setBDDCNeumannBoundaries(neu_bndr)

        appctx = self.get_appctx(pc)

        # Set coordinates if corner selection is requested or needed
        # There's no API to query from PC
        entity_dofs = V.finat_element.entity_dofs()
        vdofs = entity_dofs[min(entity_dofs)]
        has_vertex_dofs = any(len(vdofs[v]) > 0 for v in vdofs)
        corner_selection = opts.getBool("pc_bddc_corner_selection") if "pc_bddc_corner_selection" in opts else has_vertex_dofs
        if corner_selection:
            if "pc_bddc_corner_selection" not in opts:
                opts["pc_bddc_corner_selection"] = True
                rem_opts.append("pc_bddc_corner_selection")
            bddcpc.setCoordinates(get_entity_coordinates(V))

        # Provide extra information for H(div) and H(curl) problems
        tdim = mesh.topological_dimension
        use_divergence = opts.getBool("use_divergence_mat", tdim >= 2 and V.finat_element.formdegree == tdim-1)
        use_gradient = opts.getBool("use_discrete_gradient", tdim >= 3 and V.finat_element.formdegree == 1)

        if use_divergence:
            get_divergence = appctx.get("get_divergence_mat", get_divergence_mat)
            divergence = get_divergence(V, mat_type="is", cellwise=cellwise, label=label)
            try:
                div_args, div_kwargs = divergence
            except ValueError:
                div_args = (divergence,)
                div_kwargs = dict()
            bddcpc.setBDDCDivergenceMat(*div_args, **div_kwargs)
        if use_gradient:
            get_gradient = appctx.get("get_discrete_gradient", get_discrete_gradient)
            gradient = get_gradient(V)
            try:
                grad_args, grad_kwargs = gradient
            except ValueError:
                grad_args = (gradient,)
                grad_kwargs = dict()
            bddcpc.setBDDCDiscreteGradient(*grad_args, **grad_kwargs)

        # Set the user-defined primal (coarse) degrees of freedom
        primal_markers = appctx.get("primal_markers")
        if primal_markers is not None:
            primal_indices = get_primal_indices(V, primal_markers)
            primal_is = PETSc.IS().createGeneral(primal_indices.astype(PETSc.IntType), comm=pc.comm)
            bddcpc.setBDDCPrimalVerticesIS(primal_is)

        if "pc_bddc_check_level" not in opts and "debug" in opts:
            opts.setValue("pc_bddc_check_level", opts["debug"])
            rem_opts.append("pc_bddc_check_level")
        bddcpc.setFromOptions()
        for opt in rem_opts:
            del opts[opt]

        self.pc = bddcpc

    def view(self, pc, viewer=None):
        self.pc.view(viewer=viewer)

    def update(self, pc):
        for c in self.assemblers:
            c()

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)


class BrokenDirichletBC(DirichletBC):
    def __init__(self, bc):
        self.bc = bc
        V = bc.function_space().broken_space()
        g = bc._original_arg
        super().__init__(V, g, bc.sub_domain)

    @cached_property
    def nodes(self):
        u = Function(self.bc.function_space())
        self.bc.set(u, 1)
        u = broken_function(u.function_space(), val=u.dat)
        return numpy.flatnonzero(u.dat.data)


def partition_cells(mesh, target_size, name="subdomains"):
    """Divide the cells that each process holds into connected subdomains.

    Parameters
    ----------
    mesh : MeshGeometry
        The mesh, distributed or not.
    target_size : int
        The wanted number of cells per subdomain. A process holding ``n`` cells
        asks for ``round(n / target_size)`` subdomains, at least one and at
        most one per cell.
    name : str
        The name of the label.

    Returns
    -------
    PETSc.DMLabel
        A label marking the closure of each subdomain. No subdomain spans a
        process boundary, and each process numbers its own apart from those of
        the others, so a shared point carries the same set of strata wherever
        it appears.

    Raises
    ------
    ValueError
        If ``target_size`` is less than one, or if a partitioner is needed and
        PETSc was built without one.

    Notes
    -----
    A graph partitioner divides the dual graph of the owned cells, so the
    subdomain size is a target and not a guarantee. A partitioner may also
    return a part that is empty, or one that is not connected. An empty
    stratum carries no dofs, and ``-pc_bddc_detect_disconnected`` splits a
    disconnected one, so neither needs handling here.
    """
    if target_size < 1:
        raise ValueError(f"Subdomain size must be at least 1, not {target_size}")

    ncells = mesh.cell_set.size
    nsub = min(max(1, round(ncells / target_size)), max(ncells, 1))
    if nsub <= 1:
        ids = numpy.zeros(ncells, dtype=IntType)
        nsub = 1
    elif nsub == ncells:
        ids = numpy.arange(ncells, dtype=IntType)
    elif DEFAULT_PARTITIONER == "simple":
        raise ValueError(
            "Splitting a process into several subdomains needs a graph partitioner, "
            "but PETSc was built without any of parmetis, ptscotch or chaco. "
            "Reconfigure PETSc, or leave the subdomain size unset to get one "
            "subdomain per process.")
    else:
        ids = apply_partitioner(*dual_graph(mesh), nsub)

    # Number this process's strata apart from those of the others
    ids += mesh.comm.scan(nsub) - nsub

    # Mark the closure of the owned cells alone, so that this process
    # contributes only its own strata to a shared point
    plex = mesh.topology_dm
    cStart, cEnd = plex.getHeightStratum(0)
    cells = numpy.full(cEnd - cStart, -1, dtype=IntType)
    cells[mesh.topology.cell_closure[:ncells, -1] - cStart] = ids
    return complete_strata(mesh, dmcommon.create_subdomain_label(plex, cells, name))


def dual_graph(mesh):
    """Return the dual graph of the cells a process owns, in CSR format.

    Parameters
    ----------
    mesh : MeshGeometry
        The mesh, distributed or not.

    Returns
    -------
    xadj : numpy.ndarray
        The start of the adjacency list of each cell, with ``ncells + 1``
        entries.
    adjncy : numpy.ndarray
        The cells adjacent to each cell, one list after another.

    Notes
    -----
    Two cells are adjacent if they share a facet. A facet onto the halo joins
    an owned cell to a cell this process does not partition, so it drops out.
    """
    ncells = mesh.cell_set.size
    facet_cell = mesh.interior_facets.facet_cell.reshape(-1, 2)
    facet_cell = facet_cell[(facet_cell < ncells).all(axis=1)]

    src = numpy.concatenate([facet_cell[:, 0], facet_cell[:, 1]])
    dst = numpy.concatenate([facet_cell[:, 1], facet_cell[:, 0]])
    order = numpy.argsort(src, kind="stable")

    xadj = numpy.zeros(ncells + 1, dtype=IntType)
    numpy.cumsum(numpy.bincount(src, minlength=ncells), out=xadj[1:])
    return xadj, dst[order].astype(IntType)


def apply_partitioner(xadj, adjncy, nsub):
    """Divide a CSR graph into ``nsub`` parts.

    Parameters
    ----------
    xadj, adjncy : numpy.ndarray
        The graph in CSR format, as `dual_graph` returns it.
    nsub : int
        The number of parts.

    Returns
    -------
    numpy.ndarray
        The part of each vertex. A part may be empty, and may be disconnected.
    """
    nvtx = xadj.size - 1
    values = numpy.ones(adjncy.size, dtype=PETSc.ScalarType)
    adj = PETSc.Mat().createAIJWithArrays((nvtx, nvtx), (xadj, adjncy, values),
                                          comm=COMM_SELF)
    adj.assemble()

    # MatPartitioningSetNParts has no petsc4py binding, so the part count goes
    # through the options database
    prefix = "firedrake_partition_cells_"
    opts = PETSc.Options(prefix)
    opts["mat_partitioning_nparts"] = nsub
    try:
        part = PETSc.MatPartitioning().create(comm=COMM_SELF)
        part.setOptionsPrefix(prefix)
        part.setAdjacency(adj)
        part.setType(DEFAULT_PARTITIONER)
        part.setFromOptions()
        iset = PETSc.IS()
        part.apply(iset)
        ids = iset.getIndices().astype(IntType)
        iset.destroy()
        part.destroy()
    finally:
        del opts["mat_partitioning_nparts"]
        adj.destroy()
    return ids


def assemble_matis(form, bcs, parents):
    """Assemble a form posed on subdomain spaces as a ``Mat`` of type ``is``.

    Parameters
    ----------
    form : ufl.Form
        A bilinear form on the subdomain spaces.
    bcs : tuple
        The homogeneous Dirichlet conditions, on the subdomain spaces.
    parents : tuple
        The space each argument of ``form`` was decomposed from.

    Returns
    -------
    PETSc.Mat
        A ``Mat`` of type ``is`` on ``parents``, holding one block per
        subdomain of this process.
    callable
        A callable that assembles the ``Mat`` again.

    Notes
    -----
    A subdomain space already duplicates the dofs on an interface, so the form
    assembles straight into the local matrix that ``MatIS`` wants. The map
    onto ``parents`` then names the same global dof for every copy, which is
    what makes the subdomain matrices sum to the operator.
    """
    from firedrake.assemble import get_assembler

    assembler = get_assembler(form, bcs=bcs, mat_type="is")
    tensor = assembler.assemble()

    W = tuple(arg.function_space() for arg in form.arguments())
    sizes = tuple(V.dof_dset.layout_vec.getSizes() for V in parents)
    lgmaps = tuple(map(parent_local_to_global_map, W, parents))

    localmat = tensor.petscmat.getISLocalMat()
    matis = PETSc.Mat().createIS(sizes, comm=parents[0].comm)
    matis.setISAllowRepeated(any(Wsub != Vsub for Wsub, Vsub in zip(W, parents)))
    matis.setLGMap(*lgmaps)
    matis.setISLocalMat(localmat)
    matis.setUp()
    matis.assemble()

    set_bc_diagonal = bc_diagonal_setter(localmat, W[0], bcs, lgmaps[0], parents[0])
    set_bc_diagonal()

    def update():
        assembler.assemble(tensor=tensor)
        set_bc_diagonal()
        matis.assemble()
    return matis, update


def subdomain_multiplicity(lgmap, parent) -> numpy.ndarray:
    """Count the subdomains that hold each local node of a subspace.

    Parameters
    ----------
    lgmap : PETSc.LGMap
        The map from the local nodes of the subspace to the global nodes of
        ``parent``, as `~firedrake.subspace.parent_local_to_global_map`
        returns it.
    parent : WithGeometry
        The space that the subspace was built from.

    Returns
    -------
    numpy.ndarray
        The number of subdomains holding the node that each local node
        duplicates, counting those of the other processes. A node that the
        map leaves out takes a count of zero.

    Notes
    -----
    ``MatIS`` keeps the same count in ``is->counter``, and builds it the same
    way: it adds a one from every copy onto the global node, then reads the
    total back onto each copy.
    """
    indices = lgmap.block_indices
    held = indices >= 0
    mine = indices[held].astype(PETSc.IntType)

    counts = PETSc.Vec().create(comm=parent.comm)
    counts.setSizes((parent.dof_dset.size, None))
    counts.setUp()
    counts.setValues(mine, numpy.ones(mine.size, dtype=PETSc.ScalarType),
                     addv=PETSc.InsertMode.ADD_VALUES)
    counts.assemble()

    iset = PETSc.IS().createGeneral(mine, comm=parent.comm)
    total = PETSc.Vec().createSeq(mine.size, comm=COMM_SELF)
    scatter = PETSc.Scatter().create(counts, iset, total, None)
    scatter.scatter(counts, total, addv=PETSc.InsertMode.INSERT_VALUES,
                    mode=PETSc.ScatterMode.FORWARD)

    multiplicity = numpy.zeros(indices.size, dtype=PETSc.RealType)
    multiplicity[held] = total.getArray().real
    scatter.destroy()
    total.destroy()
    iset.destroy()
    counts.destroy()
    return multiplicity


def bc_diagonal_setter(mat, W, bcs, lgmap, parent):
    """Return a callable that writes the Dirichlet diagonal the operator needs.

    Parameters
    ----------
    mat : PETSc.Mat
        The local matrix of the ``MatIS``.
    W : WithGeometry
        The subspace that the rows of ``mat`` belong to.
    bcs : tuple
        The homogeneous Dirichlet conditions, on ``W``.
    lgmap : PETSc.LGMap
        The map from the rows of ``mat`` to the global nodes of ``parent``.
    parent : WithGeometry
        The space that ``W`` was built from.

    Returns
    -------
    callable
        A function of no arguments that writes the diagonal. Call it after
        every assembly, since the assembler writes the ones again each time.

    Notes
    -----
    The assembler writes a one on each copy of a constrained node, so the
    subdomain matrices sum to the number of copies instead of to one. Writing
    the reciprocal of that number on every copy makes them sum to one.

    ``PCBDDC`` gets the same treatment from PETSc itself, in
    ``MatISZeroRowsColumnsLocal``. That divides by the count of the map the
    ``MatIS`` carried while the assembler applied the conditions, which names
    a distinct node for every copy, so the division was by one.
    """
    def noop():
        pass

    nrows, ncols = mat.getSize()
    if not bcs or nrows != ncols:
        return noop

    marker = Function(W)
    for bc in bcs:
        bc.set(marker, 1)
    marked = marker.dat.data_ro_with_halos.reshape(-1) > 0.5

    # A ``MatIS`` drops the nodes that its map masks out, so the rows of the
    # local matrix number the nodes that the map keeps
    bs = W.block_size
    held = lgmap.block_indices >= 0
    node_rows = numpy.cumsum(held) - 1
    dof_rows = bs * numpy.repeat(node_rows, bs) + numpy.tile(numpy.arange(bs), held.size)

    take = marked & numpy.repeat(held, bs)
    rows = dof_rows[take].astype(PETSc.IntType)
    values = 1.0 / numpy.repeat(subdomain_multiplicity(lgmap, parent), bs)[take]

    def set_bc_diagonal():
        diagonal = mat.getDiagonal()
        diagonal.getArray()[rows] = values
        mat.setDiagonal(diagonal)
        diagonal.destroy()
    return set_bc_diagonal


def get_restricted_dofs(V, domain):
    W = FunctionSpace(V.mesh(), V.ufl_element()[domain])
    indices = get_restriction_indices(V, W)
    indices = V.dof_dset.lgmap.apply(indices)
    return PETSc.IS().createGeneral(indices, comm=V.comm)


def get_divergence_mat(V, mat_type="is", cellwise=False, label=None):
    """Assemble the exterior derivative of ``V`` tested against a cellwise constant.

    Parameters
    ----------
    V : WithGeometry
        The solution space, in H(div) or in 2D H(curl).
    mat_type : str
        The ``Mat`` type to assemble.
    cellwise : bool
        Whether to take each cell as a subdomain of its own.
    label : PETSc.DMLabel
        A label whose strata mark the subdomains, or None to take the cells of
        a process as one subdomain.

    Returns
    -------
    tuple
        The arguments and keyword arguments of `PETSc.PC.setBDDCDivergenceMat`.
    """
    from firedrake import assemble
    degree = max(as_tuple(V.ufl_element().degree()))
    Q = TensorFunctionSpace(V.mesh(), "DG", 0, variant=f"integral({degree-1})", shape=V.value_shape[:-1])

    # The exterior derivative of a 2D H(curl) space is its curl
    derivative = curl if V.ufl_element().sobolev_space == HCurl else div

    parents = (Q, V)
    if mat_type == "is":
        Wq, Wv = (subspace(Vsub, cellwise=cellwise, label=label) for Vsub in parents)
        form = inner(derivative(TrialFunction(Wv)), TestFunction(Wq)) * dx
        B, _ = assemble_matis(form, (), parents)
    else:
        form = inner(derivative(TrialFunction(V)), TestFunction(Q)) * dx
        B = assemble(form, mat_type=mat_type).petscmat
    return (B,), {}


def get_discrete_gradient(V):
    from firedrake import Constant
    from firedrake.nullspace import VectorSpaceBasis

    Q = FunctionSpace(V.mesh(), curl_to_grad(V.ufl_element()))
    gradient = tabulate_exterior_derivative(Q, V)
    basis = Function(Q)
    try:
        basis.interpolate(Constant(1))
    except NotImplementedError:
        basis.project(Constant(1))
    nsp = VectorSpaceBasis([basis])
    nsp.orthonormalize()
    gradient.setNullSpace(nsp.nullspace())
    if not Q.finat_element.has_pointwise_dual_basis:
        vdofs = get_restricted_dofs(Q, "vertex")
        gradient.compose('_elements_corners', vdofs)

    degree = max(as_tuple(Q.ufl_element().degree()))
    grad_args = (gradient,)
    grad_kwargs = {'order': degree}
    return grad_args, grad_kwargs


def get_primal_indices(V, primal_markers):
    if isinstance(primal_markers, Function):
        marker_space = primal_markers.function_space()
        if marker_space == V:
            markers = primal_markers
        elif marker_space.finat_element.space_dimension() == 1:
            shapes = (V.finat_element.space_dimension(), V.block_size)
            domain = "{[i,j]: 0 <= i < %d and 0 <= j < %d}" % shapes
            instructions = """
            for i, j
                w[i,j] = w[i,j] + t[0]
            end
            """
            markers = Function(V)
            par_loop((domain, instructions), dx, {"w": (markers, INC), "t": (primal_markers, READ)})
        else:
            raise ValueError(f"Expecting markers in either {V.ufl_element()} or DG(0).")
        primal_indices = numpy.flatnonzero(markers.dat.data >= 1E-12)
        primal_indices += V.dof_dset.layout_vec.getOwnershipRange()[0]
    else:
        primal_indices = numpy.asarray(primal_markers, dtype=PETSc.IntType)
    return primal_indices


def get_entity_coordinates(V):
    """
    Return a Function on fd.VectorFunctionSpace(mesh, V.ufl_element()) containing
    the physical coordinates of the entity associated with each degree of freedom of V.
    """
    import firedrake as fd
    from pyop2 import op2
    import numpy as np

    mesh = V.mesh()
    gdim = mesh.geometric_dimension

    base_element = V.ufl_element()
    if isinstance(base_element, (TensorElement, VectorElement)):
        base_element = base_element._sub_element
    V_target = fd.VectorFunctionSpace(mesh, base_element)
    cg1_coord = fd.VectorFunctionSpace(mesh, "CG", 1)

    out_coords = fd.Function(V_target)
    cg1_coords = fd.Function(cg1_coord).interpolate(mesh.coordinates)

    finat_element = V.finat_element
    cg1_finat = cg1_coord.finat_element
    active_entities = [
        (dim, ent_num)
        for dim, entities in finat_element.entity_dofs().items()
        for ent_num, dofs in entities.items()
        if dofs
    ]
    num_entities = len(active_entities)

    def flatten_space_mapping(entities, query_map):
        offsets = np.zeros(len(entities) + 1, dtype=np.int32)
        flat_list = []

        for idx, (dim, ent_num) in enumerate(entities):
            offsets[idx] = len(flat_list)
            flat_list.extend(query_map[dim][ent_num])

        offsets[-1] = len(flat_list)
        return offsets, np.array(flat_list, dtype=np.int32)

    # Flatten both target (V) and source (CG1) layouts
    target_dofs_map = finat_element.entity_dofs()
    cg1_closure_map = cg1_finat.entity_closure_dofs()

    v_offsets, v_flat = flatten_space_mapping(active_entities, target_dofs_map)
    cg1_offsets, cg1_flat = flatten_space_mapping(active_entities, cg1_closure_map)

    total_v_dofs = len(v_flat)
    total_cg1_dofs = len(cg1_flat)
    kernel_name = "compute_entity_coords"
    kernel_code = f"""
    void {kernel_name}(PetscScalar *out, PetscScalar *cg1_coords) {{

        // Target space represented as a flattened pair of 1D arrays
        const int v_offsets[{num_entities + 1}] = {{ {", ".join(map(str, v_offsets))} }};
        const int v_flat_mapping[{total_v_dofs}] = {{ {", ".join(map(str, v_flat))} }};

        // Source CG1 space represented as a flattened pair of 1D arrays
        const int cg1_offsets[{num_entities + 1}] = {{ {", ".join(map(str, cg1_offsets))} }};
        const int cg1_flat_mapping[{total_cg1_dofs}] = {{ {", ".join(map(str, cg1_flat))} }};

        // Loop over the flat entity index
        for (int e = 0; e < {num_entities}; ++e) {{
            int v_start = v_offsets[e];
            int v_end = v_offsets[e + 1];

            int cg1_start = cg1_offsets[e];
            int cg1_end = cg1_offsets[e + 1];
            int num_cg1_dofs = cg1_end - cg1_start;

            // Compute structural centroid tracking coordinates from CG1 vertices
            PetscScalar ent_coord[{gdim}] = {{0.0}};

            for (int j = cg1_start; j < cg1_end; ++j) {{
                int src_dof = cg1_flat_mapping[j];
                for (int c = 0; c < {gdim}; ++c) {{
                    ent_coord[c] += cg1_coords[src_dof * {gdim} + c];
                }}
            }}

            // Normalize physical coordinates for the specific entity space
            for (int c = 0; c < {gdim}; ++c) {{
                ent_coord[c] /= (PetscScalar)num_cg1_dofs;
            }}

            // Inner loop traversing the linear 1D slice for the target DoFs
            for (int i = v_start; i < v_end; ++i) {{
                int dest_dof = v_flat_mapping[i];

                for (int c = 0; c < {gdim}; ++c) {{
                    out[dest_dof * {gdim} + c] = ent_coord[c];
                }}
            }}
        }}
    }}
    """
    kernel = op2.Kernel(kernel_code, kernel_name)
    op2.par_loop(kernel, mesh.cell_set,
                 out_coords.dat(op2.WRITE, out_coords.cell_node_map()),
                 cg1_coords.dat(op2.READ, cg1_coords.cell_node_map()))
    return out_coords.dat.data.real.repeat(V.block_size, axis=0)

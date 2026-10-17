from itertools import repeat

from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.facet_split import get_restriction_indices
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, TensorFunctionSpace
from firedrake.preconditioners.fdm import broken_function, tabulate_exterior_derivative
from firedrake.preconditioners.hiptmair import curl_to_grad
from functools import cached_property, partial

from firedrake.parloops import par_loop, INC, READ
from firedrake.bcs import DirichletBC
from firedrake.mesh import DomainDecomposition, Submesh
from ufl import Form, H1, H2, JacobianDeterminant, div, dx, inner, replace
from finat.ufl import BrokenElement, TensorElement, VectorElement
from pyop2.mpi import COMM_SELF, MPI
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
    - ``'bddc_subdomain_size'`` to split the cells of each process into subdomains of
    about this many cells, with the partitioner set by ``'bddc_petscpartitioner_type'``,
    - ``'bddc_matfree'`` to set up a matrix-free MatIS if A.type == python,
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
        matfree = opts.getBool("matfree", False)

        # Get context from DM
        ctx = get_appctx(dm)

        # Set operators
        assemblers = []
        A, P = pc.getOperators()
        subdomain_size = opts.getInt("subdomain_size") if "subdomain_size" in opts else None
        decomposition = {}
        if P.type == "python" or subdomain_size is not None:
            # Reconstruct P as MatIS on the subdomains
            decomposition = {"cellwise": opts.getBool("cellwise", False),
                             "subdomain_size": subdomain_size,
                             "options_prefix": prefix}
            if P.type == "python":
                P, assembleP = create_matis(P, "aij", **decomposition)
            elif ctx.Jp is None:
                P, assembleP = create_matis(ctx.J, "aij", bcs=ctx.bcs_J, **decomposition)
            else:
                P, assembleP = create_matis(ctx.Jp, "aij", bcs=ctx.bcs_Jp, **decomposition)
            assemblers.append(assembleP)

        if P.type != "is":
            raise ValueError(f"Expecting P to be either 'matfree' or 'is', not {P.type}.")

        if A.type == "python" and matfree:
            # Reconstruct A as MatIS on the subdomains of P
            A, assembleA = create_matis(A, "matfree", **decomposition)
            assemblers.append(assembleA)
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
            allow_repeated = P.getISAllowRepeated()
            get_divergence = appctx.get("get_divergence_mat", partial(get_divergence_mat, decomposition=decomposition or None))
            divergence = get_divergence(V, mat_type="is", allow_repeated=allow_repeated)
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


def subdomain_decomposition(mesh, subdomain_size=None, options_prefix=None):
    """Return the decomposition of a mesh into the subdomains of each process.

    Parameters
    ----------
    mesh : MeshGeometry
        The mesh to decompose.
    subdomain_size : int | None
        The target number of cells in each subdomain. If ``None``, each
        process holds a single subdomain.
    options_prefix : str | None
        The options prefix of the ``PETSc.Partitioner`` that splits the cells
        of each process.

    Returns
    -------
    MeshGeometry
        The :func:`~.DomainDecomposition` of ``mesh``, whose subdomains each
        lie on one process.
    """
    key = (subdomain_size, options_prefix)
    cache = mesh._shared_data_cache["bddc_subdomain_decomposition"]
    try:
        return cache[key]
    except KeyError:
        pass
    topology = mesh.topology
    ncells = topology.cell_set.size
    nparts = 1 if subdomain_size is None else max(1, -(-ncells // subdomain_size))
    parts = numpy.zeros(ncells, dtype=PETSc.IntType)
    if nparts > 1:
        # Partition the graph of the owned cells connected through their facets
        facet_cells = topology.interior_facets.facet_cell
        facet_cells = facet_cells[numpy.all(facet_cells < ncells, axis=1)]
        rows = numpy.concatenate((facet_cells[:, 0], facet_cells[:, 1]))
        cols = numpy.concatenate((facet_cells[:, 1], facet_cells[:, 0]))
        start = numpy.zeros(ncells + 1, dtype=PETSc.IntType)
        numpy.cumsum(numpy.bincount(rows, minlength=ncells), out=start[1:])
        adjacency = cols[numpy.argsort(rows, kind="stable")].astype(PETSc.IntType)
        partitioner = PETSc.Partitioner().create(comm=COMM_SELF)
        partitioner.setOptionsPrefix(options_prefix)
        partitioner.setFromOptions()
        part_section, partition = partitioner.partition(nparts, start, adjacency)
        sizes = [part_section.getDof(part) for part in range(nparts)]
        parts[partition.indices] = numpy.repeat(numpy.arange(nparts, dtype=PETSc.IntType), sizes)
        partitioner.destroy()

    # Number the subdomains of all processes consecutively
    first = mesh.comm.exscan(nparts) or 0
    plex = mesh.topology_dm
    label_name = "firedrake_bddc_subdomains"
    plex.createLabel(label_name)
    label = plex.getLabel(label_name)
    cells = topology.cell_closure[:ncells, -1]
    order = numpy.argsort(parts, kind="stable")
    bounds = numpy.searchsorted(parts[order], numpy.arange(nparts + 1))
    for part in range(nparts):
        subdomain_cells = cells[order[bounds[part]:bounds[part+1]]].astype(PETSc.IntType)
        label.setStratumIS(first + part, PETSc.IS().createGeneral(subdomain_cells, comm=COMM_SELF))
    dd = DomainDecomposition(mesh, label_name=label_name, ignore_halo=True)
    plex.removeLabel(label_name)
    return cache.setdefault(key, dd)


def create_matis(a, local_mat_type, cellwise=False, bcs=(), subdomain_size=None, options_prefix=None):
    from firedrake.assemble import get_assembler

    def local_mesh(mesh):
        dd = subdomain_decomposition(mesh, subdomain_size, options_prefix)
        if local_mat_type == "aij" or mesh.comm.size == 1:
            return dd
        # A matrix-free local matrix acts on a mesh of the local subdomains
        cache = dd._shared_data_cache["bddc_local_submesh"]
        try:
            return cache[None]
        except KeyError:
            return cache.setdefault(None, Submesh(dd, ignore_halo=True, comm=COMM_SELF))

    def local_space(V, cellwise):
        mesh = local_mesh(V.mesh().unique())
        element = BrokenElement(V.ufl_element()) if cellwise else None
        return V.reconstruct(mesh=mesh, element=element)

    def local_argument(arg, cellwise):
        return arg.reconstruct(function_space=local_space(arg.function_space(), cellwise))

    def local_integral(it):
        extra_domain_integral_type_map = dict(it.extra_domain_integral_type_map())
        extra_domain_integral_type_map[it.ufl_domain()] = it.integral_type()
        return it.reconstruct(domain=local_mesh(it.ufl_domain()),
                              extra_domain_integral_type_map=extra_domain_integral_type_map)

    def local_bc(bc, cellwise):
        V = bc.function_space()
        Vsub = local_space(V, False)
        sub_domain = list(bc.sub_domain)
        if "on_boundary" in sub_domain:
            sub_domain.remove("on_boundary")
            sub_domain.extend(V.mesh().unique().exterior_facets.unique_markers)

        valid_markers = Vsub.mesh().unique().exterior_facets.unique_markers
        sub_domain = list(set(sub_domain) & set(valid_markers))
        bc = bc.reconstruct(V=Vsub, g=0, sub_domain=sub_domain)
        if cellwise:
            bc = BrokenDirichletBC(bc)
        return bc

    def local_to_global_map(V, cellwise):
        u = Function(V)
        shp = u.dat.data_ro.shape
        u.dat.data_wo[...] = numpy.arange(*V.dof_dset.layout_vec.getOwnershipRange()).reshape(shp)

        Vsub = local_space(V, False)
        usub = Function(Vsub).assign(u)
        if cellwise:
            usub = broken_function(usub.function_space(), val=usub.dat)
        indices = usub.dat.data_ro.astype(PETSc.IntType)
        return PETSc.LGMap().create(indices, comm=V.comm)

    if isinstance(a, Form):
        form = a
        args = a.arguments()
        comm = args[0].function_space().comm
        sizes = tuple(arg.function_space().dof_dset.layout_vec.getSizes() for arg in args)
    elif isinstance(a, PETSc.Mat):
        assert a.type == "python"
        ctx = a.getPythonContext()
        form = ctx.a
        bcs = ctx.bcs
        comm = a.comm
        sizes = a.getSizes()

    local_form = replace(form, {arg: local_argument(arg, cellwise) for arg in form.arguments()})
    local_form = Form(list(map(local_integral, local_form.integrals())))
    local_bcs = tuple(map(local_bc, bcs, repeat(cellwise)))

    if local_mat_type == "aij":
        # The local matrix of the MatIS on the subdomains holds the Neumann
        # matrices of the subdomains on this process
        assembler = get_assembler(local_form, bcs=local_bcs, mat_type="is")
        tensor = assembler.assemble()
        local_mat = tensor.petscmat.getISLocalMat()
    else:
        assembler = get_assembler(local_form, bcs=local_bcs, mat_type=local_mat_type)
        tensor = assembler.assemble()
        local_mat = tensor.petscmat

    rmap = local_to_global_map(form.arguments()[0].function_space(), cellwise)
    cmap = local_to_global_map(form.arguments()[1].function_space(), cellwise)
    # Subdomains on the same process share the degrees of freedom on their interface
    repeated = any(len(numpy.unique(m.indices)) < len(m.indices) for m in (rmap, cmap))
    repeated = form.arguments()[0].function_space().comm.allreduce(repeated, op=MPI.LOR)

    Amatis = PETSc.Mat().createIS(sizes, comm=comm)
    Amatis.setISAllowRepeated(repeated)
    Amatis.setLGMap(rmap, cmap)
    Amatis.setISLocalMat(local_mat)
    Amatis.setUp()
    Amatis.assemble()

    def update():
        assembler.assemble(tensor=tensor)
        Amatis.assemble()
    return Amatis, update


def get_restricted_dofs(V, domain):
    W = FunctionSpace(V.mesh(), V.ufl_element()[domain])
    indices = get_restriction_indices(V, W)
    indices = V.dof_dset.lgmap.apply(indices)
    return PETSc.IS().createGeneral(indices, comm=V.comm)


def get_divergence_mat(V, mat_type="is", allow_repeated=False, decomposition=None):
    from firedrake import assemble
    degree = max(as_tuple(V.ufl_element().degree()))
    Q = TensorFunctionSpace(V.mesh(), "DG", 0, variant=f"integral({degree-1})", shape=V.value_shape[:-1])

    if mat_type == "is" and decomposition is not None:
        # The divergence must have the local numbering of the velocity in the
        # preconditioner built by create_matis on these subdomains
        form = inner(div(TrialFunction(V)), TestFunction(Q)) * dx
        B, _ = create_matis(form, "aij", **decomposition)
    elif V.finat_element.complex.is_macrocell() or V.finat_element.formdegree != Q.finat_element.formdegree-1:
        form = inner(div(TrialFunction(V)), TestFunction(Q)) * dx
        if mat_type == "is" and allow_repeated:
            B, _ = create_matis(form, "aij", cellwise=allow_repeated)
        else:
            B = assemble(form, mat_type=mat_type).petscmat
    else:
        B = tabulate_exterior_derivative(V, Q, mat_type=mat_type, allow_repeated=allow_repeated)
        Jdet = JacobianDeterminant(V.mesh())
        s = assemble(inner(TrialFunction(Q)*(1/Jdet), TestFunction(Q))*dx(degree=0), diagonal=True)
        with s.dat.vec as svec:
            B.diagonalScale(svec, None)

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

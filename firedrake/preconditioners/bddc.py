from itertools import repeat

from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.facet_split import get_restriction_indices
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace
from firedrake.preconditioners.fdm import broken_function, tabulate_exterior_derivative
from firedrake.preconditioners.hiptmair import curl_to_grad
from functools import cached_property

from firedrake.parloops import par_loop, INC, READ
from firedrake.bcs import DirichletBC
from firedrake.mesh import Submesh
from ufl import Form, L2, H1, H2, JacobianDeterminant, div, dx, inner, replace
from finat.ufl import BrokenElement
from pyop2.mpi import COMM_SELF
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
        V = get_function_space(dm)

        # Create new PC object as BDDC type
        bddcpc = PETSc.PC().create(comm=pc.comm)
        bddcpc.incrementTabLevel(1, parent=pc)
        bddcpc.setOptionsPrefix(prefix)
        bddcpc.setType(PETSc.PC.Type.BDDC)

        opts = PETSc.Options(bddcpc.getOptionsPrefix())
        matfree = opts.getBool("matfree", False)

        # Set operators
        assemblers = []
        A, P = pc.getOperators()
        if P.type == "python":
            # Reconstruct P as MatIS
            cellwise = opts.getBool("cellwise", False)
            P, assembleP = create_matis(P, "aij", cellwise=cellwise)
            assemblers.append(assembleP)

        if P.type != "is":
            raise ValueError(f"Expecting P to be either 'matfree' or 'is', not {P.type}.")

        if A.type == "python" and matfree:
            # Reconstruct A as MatIS
            A, assembleA = create_matis(A, "matfree", cellwise=P.getISAllowRepeated())
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
            allow_repeated = P.getISAllowRepeated()
            get_divergence = appctx.get("get_divergence_mat", get_divergence_mat)
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


def create_matis(a, local_mat_type, cellwise=False, bcs=()):
    from firedrake.assemble import get_assembler

    def local_mesh(mesh):
        key = "local_submesh"
        cache = mesh._shared_data_cache["local_submesh_cache"]
        try:
            return cache[key]
        except KeyError:
            if mesh.comm.size > 1:
                submesh = Submesh(mesh, ignore_halo=True, comm=COMM_SELF)
            else:
                submesh = None
            return cache.setdefault(key, submesh)

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

    assembler = get_assembler(local_form, bcs=local_bcs, mat_type=local_mat_type)
    tensor = assembler.assemble()

    rmap = local_to_global_map(form.arguments()[0].function_space(), cellwise)
    cmap = local_to_global_map(form.arguments()[1].function_space(), cellwise)

    Amatis = PETSc.Mat().createIS(sizes, comm=comm)
    Amatis.setISAllowRepeated(cellwise)
    Amatis.setLGMap(rmap, cmap)
    Amatis.setISLocalMat(tensor.petscmat)
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


def get_divergence_mat(V, mat_type="is", allow_repeated=False):
    from firedrake import assemble
    degree = max(as_tuple(V.ufl_element().degree()))
    Q = TensorFunctionSpace(V.mesh(), "DG", 0, variant=f"integral({degree-1})", shape=V.value_shape[:-1])

    if V.finat_element.complex.is_macrocell() or V.finat_element.formdegree != Q.finat_element.formdegree-1:
        form = inner(div(TrialFunction(V)), TestFunction(Q)) * dx
        if mat_type == "is" and allow_repeated:
            B, _ = create_matis(form, "aij", allow_repeated)
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

    V_target = fd.VectorFunctionSpace(mesh, V.ufl_element())
    V_cg1_coord = fd.VectorFunctionSpace(mesh, "CG", 1)

    out_coords = fd.Function(V_target)
    cg1_coords = fd.Function(V_cg1_coord).interpolate(mesh.coordinates)

    finat_element = V.finat_element
    cg1_finat = V_cg1_coord.finat_element
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

    kernel_code = f"""
    void compute_target_coords_loop(PetscScalar *out, PetscScalar *cg1_coords) {{

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

    # 6. Parallel Loop over the full cell mapping set using op2.WRITE
    kernel = op2.Kernel(kernel_code, "compute_target_coords_loop")

    op2.par_loop(kernel, mesh.cell_set,
                 out_coords.dat(op2.WRITE, out_coords.cell_node_map()),
                 cg1_coords.dat(op2.READ, cg1_coords.cell_node_map()))

    return out_coords.dat.data

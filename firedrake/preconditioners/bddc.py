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
from firedrake.parloops import par_loop, INC, READ
from firedrake.utils import cached_property
from firedrake.bcs import DirichletBC
from firedrake.mesh import Submesh
from ufl import Form, H1, H2, JacobianDeterminant, dx, inner, replace
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
    coarse space. If a DG(0) Function is provided, then all degrees of freedom on the cell
    are marked. Any nonzero value is counted as a marked degree of freedom/cell.
    Alternatively, ``'primal_markers'`` can be a list with the global degrees of freedom
    to be supplied directly to ``PETSc.PC.setBDDCPrimalVerticesIS``.
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

        # Do not use CSR of local matrix to define dofs connectivity unless requested
        # Using the CSR only makes sense for H1/H2 problems
        is_h1h2 = V.ufl_element().sobolev_space in {H1, H2}
        if "pc_bddc_use_local_mat_graph" not in opts and (not is_h1h2 or not V.finat_element.has_pointwise_dual_basis):
            opts["pc_bddc_use_local_mat_graph"] = False

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

        # Set coordinates only if corner selection is requested
        # There's no API to query from PC
        if "pc_bddc_corner_selection" in opts:
            degree = max(as_tuple(V.ufl_element().degree()))
            variant = V.ufl_element().variant()
            W = VectorFunctionSpace(mesh, "Lagrange", degree, variant=variant)
            coords = Function(W).interpolate(mesh.coordinates)
            bddcpc.setCoordinates(coords.dat.data_ro.repeat(V.block_size, axis=0))

        tdim = mesh.topological_dimension
        if tdim >= 2 and V.finat_element.formdegree == tdim-1:
            allow_repeated = P.getISAllowRepeated()
            get_divergence = appctx.get("get_divergence_mat", get_divergence_mat)
            divergence = get_divergence(V, mat_type="is", allow_repeated=allow_repeated)
            try:
                div_args, div_kwargs = divergence
            except ValueError:
                div_args = (divergence,)
                div_kwargs = dict()
            bddcpc.setBDDCDivergenceMat(*div_args, **div_kwargs)

        elif tdim >= 3 and V.finat_element.formdegree == 1:
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

        bddcpc.setFromOptions()
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


def create_matis(Amat, local_mat_type, cellwise=False):
    from firedrake.assemble import get_assembler

    def local_mesh(mesh):
        key = "local_submesh"
        cache = mesh._shared_data_cache["local_submesh_cache"]
        try:
            return cache[key]
        except KeyError:
            if mesh.comm.size > 1:
                submesh = Submesh(mesh, mesh.topological_dimension, None, ignore_halo=True, reorder=False, comm=COMM_SELF)
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
        u.dat.data_wo[:] = numpy.arange(*V.dof_dset.layout_vec.getOwnershipRange())

        Vsub = local_space(V, False)
        usub = Function(Vsub).assign(u)
        if cellwise:
            usub = broken_function(usub.function_space(), val=usub.dat)
        indices = usub.dat.data_ro.astype(PETSc.IntType)
        return PETSc.LGMap().create(indices, comm=V.comm)

    assert Amat.type == "python"
    ctx = Amat.getPythonContext()
    form = ctx.a
    bcs = ctx.bcs

    local_form = replace(form, {arg: local_argument(arg, cellwise) for arg in form.arguments()})
    local_form = Form(list(map(local_integral, local_form.integrals())))
    local_bcs = tuple(map(local_bc, bcs, repeat(cellwise)))

    assembler = get_assembler(local_form, bcs=local_bcs, mat_type=local_mat_type)
    tensor = assembler.assemble()

    rmap = local_to_global_map(form.arguments()[0].function_space(), cellwise)
    cmap = local_to_global_map(form.arguments()[1].function_space(), cellwise)

    Amatis = PETSc.Mat().createIS(Amat.getSizes(), comm=Amat.getComm())
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

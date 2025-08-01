from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.facet_split import restrict, get_restriction_indices
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace
from firedrake.preconditioners.fdm import tabulate_exterior_derivative
from firedrake.preconditioners.hiptmair import curl_to_grad
from ufl import curl, div, H1, H2, HCurl, HDiv, inner, dx, JacobianDeterminant
from pyop2.utils import as_tuple
import numpy

__all__ = ("BDDCPC",)


class BDDCPC(PCBase):
    """PC for PETSc PCBDDC (Balancing Domain Decomposition by Constraints).
    This is a domain decomposition method using subdomains defined by the
    blocks in a Mat of type IS.

    Internally, this PC creates a PETSc PCBDDC object that can be controlled by
    the options:
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
    """

    _prefix = "bddc_"

    def initialize(self, pc):
        # Get context from pc
        _, P = pc.getOperators()
        dm = pc.getDM()
        self.prefix = (pc.getOptionsPrefix() or "") + self._prefix

        V = get_function_space(dm)
        variant = V.ufl_element().variant()
        sobolev_space = V.ufl_element().sobolev_space

        # Create new PC object as BDDC type
        bddcpc = PETSc.PC().create(comm=pc.comm)
        bddcpc.incrementTabLevel(1, parent=pc)
        bddcpc.setOptionsPrefix(self.prefix)
        bddcpc.setOperators(*pc.getOperators())
        bddcpc.setType(PETSc.PC.Type.BDDC)

        opts = PETSc.Options(bddcpc.getOptionsPrefix())
        # Do not use CSR of local matrix to define dofs connectivity unless requested
        # Using the CSR only makes sense for H1/H2 problems
        is_h1h2 = sobolev_space in [H1, H2]
        if "pc_bddc_use_local_mat_graph" not in opts and (not is_h1h2 or variant == "fdm"):
            opts["pc_bddc_use_local_mat_graph"] = False

        # Handle boundary dofs
        ctx = get_appctx(dm)
        bcs = tuple(ctx._problem.bcs)
        if V.extruded:
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
        degree = max(as_tuple(V.ufl_element().degree()))

        # Set coordinates only if corner selection is requested
        # There's no API to query from PC
        if "pc_bddc_corner_selection" in opts:
            W = VectorFunctionSpace(V.mesh(), "Lagrange", degree, variant=variant)
            coords = Function(W).interpolate(V.mesh().coordinates)
            bddcpc.setCoordinates(coords.dat.data_ro.repeat(V.block_size, axis=0))

        tdim = V.mesh().topological_dimension()
        if tdim >= 2 and V.finat_element.formdegree == tdim-1:
            get_divergence = appctx.get("get_divergence_mat", get_divergence_mat)
            qdegree = degree - 1
            if variant in ("fdm", "integral"):
                qdegree = 0
            A, P = pc.getOperators()
            allow_repeated = P.getISAllowRepeated()
            divergence = get_divergence(V, degree=qdegree, mat_type="is", allow_repeated=allow_repeated)
            try:
                div_args, div_kwargs = divergence
            except ValueError:
                div_args = (divergence,)
                div_kwargs = dict()
            bddcpc.setBDDCDivergenceMat(*div_args, **div_kwargs)
        elif sobolev_space == HCurl:
            # Should we use a callable like for hypre_ams?
            gradient = appctx.get("discrete_gradient", None)
            if gradient is None:
                from firedrake.preconditioners.fdm import tabulate_exterior_derivative
                from firedrake.preconditioners.hiptmair import curl_to_grad
                Q = V.reconstruct(element=curl_to_grad(V.ufl_element()))
                gradient = tabulate_exterior_derivative(Q, V)
                if variant == 'fdm':
                    corners = get_vertex_dofs(Q)
                    gradient.compose('_elements_corners', corners)
                grad_args = (gradient,)
                grad_kwargs = dict()
            bddcpc.setBDDCDiscreteGradient(*grad_args, **grad_kwargs)

        bddcpc.setFromOptions()
        self.pc = bddcpc

    def view(self, pc, viewer=None):
        self.pc.view(viewer=viewer)

    def update(self, pc):
        pass

    def apply(self, pc, x, y):
        self.pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self.pc.applyTranspose(x, y)


def get_vertex_dofs(V):
    W = V.reconstruct(element=restrict(V.ufl_element(), "vertex"))
    indices = get_restriction_indices(V, W)
    indices = V.dof_dset.lgmap.apply(indices)
    vertex_dofs = PETSc.IS().createGeneral(indices, comm=V.comm)
    return vertex_dofs


def get_divergence_mat(V, degree=None, mat_type="aij", allow_repeated=False):
    sobolev_space = V.ufl_element().sobolev_space
    vdegree = max(as_tuple(V.ufl_element().degree()))
    d = {HCurl: curl, HDiv: div}[sobolev_space]
    if V.shape == ():
        make_function_space = FunctionSpace
    elif len(V.shape) == 1:
        make_function_space = VectorFunctionSpace
    else:
        make_function_space = TensorFunctionSpace

    if degree is None:
        degree = vdegree-1
    Q = make_function_space(V.mesh(), "DG", degree)
    if False:
        from firedrake import assemble
        b = inner(d(TrialFunction(V)), TestFunction(Q)) * dx(degree=degree+vdegree-1)
        B = assemble(b, mat_type=mat_type).petscmat
    else:
        B = tabulate_exterior_derivative(V, Q, mat_type=mat_type, allow_repeated=allow_repeated)
        # Fix sign
        tdim = V.mesh().topological_dimension()
        alpha = (-1) ** (tdim-1)
        Jdet = JacobianDeterminant(V.mesh())
        s = Function(Q).interpolate(alpha * abs(Jdet) / Jdet)
        with s.dat.vec as svec:
            B.diagonalScale(svec, None)

    return (B,), dict()


def get_discrete_gradient(V):
    degree = max(as_tuple(V.ufl_element().degree()))
    variant = V.ufl_element().variant()
    Q = FunctionSpace(V.mesh(), curl_to_grad(V.ufl_element()))
    gradient = tabulate_exterior_derivative(Q, V)
    if variant == 'fdm':
        corners = get_vertex_dofs(Q)
        gradient.compose('_elements_corners', corners)
    grad_args = (gradient,)
    grad_kwargs = {'order': degree}
    return grad_args, grad_kwargs

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
from ufl import H1, H2, inner, dx, JacobianDeterminant
from pyop2.utils import as_tuple
import gem
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
        if "pc_bddc_use_local_mat_graph" not in opts and (not is_h1h2 or not is_lagrange(V.finat_element)):
            opts["pc_bddc_use_local_mat_graph"] = False

        # Handle boundary dofs
        ctx = get_appctx(dm)
        bcs = tuple(ctx._problem.dirichlet_bcs())
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

        tdim = V.mesh().topological_dimension
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


def get_restricted_dofs(V, domain):
    W = FunctionSpace(V.mesh(), restrict(V.ufl_element(), domain))
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
    if not is_lagrange(Q.finat_element):
        vdofs = get_restricted_dofs(Q, "vertex")
        gradient.compose('_elements_corners', vdofs)

    degree = max(as_tuple(Q.ufl_element().degree()))
    grad_args = (gradient,)
    grad_kwargs = {'order': degree}
    return grad_args, grad_kwargs


def is_lagrange(finat_element):
    """Returns whether finat_element.dual_basis consists only of point evaluation dofs."""
    try:
        Q, ps = finat_element.dual_basis
    except NotImplementedError:
        return False
    # Inspect the weight matrix
    # Lagrange elements have gem.Delta as the only terminal nodes
    children = [Q]
    while children:
        nodes = []
        for c in children:
            if isinstance(c, gem.Delta):
                pass
            elif isinstance(c, gem.gem.Terminal):
                return False
            else:
                nodes.extend(c.children)
        children = nodes
    return True

from firedrake.preconditioners.base import PCBase
from firedrake.preconditioners.patch import bcdofs
from firedrake.preconditioners.facet_split import restrict, get_restriction_indices
from firedrake.petsc import PETSc
from firedrake.dmhooks import get_function_space, get_appctx
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace, VectorFunctionSpace, TensorFunctionSpace
from ufl import curl, div, HCurl, HDiv, inner, dx
from pyop2.utils import as_tuple
import finat
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

    This PC also inspects optional arguments supplied in the application context:
    - ``'discrete_gradient'`` for 3D problems in H(curl), this sets the arguments
    (a Mat tabulating the gradient of the auxiliary H1 space) and
    keyword arguments supplied to ``PETSc.PC.setBDDCDiscreteGradient``.
    - ``'divergence_mat'`` for problems in H(div) (resp. 2D H(curl)), this sets the Mat with the
    assembled bilinear form testing the divergence (curl) against an L2 space.
    """

    _prefix = "bddc_"

    def initialize(self, pc):
        # Get context from pc
        _, P = pc.getOperators()
        dm = pc.getDM()
        self.prefix = (pc.getOptionsPrefix() or "") + self._prefix

        V = get_function_space(dm)
        variant = V.ufl_element().variant()
        is_lag = is_lagrange(V.finat_element)

        # Create new PC object as BDDC type
        bddcpc = PETSc.PC().create(comm=pc.comm)
        bddcpc.incrementTabLevel(1, parent=pc)
        bddcpc.setOptionsPrefix(self.prefix)
        bddcpc.setOperators(*pc.getOperators())
        bddcpc.setType(PETSc.PC.Type.BDDC)

        opts = PETSc.Options(bddcpc.getOptionsPrefix())
        # Do not use CSR of local matrix to define dofs connectivity unless requested
        # Using the CSR only makes sense for H1 problems
        if "pc_bddc_use_local_mat_graph" not in opts and (not is_lag or variant == "fdm"):
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

        V.dof_dset.lgmap.apply(dir_nodes, result=dir_nodes)
        dir_bndr = PETSc.IS().createGeneral(dir_nodes, comm=pc.comm)
        bddcpc.setBDDCDirichletBoundaries(dir_bndr)

        V.dof_dset.lgmap.apply(neu_nodes, result=neu_nodes)
        neu_bndr = PETSc.IS().createGeneral(neu_nodes, comm=pc.comm)
        bddcpc.setBDDCNeumannBoundaries(neu_bndr)

        appctx = self.get_appctx(pc)
        sobolev_space = V.ufl_element().sobolev_space

        # Set coordinates
        if is_lag:
            degree = V.ufl_element().embedded_superdegree
            W = VectorFunctionSpace(V.mesh(), "Lagrange", degree, variant=variant)
            coords = Function(W).interpolate(V.mesh().coordinates)
            bddcpc.setCoordinates(coords.dat.data_ro.repeat(V.block_size, axis=0))

        tdim = V.mesh().topological_dimension()
        degree = max(as_tuple(V.ufl_element().degree()))
        if tdim >= 2 and V.finat_element.formdegree == tdim-1:
            B = appctx.get("divergence_mat", None)
            if B is None:
                from firedrake.assemble import assemble
                d = {HCurl: curl, HDiv: div}[sobolev_space]
                Q = V.reconstruct(family="DG", degree=degree-1)
                b = inner(d(TrialFunction(V)), TestFunction(Q)) * dx(degree=2*(degree-1))
                B = assemble(b, mat_type="is")
            bddcpc.setBDDCDivergenceMat(B.petscmat)
        elif sobolev_space == HCurl:
            gradient = appctx.get("discrete_gradient", None)
            if gradient is None:
                from firedrake.preconditioners.fdm import tabulate_exterior_derivative
                from firedrake.preconditioners.hiptmair import curl_to_grad
                Q = V.reconstruct(element=curl_to_grad(V.ufl_element()))
                gradient = tabulate_exterior_derivative(Q, V)
                corners = get_vertex_dofs(Q)
                gradient.compose('_elements_corners', corners)
                grad_args = (gradient,)
                grad_kwargs = {'order': degree}
            else:
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


def get_vertex_dofs(V):
    W = V.reconstruct(element=restrict(V.ufl_element(), "vertex"))
    indices = get_restriction_indices(V, W)
    V.dof_dset.lgmap.apply(indices, result=indices)
    vertex_dofs = PETSc.IS().createGeneral(indices, comm=V.comm)
    return vertex_dofs


def is_lagrange(finat_element):

    if isinstance(finat_element, finat.FlattenedDimensions):
        return is_lagrange(finat_element.product)
    elif isinstance(finat_element, finat.TensorProductElement):
        return all(map(is_lagrange, finat_element.factors))
    elif isinstance(finat_element, finat.EnrichedElement):
        return all(map(is_lagrange, finat_element.elements))

    nodes = finat_element.fiat_equivalent.dual_basis()
    for node in nodes:
        try:
            pt, = node.get_point_dict()
        except (ValueError, TypeError):
            return False
    return True

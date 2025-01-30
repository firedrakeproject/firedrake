import abc
import numpy

from pyop2.utils import as_tuple
from firedrake.bcs import DirichletBC
from firedrake.petsc import PETSc
from firedrake.preconditioners.base import PCBase
from firedrake.functionspace import FunctionSpace
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.preconditioners.hypre_ams import chop
from firedrake.preconditioners.facet_split import restrict
from firedrake.parameters import parameters
from firedrake_citations import Citations
from firedrake.interpolation import Interpolator
from ufl.algorithms.ad import expand_derivatives
import firedrake.dmhooks as dmhooks
import firedrake.utils as utils
import ufl
import finat.ufl


__all__ = ("TwoLevelPC", "HiptmairPC")


class TwoLevelPC(PCBase):
    """ PC for two-level methods

    should implement:
    - :meth:`coarsen`
    """
    @abc.abstractmethod
    def coarsen(self, pc):
        """Return a tuple with coarse bilinear form, coarse
           boundary conditions, and coarse-to-fine interpolation matrix
        """
        raise NotImplementedError

    def initialize(self, pc):
        from firedrake.assemble import get_assembler
        A, P = pc.getOperators()
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()

        coarse_operator, coarse_space_bcs, interp_petscmat = self.coarsen(pc)

        # Handle the coarse operator
        coarse_options_prefix = options_prefix + "mg_coarse_"
        coarse_mat_type = opts.getString(coarse_options_prefix + "mat_type",
                                         parameters["default_matrix_type"])
        form_assembler = get_assembler(coarse_operator, bcs=coarse_space_bcs, form_compiler_parameters=fcp, mat_type=coarse_mat_type, options_prefix=coarse_options_prefix)
        self.coarse_op = form_assembler.allocate()
        self._assemble_coarse_op = form_assembler.assemble
        self._assemble_coarse_op(tensor=self.coarse_op)
        coarse_opmat = self.coarse_op.petscmat

        # We set up a PCMG object that uses the constructed interpolation
        # matrix to generate the restriction/prolongation operators.
        # This is a two-level multigrid preconditioner.
        pcmg = PETSc.PC().create(comm=pc.comm)
        pcmg.incrementTabLevel(1, parent=pc)

        pcmg.setType(pc.Type.MG)
        pcmg.setOptionsPrefix(options_prefix)
        pcmg.setMGLevels(2)
        pcmg.setMGType(pc.MGType.ADDITIVE)
        pcmg.setMGCycleType(pc.MGCycleType.V)
        pcmg.setMGInterpolation(1, interp_petscmat)
        # FIXME the default for MGRScale is created with the wrong shape when dim(coarse) > dim(fine)
        # FIXME there is no need for injection in a KSP context, probably this comes from the snes_ctx below
        # as workaround define injection as the restriction of the solution times a zero vector
        pcmg.setMGRScale(1, interp_petscmat.createVecRight())
        pcmg.setOperators(A=A, P=P)

        coarse_solver = pcmg.getMGCoarseSolve()
        coarse_solver.setOperators(A=coarse_opmat, P=coarse_opmat)
        # coarse space dm
        coarse_space = coarse_operator.arguments()[-1].function_space()
        coarse_dm = coarse_space.dm
        coarse_solver.setDM(coarse_dm)
        coarse_solver.setDMActive(False)
        pcmg.setDM(pc.getDM())
        pcmg.setFromOptions()
        self.pc = pcmg
        self._dm = coarse_dm

        prefix = coarse_solver.getOptionsPrefix()
        # Create new appctx
        self._ctx_ref = self.new_snes_ctx(pc,
                                          coarse_operator,
                                          coarse_space_bcs,
                                          coarse_mat_type,
                                          fcp,
                                          options_prefix=prefix)

        with dmhooks.add_hooks(coarse_dm, self, appctx=self._ctx_ref, save=False):
            coarse_solver.setFromOptions()

    def update(self, pc):
        self._assemble_coarse_op(tensor=self.coarse_op)
        self.pc.setUp()

    def apply(self, pc, X, Y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.apply(X, Y)

    def applyTranspose(self, pc, X, Y):
        dm = self._dm
        with dmhooks.add_hooks(dm, self, appctx=self._ctx_ref):
            self.pc.applyTranspose(X, Y)

    def view(self, pc, viewer=None):
        super(TwoLevelPC, self).view(pc, viewer)
        if hasattr(self, "pc"):
            viewer.printfASCII("Two level PC\n")
            self.pc.view(viewer)


class HiptmairPC(TwoLevelPC):
    """A two-level method for H(curl) or H(div) problems with an auxiliary
    potential space in H^1 or H(curl), respectively.

    Internally this creates a PETSc PCMG object that can be controlled by
    options using the extra options prefix ``hiptmair_mg_``.

    This allows for effective multigrid relaxation methods with patch solves
    centered around vertices for H^1, edges for H(curl), or faces for H(div).
    For the lowest-order spaces this corresponds to point-Jacobi.

    The H(div) auxiliary vector potential problem in H(curl) is singular for
    high-order.  This can be overcome by pertubing the problem by a multiple of
    the mass matrix. The scaling factor can be provided (defaulting to 0) by
    providing a scalar in the application context, keyed on
    ``"hiptmair_shift"``.
    """

    _prefix = "hiptmair_"

    def coarsen(self, pc):
        Citations().register("Hiptmair1998")
        appctx = self.get_appctx(pc)

        a, bcs = self.form(pc)
        V = a.arguments()[-1].function_space()
        mesh = V.mesh()
        element = V.ufl_element()
        formdegree = V.finat_element.formdegree
        if formdegree == 1:
            celement = curl_to_grad(element)
        elif formdegree == 2:
            celement = div_to_curl(element)
        else:
            raise ValueError("Hiptmair decomposition not available for", element)

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options(options_prefix)
        domain = opts.getString("mg_coarse_restriction_domain", "")
        if domain:
            celement = restrict(celement, domain)

        coarse_space = FunctionSpace(mesh, celement)
        assert coarse_space.finat_element.formdegree + 1 == formdegree
        coarse_space_bcs = [bc.reconstruct(V=coarse_space, g=0) for bc in bcs]

        if element.sobolev_space == ufl.HDiv:
            G_callback = appctx.get("get_curl", None)
            dminus = ufl.curl
            if V.shape:
                dminus = lambda u: ufl.as_vector([ufl.curl(u[k, ...])
                                                  for k in range(u.ufl_shape[0])])
        else:
            G_callback = appctx.get("get_gradient", None)
            dminus = ufl.grad

        # Get only the zero-th order term of the form
        replace_dict = {ufl.grad(t): ufl.zero(ufl.grad(t).ufl_shape) for t in a.arguments()}
        beta = ufl.replace(expand_derivatives(a), replace_dict)

        test = TestFunction(coarse_space)
        trial = TrialFunction(coarse_space)
        coarse_operator = beta(dminus(test), dminus(trial))

        zero_beta = opts.getBool("zero_beta_poisson", True)
        if zero_beta:
            from firedrake.assemble import assemble
            # Remove coarse nodes where beta is zero
            coarse_diagonal = assemble(coarse_operator, diagonal=True)
            diag = numpy.abs(coarse_diagonal.dat.data_ro_with_halos)
            atol = numpy.max(diag) * 1E-10
            bc_nodes = numpy.flatnonzero(diag <= atol).astype(PETSc.IntType)
            coarse_space_bcs.append(BCFromNodes(coarse_space, 0, bc_nodes))

        cdegree = max(as_tuple(celement.degree()))
        if formdegree > 1 and cdegree > 1:
            shift = appctx.get("hiptmair_shift", None)
            if shift is not None:
                b = beta(test, shift * trial)
                coarse_operator += ufl.Form(b.integrals_by_type("cell"))

        coarse_space_bcs = tuple(coarse_space_bcs)
        if G_callback is None:
            interp_petscmat = chop(Interpolator(dminus(test), V, bcs=bcs + coarse_space_bcs).callable().handle)
        else:
            interp_petscmat = G_callback(coarse_space, V, coarse_space_bcs, bcs)

        return coarse_operator, coarse_space_bcs, interp_petscmat


def curl_to_grad(ele):
    if isinstance(ele, finat.ufl.VectorElement):
        return type(ele)(curl_to_grad(ele._sub_element), dim=ele.num_sub_elements)
    elif isinstance(ele, finat.ufl.TensorElement):
        return type(ele)(curl_to_grad(ele._sub_element), shape=ele._shape, symmetry=ele.symmetry())
    elif isinstance(ele, finat.ufl.MixedElement):
        return type(ele)(*(curl_to_grad(e) for e in ele.sub_elements))
    elif isinstance(ele, finat.ufl.RestrictedElement):
        return finat.ufl.RestrictedElement(curl_to_grad(ele._element), ele.restriction_domain())
    else:
        cell = ele.cell
        family = ele.family()
        variant = ele.variant()
        degree = ele.degree()
        if family.startswith("Sminus"):
            family = "S"
        else:
            if family in ["Nedelec 2nd kind H(curl)", "Brezzi-Douglas-Marini"]:
                degree = degree + 1
            family = "CG"
            if isinstance(degree, tuple) and isinstance(cell, ufl.TensorProductCell):
                cells = ele.cell.sub_cells()
                elems = [finat.ufl.FiniteElement(family, cell=c, degree=d, variant=variant) for c, d in zip(cells, degree)]
                return finat.ufl.TensorProductElement(*elems, cell=cell)
        return finat.ufl.FiniteElement(family, cell=cell, degree=degree, variant=variant)


def div_to_curl(ele):
    if isinstance(ele, finat.ufl.VectorElement):
        return type(ele)(div_to_curl(ele._sub_element), dim=ele.num_sub_elements)
    elif isinstance(ele, finat.ufl.TensorElement):
        return type(ele)(div_to_curl(ele._sub_element), shape=ele._shape, symmetry=ele.symmetry())
    elif isinstance(ele, finat.ufl.MixedElement):
        return type(ele)(*(div_to_curl(e) for e in ele.sub_elements))
    elif isinstance(ele, finat.ufl.RestrictedElement):
        return finat.ufl.RestrictedElement(div_to_curl(ele._element), ele.restriction_domain())
    elif isinstance(ele, finat.ufl.EnrichedElement):
        return type(ele)(*(div_to_curl(e) for e in reversed(ele._elements)))
    elif isinstance(ele, finat.ufl.TensorProductElement):
        return type(ele)(*(div_to_curl(e) for e in ele.sub_elements), cell=ele.cell)
    elif isinstance(ele, finat.ufl.WithMapping):
        return type(ele)(div_to_curl(ele.wrapee), ele.mapping())
    elif isinstance(ele, finat.ufl.BrokenElement):
        return type(ele)(div_to_curl(ele._element))
    elif isinstance(ele, finat.ufl.HDivElement):
        return finat.ufl.HCurlElement(div_to_curl(ele._element))
    elif isinstance(ele, finat.ufl.HCurlElement):
        raise ValueError("Expecting an H(div) element")
    else:
        degree = ele.degree()
        family = ele.family()
        if family in ["Lagrange", "CG", "Q"]:
            family = "DG" if ele.cell.is_simplex() else "DQ"
            degree = degree - 1
        elif family in ["Discontinuous Lagrange", "DG", "DQ"]:
            family = "CG"
            degree = degree + 1
        else:
            if family == "Brezzi-Douglas-Marini":
                degree = degree + 1
            replace_dict = {
                "Raviart-Thomas": "N1curl",
                "Brezzi-Douglas-Marini": "N2curl",
                "RTCF": "RTCE",
                "NCF": "NCE",
                "SminusF": "SminusE",
                "SminusDiv": "SminusCurl",
            }
            family = replace_dict.get(family, None)
            if family is None:
                raise ValueError("Unexpected family %s" % family)
        return ele.reconstruct(degree=degree, family=family)


class BCFromNodes(DirichletBC):

    def __init__(self, V, g, nodes):
        self._nodes = nodes
        super(BCFromNodes, self).__init__(V, g, tuple())

    @utils.cached_property
    def nodes(self):
        return self._nodes

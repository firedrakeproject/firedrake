import abc
import ufl
import finat.ufl
from ufl.domain import extract_unique_domain

import firedrake
from firedrake.petsc import PETSc
from firedrake.utils import cached_property, complex_mode, SLATE_SUPPORTS_COMPLEX
from firedrake import functionspaceimpl
from firedrake import function
from firedrake.adjoint_utils import annotate_project
from finat import HDivTrace


__all__ = ['project', 'Projector']


def sanitise_input(v, V):
    if isinstance(v, function.Function):
        return v
    elif isinstance(v, ufl.classes.Expr):
        return v
    else:
        raise ValueError("Can't project from source object %r" % v)


def create_output(V, name=None):
    if isinstance(V, functionspaceimpl.WithGeometry):
        return function.Function(V, name=name)
    elif isinstance(V, function.Function):
        return V
    else:
        raise ValueError("Can't project into target object %r" % V)


def check_meshes(source, target):
    source_mesh = extract_unique_domain(source)
    target_mesh = extract_unique_domain(target)
    if source_mesh is None:
        source_mesh = target_mesh
    if target_mesh is None:
        raise ValueError("Target space must have a mesh")
    if source_mesh.ufl_cell() != target_mesh.ufl_cell():
        raise ValueError("Mismatching cells in source (%r) and target (%r) meshes" %
                         (source_mesh.ufl_cell(), target_mesh.ufl_cell()))
    return source_mesh, target_mesh


@PETSc.Log.EventDecorator()
@annotate_project
def project(v, V, bcs=None,
            solver_parameters=None,
            form_compiler_parameters=None,
            use_slate_for_inverse=True,
            name=None,
            ad_block_tag=None):
    """Project a UFL expression into a :class:`.FunctionSpace`
    It is possible to project onto the trace space 'DGT', but not onto
    other trace spaces e.g. into the restriction of CG onto the facets.

    :arg v: the :class:`ufl.core.expr.Expr` to project
    :arg V: the :class:`.FunctionSpace` or :class:`.Function` to project into
    :kwarg bcs: boundary conditions to apply in the projection
    :kwarg solver_parameters: parameters to pass to the solver used when
         projecting.
    :kwarg form_compiler_parameters: parameters to the form compiler
    :kwarg use_slate_for_inverse: compute mass inverse cell-wise using
         SLATE (ignored for non-DG function spaces).
    :kwarg name: name of the resulting :class:`.Function`
    :kwarg ad_block_tag: string for tagging the resulting block on the Pyadjoint tape

    If ``V`` is a :class:`.Function` then ``v`` is projected into
    ``V`` and ``V`` is returned. If `V` is a :class:`.FunctionSpace`
    then ``v`` is projected into a new :class:`.Function` and that
    :class:`.Function` is returned."""

    val = Projector(v, V, bcs=bcs, solver_parameters=solver_parameters,
                    form_compiler_parameters=form_compiler_parameters,
                    use_slate_for_inverse=use_slate_for_inverse).project()
    val.rename(name)
    return val


class Assigner(object):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def project(self):
        self.target.assign(self.source)
        return self.target


class ProjectorBase(object, metaclass=abc.ABCMeta):
    def __init__(self, source, target, bcs=None, solver_parameters=None,
                 form_compiler_parameters=None, constant_jacobian=True,
                 use_slate_for_inverse=True):
        if solver_parameters is None:
            solver_parameters = {}
        else:
            solver_parameters = solver_parameters.copy()
        solver_parameters.setdefault("ksp_type", "cg")
        solver_parameters.setdefault("ksp_rtol", 1e-8)
        mat_type = solver_parameters.get("mat_type", firedrake.parameters["default_matrix_type"])
        if mat_type == "nest":
            solver_parameters.setdefault("pc_type", "fieldsplit")
            solver_parameters.setdefault("fieldsplit_pc_type", "bjacobi")
            solver_parameters.setdefault("fieldsplit_sub_pc_type", "icc")
        elif mat_type == "matfree":
            solver_parameters.setdefault("pc_type", "jacobi")
        else:
            solver_parameters.setdefault("pc_type", "bjacobi")
            solver_parameters.setdefault("sub_pc_type", "icc")
        self.source = source
        self.target = target
        self.solver_parameters = solver_parameters
        self.form_compiler_parameters = form_compiler_parameters
        self.bcs = bcs
        self.constant_jacobian = constant_jacobian
        try:
            element = self.target.function_space().finat_element
            is_dg = element.entity_dofs() == element.entity_closure_dofs()
            is_variable_layers = self.target.function_space().mesh().variable_layers
        except AttributeError:
            # Mixed space
            is_dg = False
            is_variable_layers = True
        self.use_slate_for_inverse = (use_slate_for_inverse and is_dg and not is_variable_layers
                                      and (not complex_mode or SLATE_SUPPORTS_COMPLEX))

    @cached_property
    def A(self):
        u = firedrake.TrialFunction(self.target.function_space())
        v = firedrake.TestFunction(self.target.function_space())
        F = self.target.function_space()
        mixed = isinstance(F.ufl_element(), finat.ufl.MixedElement)
        if not mixed and isinstance(F.finat_element, HDivTrace):
            if F.extruded:
                a = (firedrake.inner(u, v)*firedrake.ds_t
                     + firedrake.inner(u, v)*firedrake.ds_v
                     + firedrake.inner(u, v)*firedrake.ds_b
                     + firedrake.inner(u('+'), v('+'))*firedrake.dS_h
                     + firedrake.inner(u('+'), v('+'))*firedrake.dS_v)
            else:
                a = (firedrake.inner(u, v)*firedrake.ds
                     + firedrake.inner(u('+'), v('+'))*firedrake.dS)
        else:
            a = firedrake.inner(u, v)*firedrake.dx
        if self.use_slate_for_inverse:
            a = firedrake.Tensor(a).inv
        A = firedrake.assemble(a, bcs=self.bcs,
                               mat_type=self.solver_parameters.get("mat_type"),
                               form_compiler_parameters=self.form_compiler_parameters)
        return A

    @cached_property
    def solver(self):
        return firedrake.LinearSolver(self.A, solver_parameters=self.solver_parameters)

    @property
    def apply_massinv(self):
        if not self.constant_jacobian:
            firedrake.assemble(self.A.a, tensor=self.A, bcs=self.bcs,
                               form_compiler_parameters=self.form_compiler_parameters)
        if self.use_slate_for_inverse:
            def solve(x, b):
                with x.dat.vec_wo as x_, b.dat.vec_ro as b_:
                    self.A.petscmat.mult(b_, x_)
            return solve
        else:
            return self.solver.solve

    @cached_property
    def residual(self):
        return firedrake.Cofunction(self.target.function_space().dual())

    @abc.abstractproperty
    def rhs(self):
        pass

    def project(self):
        self.apply_massinv(self.target, self.rhs)
        return self.target


class BasicProjector(ProjectorBase):
    """
    A basic projector projects a UFL expression into a function space
    and places the result in a function from that function space,
    allowing the solver to be reused. The difference to the
    :class:`.SupermeshProjector` is that both function spaces are
    defined on the same mesh.
    """

    @cached_property
    def rhs_form(self):
        v = firedrake.TestFunction(self.target.function_space())
        F = self.target.function_space()
        mixed = isinstance(F.ufl_element(), finat.ufl.MixedElement)
        if not mixed and isinstance(F.finat_element, HDivTrace):
            # Project onto a trace space by supplying the respective form on the facets.
            # The measures on the facets differ between extruded and non-extruded mesh.
            # FIXME The restrictions of cg onto the facets is also a trace space,
            # but we only cover DGT.
            if F.extruded:
                form = (firedrake.inner(self.source, v)*firedrake.ds_t
                        + firedrake.inner(self.source, v)*firedrake.ds_v
                        + firedrake.inner(self.source, v)*firedrake.ds_b
                        + firedrake.inner(firedrake.avg(self.source), firedrake.avg(v))*firedrake.dS_h
                        + firedrake.inner(firedrake.avg(self.source), firedrake.avg(v))*firedrake.dS_v)
            else:
                form = (firedrake.inner(self.source, v)*firedrake.ds
                        + firedrake.inner(firedrake.avg(self.source), firedrake.avg(v))*firedrake.dS)

        else:
            form = firedrake.inner(self.source, v)*firedrake.dx
        return form

    @cached_property
    def assembler(self):
        from firedrake.assemble import get_assembler
        return get_assembler(self.rhs_form,
                             form_compiler_parameters=self.form_compiler_parameters).assemble

    @property
    def rhs(self):
        self.assembler(tensor=self.residual)
        return self.residual


class SupermeshProjector(ProjectorBase):
    @cached_property
    def mixed_mass(self):
        from firedrake.supermeshing import assemble_mixed_mass_matrix
        return assemble_mixed_mass_matrix(self.source.function_space(),
                                          self.target.function_space())

    @property
    def rhs(self):
        with self.source.dat.vec_ro as u, self.residual.dat.vec_wo as v:
            self.mixed_mass.mult(u, v)
        return self.residual


@PETSc.Log.EventDecorator()
def Projector(v, v_out, bcs=None, solver_parameters=None,
              form_compiler_parameters=None, constant_jacobian=True,
              use_slate_for_inverse=False):
    """
    A projector projects a UFL expression into a function space
    and places the result in a function from that function space,
    allowing the solver to be reused. Projection reverts to an assign
    operation if ``v`` is a :class:`.Function` and belongs to the same
    function space as ``v_out``.
    It is possible to project onto the trace space 'DGT', but not onto
    other trace spaces e.g. into the restriction of CG onto the facets.

    :arg v: the :class:`ufl.core.expr.Expr` or
         :class:`.Function` to project
    :arg V: :class:`.Function` (or :class:`~.FunctionSpace`) to put the result in.
    :arg bcs: an optional set of :class:`.DirichletBC` objects to apply
              on the target function space.
    :arg solver_parameters: parameters to pass to the solver used when
         projecting.
    :arg constant_jacobian: Is the projection matrix constant between calls?
        Say False if you have moving meshes.
    :arg use_slate_for_inverse: compute mass inverse cell-wise using
         SLATE (only valid for DG function spaces).
    """
    target = create_output(v_out)
    source = sanitise_input(v, target.function_space())
    source_mesh, target_mesh = check_meshes(source, target)
    if source.ufl_shape != target.ufl_shape:
        raise ValueError("Shape mismatch between source %s and target %s in project" %
                         (source.ufl_shape, target.ufl_shape))
    if isinstance(v, function.Function) and not bcs and v.function_space() == target.function_space():
        return Assigner(source, target)
    elif source_mesh == target_mesh:
        return BasicProjector(source, target, bcs=bcs, solver_parameters=solver_parameters,
                              form_compiler_parameters=form_compiler_parameters,
                              constant_jacobian=constant_jacobian,
                              use_slate_for_inverse=use_slate_for_inverse)
    else:
        if bcs is not None:
            raise ValueError("Haven't implemented supermesh projection with boundary conditions yet, sorry!")
        if not isinstance(source, function.Function) or source.ufl_element().family() == "Real":
            raise NotImplementedError("Only for source Functions, not %s" % type(source))
        return SupermeshProjector(source, target, bcs=bcs, solver_parameters=solver_parameters,
                                  form_compiler_parameters=form_compiler_parameters,
                                  constant_jacobian=constant_jacobian,
                                  use_slate_for_inverse=use_slate_for_inverse)

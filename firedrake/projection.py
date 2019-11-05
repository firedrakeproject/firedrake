import abc
import ufl

import firedrake
from firedrake.utils import cached_property
from firedrake import expression
from firedrake import functionspace
from firedrake import functionspaceimpl
from firedrake import function
from pyop2.utils import as_tuple


__all__ = ['project', 'Projector']


def sanitise_input(v, V):
    if isinstance(v, expression.Expression):
        shape = v.value_shape()
        # Build a function space that supports PointEvaluation so that
        # we can interpolate into it.
        deg = max(as_tuple(V.ufl_element().degree()))

        if v.rank() == 0:
            fs = functionspace.FunctionSpace(V.mesh(), 'DG', deg+1)
        elif v.rank() == 1:
            fs = functionspace.VectorFunctionSpace(V.mesh(), 'DG',
                                                   deg+1,
                                                   dim=shape[0])
        else:
            fs = functionspace.TensorFunctionSpace(V.mesh(), 'DG',
                                                   deg+1,
                                                   shape=shape)
        f = function.Function(fs)
        f.interpolate(v)
        return f
    elif isinstance(v, function.Function):
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
    source_mesh = source.ufl_domain()
    target_mesh = target.ufl_domain()
    if source_mesh is None:
        source_mesh = target_mesh
    if target_mesh is None:
        raise ValueError("Target space must have a mesh")
    if source_mesh.ufl_cell() != target_mesh.ufl_cell():
        raise ValueError("Mismatching cells in source (%r) and target (%r) meshes" %
                         (source_mesh.ufl_cell(), target_mesh.ufl_cell()))
    return source_mesh, target_mesh


def project(v, V, bcs=None,
            solver_parameters=None,
            form_compiler_parameters=None,
            use_slate_for_inverse=True,
            name=None):
    """Project an :class:`.Expression` or :class:`.Function` into a :class:`.FunctionSpace`

    :arg v: the :class:`.Expression`, :class:`ufl.Expr` or
         :class:`.Function` to project
    :arg V: the :class:`.FunctionSpace` or :class:`.Function` to project into
    :arg bcs: boundary conditions to apply in the projection
    :arg solver_parameters: parameters to pass to the solver used when
         projecting.
    :arg form_compiler_parameters: parameters to the form compiler
    :arg use_slate_for_inverse: compute mass inverse cell-wise using
         SLATE (ignored for non-DG function spaces).
    :arg name: name of the resulting :class:`.Function`

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
        self.use_slate_for_inverse = use_slate_for_inverse and is_dg and not is_variable_layers

    @cached_property
    def A(self):
        u = firedrake.TrialFunction(self.target.function_space())
        v = firedrake.TestFunction(self.target.function_space())
        a = firedrake.inner(u, v)*firedrake.dx
        if self.use_slate_for_inverse:
            a = firedrake.Tensor(a).inv
        A = firedrake.assemble(a, bcs=self.bcs,
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
        return firedrake.Function(self.target.function_space())

    @abc.abstractproperty
    def rhs(self):
        pass

    def project(self):
        self.apply_massinv(self.target, self.rhs)
        return self.target


class BasicProjector(ProjectorBase):

    @cached_property
    def rhs_form(self):
        v = firedrake.TestFunction(self.target.function_space())
        form = firedrake.inner(self.source, v)*firedrake.dx
        return form

    @cached_property
    def assembler(self):
        from firedrake.assemble import create_assembly_callable
        return create_assembly_callable(self.rhs_form, tensor=self.residual,
                                        form_compiler_parameters=self.form_compiler_parameters)

    @property
    def rhs(self):
        self.assembler()
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


def Projector(v, v_out, bcs=None, solver_parameters=None,
              form_compiler_parameters=None, constant_jacobian=True,
              use_slate_for_inverse=False):
    """
    A projector projects a UFL expression into a function space
    and places the result in a function from that function space,
    allowing the solver to be reused. Projection reverts to an assign
    operation if ``v`` is a :class:`.Function` and belongs to the same
    function space as ``v_out``.

    :arg v: the :class:`ufl.Expr` or
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
        if not isinstance(source, function.Function):
            raise NotImplementedError("Only for source Functions, not %s" % type(source))
        return SupermeshProjector(source, target, bcs=bcs, solver_parameters=solver_parameters,
                                  form_compiler_parameters=form_compiler_parameters,
                                  constant_jacobian=constant_jacobian,
                                  use_slate_for_inverse=use_slate_for_inverse)

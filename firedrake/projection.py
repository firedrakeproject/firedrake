import ufl

import firedrake
from firedrake import expression
from firedrake import functionspace
from firedrake import functionspaceimpl
from firedrake import function
import firedrake.variational_solver as vs
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
    if source_mesh.ufl_cell() != target_mesh.ufl_cell():
        raise ValueError("Mismatching cells in source (%r) and target (%r) meshes" %
                         (source_mesh.ufl_cell(), target_mesh.ufl_cell()))
    if source_mesh != target_mesh:
        raise ValueError("Can't projecting between non-matching meshes yet")
    return source_mesh, target_mesh


def project(v, V, bcs=None,
            solver_parameters=None,
            form_compiler_parameters=None,
            name=None):
    """Project an :class:`.Expression` or :class:`.Function` into a :class:`.FunctionSpace`

    :arg v: the :class:`.Expression`, :class:`ufl.Expr` or
         :class:`.Function` to project
    :arg V: the :class:`.FunctionSpace` or :class:`.Function` to project into
    :arg bcs: boundary conditions to apply in the projection
    :arg solver_parameters: parameters to pass to the solver used when
         projecting.
    :arg form_compiler_parameters: parameters to the form compiler
    :arg name: name of the resulting :class:`.Function`

    If ``V`` is a :class:`.Function` then ``v`` is projected into
    ``V`` and ``V`` is returned. If `V` is a :class:`.FunctionSpace`
    then ``v`` is projected into a new :class:`.Function` and that
    :class:`.Function` is returned."""

    val = Projector(v, V, bcs=bcs, solver_parameters=solver_parameters,
                    form_compiler_parameters=form_compiler_parameters).project()
    val.rename(name)
    return val


class Projector(object):
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
    """

    def __init__(self, v, v_out, bcs=None, solver_parameters=None,
                 form_compiler_parameters=None, constant_jacobian=True):
        target = create_output(v_out)
        source = sanitise_input(v, target.function_space())
        source_mesh, target_mesh = check_meshes(source, target)
        if source.ufl_shape != target.ufl_shape:
            raise ValueError("Shape mismatch between source %s and target %s in project" %
                             (source.ufl_shape, target.ufl_shape))

        self._use_assign = (isinstance(v, function.Function)
                            and not self.bcs
                            and v.function_space() == v_out.function_space())
        self.source = source
        self.target = target
        self.bcs = bcs

        if not self._use_assign:
            V = target.function_space()

            p = firedrake.TestFunction(V)
            q = firedrake.TrialFunction(V)

            a = ufl.inner(p, q)*ufl.dx
            L = ufl.inner(p, source)*ufl.dx

            problem = vs.LinearVariationalProblem(a, L, target, bcs=self.bcs,
                                                  form_compiler_parameters=form_compiler_parameters,
                                                  constant_jacobian=constant_jacobian)

            if solver_parameters is None:
                solver_parameters = {}
            else:
                solver_parameters = solver_parameters.copy()
            solver_parameters.setdefault("ksp_type", "cg")

            self.solver = vs.LinearVariationalSolver(problem,
                                                     solver_parameters=solver_parameters)

    def project(self):
        """
        Apply the projection.
        """
        if self._use_assign:
            self.target.assign(self.source)
        else:
            self.solver.solve()
        return self.target

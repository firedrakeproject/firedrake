from functionspace import FunctionSpaceBase, FunctionSpace, VectorFunctionSpace
import function
from expression import Expression
from solving import solve
import ufl
from ufl_expr import TrialFunction, TestFunction


__all__ = ['project']


def project(v, V, bcs=None, mesh=None,
            solver_parameters=None,
            form_compiler_parameters=None,
            name=None):
    """Project an :class:`.Expression` or :class:`.Function` into a :class:`.FunctionSpace`

    :arg v: the :class:`.Expression` or :class:`.Function` to project
    :arg V: the :class:`.FunctionSpace` or :class:`.Function` to project into
    :arg bcs: boundary conditions to apply in the projection
    :arg mesh: the mesh to project into
    :arg solver_type: linear solver to use
    :arg preconditioner_type: preconditioner to use
    :arg form_compiler_parameters: parameters to the form compiler
    :arg name: name of the resulting :class:`.Function`

    If ``V`` is a :class:`.Function` then ``v`` is projected into
    ``V`` and ``V`` is returned. If `V` is a :class:`.FunctionSpace`
    then ``v`` is projected into a new :class:`.Function` and that
    :class:`.Function` is returned.

    Currently, `bcs`, `mesh` and `form_compiler_parameters` are ignored."""
    if isinstance(V, FunctionSpaceBase):
        ret = function.Function(V, name=name)
    elif isinstance(V, function.Function):
        ret = V
        V = V.function_space()
    else:
        raise RuntimeError(
            'Can only project into functions and function spaces, not %r'
            % type(V))

    if isinstance(v, Expression):
        shape = v.shape()
        # Build a function space that supports PointEvaluation so that
        # we can interpolate into it.
        if isinstance(V.ufl_element().degree(), tuple):
            deg = max(V.ufl_element().degree())
        else:
            deg = V.ufl_element().degree()

        if v.rank() == 0:
            fs = FunctionSpace(V.mesh(), 'DG', deg+1)
        elif v.rank() == 1:
            fs = VectorFunctionSpace(V.mesh(), 'DG',
                                     deg+1,
                                     dim=shape[0])
        else:
            raise NotImplementedError(
                "Don't know how to project onto tensor-valued function spaces")
        f = function.Function(fs)
        f.interpolate(v)
        v = f

    if v.shape() != ret.shape():
        raise RuntimeError('Shape mismatch between source %s and target function spaces %s in project' % (v.shape(), ret.shape()))

    if v.function_space().mesh() != ret.function_space().mesh():
        raise RuntimeError("Can't project between mismatching meshes")

    p = TestFunction(V)
    q = TrialFunction(V)
    a = ufl.inner(p, q) * V.mesh()._dx
    L = ufl.inner(p, v) * V.mesh()._dx

    # Default to 1e-8 relative tolerance
    if solver_parameters is None:
        solver_parameters = {'ksp_type': 'cg', 'ksp_rtol': 1e-8}
    else:
        solver_parameters.setdefault('ksp_type', 'cg')
        solver_parameters.setdefault('ksp_rtol', 1e-8)

    solve(a == L, ret, bcs=bcs,
          solver_parameters=solver_parameters,
          form_compiler_parameters=form_compiler_parameters)
    return ret


class Compat:
    # A bag class to act as a namespace for compat.
    pass


def compat(backend):
    compat = Compat()

    if backend.__name__ == "firedrake":

        compat.FunctionSpaceType = (backend.functionspaceimpl.FunctionSpace,
                                    backend.functionspaceimpl.WithGeometry,
                                    backend.functionspaceimpl.MixedFunctionSpace)
        compat.FunctionSpace = backend.FunctionSpace

        compat.MeshType = backend.mesh.MeshGeometry

        def extract_subfunction(u, V):
            """If V is a subspace of the function-space of u, return the component of u that is in that subspace."""
            if V.index is not None:
                # V is an indexed subspace of a MixedFunctionSpace
                return u.sub(V.index)
            elif V.component is not None:
                # V is a vector component subspace.
                # The vector functionspace V.parent may itself be a subspace
                # so call this function recursively
                return extract_subfunction(u, V.parent).sub(V.component)
            else:
                return u
        compat.extract_subfunction = extract_subfunction

        def create_bc(bc, value=None, homogenize=None):
            """Create a new bc object from an existing one.

            :arg bc: The :class:`~.DirichletBC` to clone.
            :arg value: A new value to use.
            :arg homogenize: If True, return a homogeneous version of the bc.

            One cannot provide both ``value`` and ``homogenize``, but
            should provide at least one.
            """
            if value is None and homogenize is None:
                raise ValueError("No point cloning a bc if you're not changing its values")
            if value is not None and homogenize is not None:
                raise ValueError("Cannot provide both value and homogenize")
            if homogenize:
                value = 0
            return bc.reconstruct(g=value)
        compat.create_bc = create_bc

        # Most of this is to deal with Firedrake assembly returning
        # Function whereas Dolfin returns Vector.
        def function_from_vector(V, vector, cls=backend.Function):
            """Create a new Function sharing data.

            :arg V: The function space
            :arg vector: The data to share.
            """
            return cls(V, val=vector)
        compat.function_from_vector = function_from_vector

        def inner(a, b):
            """Compute the l2 inner product of a and b.

            :arg a: a Function.
            :arg b: a Vector.
            """
            return a.vector().inner(b)
        compat.inner = inner

        def extract_bc_subvector(value, Vtarget, bc):
            """Extract from value (a function in a mixed space), the sub
            function corresponding to the part of the space bc applies
            to.  Vtarget is the target (collapsed) space."""
            r = value
            for idx in bc._indices:
                r = r.sub(idx)
            assert Vtarget == r.function_space()
            return r.vector()
        compat.extract_bc_subvector = extract_bc_subvector

        def extract_mesh_from_form(form):
            """Takes in a form and extracts a mesh which can be used to construct function spaces.

            Dolfin only accepts dolfin.cpp.mesh.Mesh types for function spaces, while firedrake use ufl.Mesh.

            Args:
                form (ufl.Form): Form to extract mesh from

            Returns:
                ufl.Mesh: The extracted mesh

            """
            return form.ufl_domain()
        compat.extract_mesh_from_form = extract_mesh_from_form

        def constant_function_firedrake_compat(value):
            """Takes a Function/vector and returns the array.

            The Function should belong to the space of Reals.
            This function is needed because Firedrake does not
            accept a Function as argument to Constant constructor.
            It does accept vector (which is what we work with in dolfin),
            but since we work with Functions instead of vectors in firedrake,
            this function call is needed in firedrake_adjoint.

            Args:
                value (Function): A Function to convert

            Returns:
                numpy.ndarray: A numpy array of the function values.

            """
            return value.dat.data
        compat.constant_function_firedrake_compat = constant_function_firedrake_compat

        def assemble_adjoint_value(*args, **kwargs):
            """A wrapper around Firedrake's assemble that returns a Vector
            instead of a Function when assembling a 1-form."""
            result = backend.assemble(*args, **kwargs)
            if isinstance(result, backend.Function):
                return result.vector()
            else:
                return result
        compat.assemble_adjoint_value = assemble_adjoint_value

        def gather(vec):
            return vec.gather()
        compat.gather = gather

        compat.linalg_solve = backend.solve

        class Expression(object):
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("Firedrake does not have Expression objects")

        compat.ExpressionType = Expression

        compat.Expression = Expression

        def type_cast_function(obj, cls):
            """Type casts Function object `obj` to an instance of `cls`.

            Useful when converting backend.Function to overloaded Function.
            """
            return function_from_vector(obj.function_space(), obj.vector(), cls=cls)
        compat.type_cast_function = type_cast_function

        def create_constant(*args, **kwargs):
            """Initialises a firedrake.Constant object and returns it."""
            from firedrake import Constant
            return Constant(*args, **kwargs)
        compat.create_constant = create_constant

        def create_function(*args, **kwargs):
            """Initialises a firedrake.Function object and returns it."""
            from firedrake import Function
            return Function(*args, **kwargs)
        compat.create_function = create_function

        def isconstant(expr):
            """Check whether expression is constant type.
            In firedrake this is a function in the real space
            Ie: `firedrake.Function(FunctionSpace(mesh, "R"))`"""
            if isinstance(expr, backend.Constant):
                raise ValueError("Firedrake Constant requires a domain to work with pyadjoint")
            return isinstance(expr, backend.Function) and expr.ufl_element().family() == "Real"
        compat.isconstant = isconstant

    else:
        compat.Expression = backend.Expression
        compat.MatrixType = (backend.cpp.la.Matrix, backend.cpp.la.GenericMatrix)
        compat.VectorType = backend.cpp.la.GenericVector
        compat.FunctionType = backend.cpp.function.Function
        compat.FunctionSpace = backend.FunctionSpace
        compat.FunctionSpaceType = backend.cpp.function.FunctionSpace
        compat.ExpressionType = backend.function.expression.BaseExpression

        compat.MeshType = backend.Mesh

        compat.backend_fs_sub = backend.FunctionSpace.sub

        def _fs_sub(self, i):
            V = compat.backend_fs_sub(self, i)
            V._ad_parent_space = self
            return V
        backend.FunctionSpace.sub = _fs_sub

        compat.backend_fs_collapse = backend.FunctionSpace.collapse

        def _fs_collapse(self, collapsed_dofs=False):
            """Overloaded FunctionSpace.collapse to limit the amount of MPI communicator created.
            """
            if not hasattr(self, "_ad_collapsed_space"):
                # Create collapsed space
                self._ad_collapsed_space = compat.backend_fs_collapse(self, collapsed_dofs=True)

            if collapsed_dofs:
                return self._ad_collapsed_space
            else:
                return self._ad_collapsed_space[0]
        compat.FunctionSpace.collapse = _fs_collapse

        def extract_subfunction(u, V):
            component = V.component()
            r = u
            for idx in component:
                r = r.sub(int(idx))
            return r
        compat.extract_subfunction = extract_subfunction

        def create_bc(bc, value=None, homogenize=None):
            """Create a new bc object from an existing one.

            :arg bc: The :class:`~.DirichletBC` to clone.
            :arg value: A new value to use.
            :arg homogenize: If True, return a homogeneous version of the bc.

            One cannot provide both ``value`` and ``homogenize``, but
            should provide at least one.
            """
            if value is None and homogenize is None:
                raise ValueError("No point cloning a bc if you're not changing its values")
            if value is not None and homogenize is not None:
                raise ValueError("Cannot provide both value and homogenize")
            if homogenize:
                bc = backend.DirichletBC(bc)
                bc.homogenize()
                return bc
            try:
                # FIXME: Not perfect handling of Initialization, wait for development in dolfin.DirihcletBC
                bc = backend.DirichletBC(backend.FunctionSpace(bc.function_space()),
                                         value, *bc.domain_args)
            except AttributeError:
                bc = backend.DirichletBC(backend.FunctionSpace(bc.function_space()),
                                         value,
                                         bc.sub_domain, method=bc.method())
            return bc
        compat.create_bc = create_bc

        def function_from_vector(V, vector, cls=backend.Function):
            """Create a new Function from a vector.

            :arg V: The function space
            :arg vector: The vector data.
            """
            if isinstance(vector, backend.cpp.la.PETScVector)\
               or isinstance(vector, backend.cpp.la.Vector):
                pass
            elif not isinstance(vector, backend.Vector):
                # If vector is a fenics_adjoint.Function, which does not inherit
                # backend.cpp.function.Function with pybind11
                vector = vector._cpp_object
            r = cls(V)
            r.vector()[:] = vector
            return r
        compat.function_from_vector = function_from_vector

        def inner(a, b):
            """Compute the l2 inner product of a and b.

            :arg a: a Vector.
            :arg b: a Vector.
            """
            return a.inner(b)
        compat.inner = inner

        def extract_bc_subvector(value, Vtarget, bc):
            """Extract from value (a function in a mixed space), the sub
            function corresponding to the part of the space bc applies
            to.  Vtarget is the target (collapsed) space."""
            assigner = backend.FunctionAssigner(Vtarget, backend.FunctionSpace(bc.function_space()))
            output = backend.Function(Vtarget)
            assigner.assign(output, extract_subfunction(value, backend.FunctionSpace(bc.function_space())))
            return output.vector()
        compat.extract_bc_subvector = extract_bc_subvector

        def extract_mesh_from_form(form):
            """Takes in a form and extracts a mesh which can be used to construct function spaces.

            Dolfin only accepts dolfin.cpp.mesh.Mesh types for function spaces, while firedrake use ufl.Mesh.

            Args:
                form (ufl.Form): Form to extract mesh from

            Returns:
                dolfin.Mesh: The extracted mesh

            """
            return form.ufl_domain().ufl_cargo()
        compat.extract_mesh_from_form = extract_mesh_from_form

        def constant_function_firedrake_compat(value):
            """Only needed on firedrake side.

            See docstring for the firedrake version of this function above.
            """
            return value
        compat.constant_function_firedrake_compat = constant_function_firedrake_compat

        def assemble_adjoint_value(*args, **kwargs):
            """Wrapper that assembles a matrix with boundary conditions"""
            bcs = kwargs.pop("bcs", ())
            result = backend.assemble(*args, **kwargs)
            for bc in bcs:
                bc.apply(result)
            return result
        compat.assemble_adjoint_value = assemble_adjoint_value

        def gather(vec):
            import numpy
            if isinstance(vec, backend.cpp.function.Function):
                vec = vec.vector()

            if isinstance(vec, backend.cpp.la.GenericVector):
                arr = vec.gather(numpy.arange(vec.size(), dtype='I'))
            elif isinstance(vec, list):
                return list(map(gather, vec))
            else:
                arr = vec  # Assume it's a gathered numpy array already

            return arr
        compat.gather = gather

        def linalg_solve(A, x, b, *args, **kwargs):
            """Linear system solve that has a firedrake compatible interface.

            Throws away kwargs and uses b.vector() as RHS if
            b is not a GenericVector instance.

            """
            if not isinstance(b, backend.GenericVector):
                b = b.vector()
            return backend.solve(A, x, b, *args)
        compat.linalg_solve = linalg_solve

        def type_cast_function(obj, cls):
            """Type casts Function object `obj` to an instance of `cls`.

            Useful when converting backend.Function to overloaded Function.
            """
            return cls(obj.function_space(), obj._cpp_object)
        compat.type_cast_function = type_cast_function

        def create_constant(*args, **kwargs):
            """Initialise a fenics_adjoint.Constant object and return it."""
            from fenics_adjoint import Constant
            # Dolfin constants do not have domains
            _ = kwargs.pop("domain", None)
            return Constant(*args, **kwargs)
        compat.create_constant = create_constant

        def create_function(*args, **kwargs):
            """Initialises a fenics_adjoint.Function object and returns it."""
            from fenics_adjoint import Function
            return Function(*args, **kwargs)
        compat.create_function = create_function

        def isconstant(expr):
            """Check whether expression is constant type."""
            return isinstance(expr, backend.Constant)
        compat.isconstant = isconstant

    return compat

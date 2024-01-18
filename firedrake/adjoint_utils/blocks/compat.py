
class Compat:
    # A bag class to act as a namespace for compat.
    pass


def compat(backend):
    compat = Compat()

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
        return r
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

    compat.assemble_adjoint_value = backend.assemble

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


    return compat

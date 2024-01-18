
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

    def isconstant(expr):
        """Check whether expression is constant type.
        In firedrake this is a function in the real space
        Ie: `firedrake.Function(FunctionSpace(mesh, "R"))`"""
        if isinstance(expr, backend.Constant):
            raise ValueError("Firedrake Constant requires a domain to work with pyadjoint")
        return isinstance(expr, backend.Function) and expr.ufl_element().family() == "Real"
    compat.isconstant = isconstant


    return compat

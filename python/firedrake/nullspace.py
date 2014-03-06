from petsc4py import PETSc
from types import IndexedFunctionSpace


class VectorSpaceBasis(object):
    """Build a basis for a vector space.

    You can use this basis to express the null space of a singular operator.

    :arg vecs: a list of :class:`.Vector`\s or :class:`.Functions`
         spanning the space.  Note that these must be orthonormal
    :arg constant: does the null space include the constant vector?
         If you pass ``constant=True`` you should not also include the
         constant vector in the list of ``vecs`` you supply.

    .. warning::

       The vectors you pass in to this object are *not* copied.  You
       should therefore not modify them after instantiation since the
       basis will then be incorrect.
    """
    def __init__(self, vecs=None, constant=False):
        if vecs is None and not constant:
            raise RuntimeError("Must either provide a list of null space vectors, or constant keyword (or both)")
        self._vecs = vecs or []
        self._petsc_vecs = []
        for v in self._vecs:
            with v.dat.vec_ro as v_:
                self._petsc_vecs.append(v_)
        if not self.is_orthonormal():
            raise RuntimeError("Provided vectors must be orthonormal")
        self._nullspace = PETSc.NullSpace().create(constant=constant,
                                                   vectors=self._petsc_vecs)

    @property
    def nullspace(self):
        """The PETSc NullSpace object for this :class:`.VectorSpaceBasis`"""
        return self._nullspace

    def orthogonalize(self, b):
        """Orthogonalize ``b`` with respect to this :class:`.VectorSpaceBasis`.

        :arg b: a :class:`.Function`"""
        raise NotImplementedError

    def is_orthonormal(self):
        """Is this vector space basis orthonormal?"""
        for i, iv in enumerate(self._petsc_vecs):
            for j, jv in enumerate(self._petsc_vecs):
                dij = 1 if i == j else 0
                if abs(iv.dot(jv) - dij) > 1e-14:
                    return False
        return True

    def is_orthogonal(self):
        """Is this vector space basis orthogonal?"""
        for i, iv in enumerate(self._petsc_vecs):
            for j, jv in enumerate(self._petsc_vecs):
                if i == j:
                    continue
                if abs(iv.dot(jv)) > 1e-14:
                    return False
        return True

    def _apply(self, matrix):
        """Set this VectorSpaceBasis as a nullspace for a matrix

        :arg matrix: a :class:`pyop2.op2.Mat` whose nullspace should be set."""
        matrix.handle.setNullSpace(self.nullspace)

    def __iter__(self):
        """Yield self when iterated over"""
        yield self


class MixedVectorSpaceBasis(object):
    """A basis for a mixed vector space

    :arg bases: an iterable of bases for the null spaces of the
         subspaces in the mixed space.

    You can use this to express the null space of a singular operator
    on a mixed space.  The bases you supply will be used to set null
    spaces for each of the diagonal blocks in the operator.  If you
    only care about the null space on one of the blocks, you can pass
    an indexed function space as a placeholder in the positions you
    don't care about.

    For example, consider a mixed poisson discretisation with pure
    Neumann boundary conditions::

        V = FunctionSpace(mesh, "BDM", 1)
        Q = FunctionSpace(mesh, "DG", 0)

        W = V*Q

        sigma, u = TrialFunctions(W)
        tau, v = TestFunctions(W)

        a = (inner(sigma, tau) + div(sigma)*v + div(tau)*u)*dx

    The null space of this operator is a constant function in ``Q``.
    If we solve the problem with a Schur complement, we only care
    about projecting the null space out of the ``QxQ`` block.  We can
    do this like so ::

        nullspace = MixedVectorSpaceBasis([W[0], VectorSpaceBasis(constant=True)])
        solve(a == ..., nullspace=nullspace)

    """
    def __init__(self, bases):
        if not all(isinstance(basis, (VectorSpaceBasis, IndexedFunctionSpace))
                   for basis in bases):
            raise RuntimeError("MixedVectorSpaceBasis can only contain vector space bases or indexed function spaces")
        for i, basis in enumerate(bases):
            if isinstance(basis, IndexedFunctionSpace):
                if i != basis.index:
                    raise RuntimeError("FunctionSpace with index %d cannot appear at position %d" % (basis.index, i))
        self._bases = bases

    def _apply(self, matrix):
        """Set this MixedVectorSpaceBasis as a nullspace for a matrix

        :arg matrix: a :class:`pyop2.op2.Mat` whose nullspace should be set."""
        rows, cols = matrix.sparsity.shape
        if rows != cols:
            raise RuntimeError("Can only apply nullspace to square operator")
        if rows != len(self):
            raise RuntimeError("Shape of matrix (%d, %d) does not match size of nullspace %d" %
                               (rows, cols, len(self)))
        for i, basis in enumerate(self):
            if not isinstance(basis, VectorSpaceBasis):
                continue
            basis._apply(matrix[i, i])

    def __iter__(self):
        """Yield the individual bases making up this MixedVectorSpaceBasis"""
        for basis in self._bases:
            yield basis

    def __len__(self):
        """The number of bases in this MixedVectorSpaceBasis"""
        return len(self._bases)

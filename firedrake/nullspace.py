from __future__ import absolute_import
from numpy import prod

from pyop2 import op2

from firedrake import function
from firedrake.petsc import PETSc


__all__ = ['VectorSpaceBasis', 'MixedVectorSpaceBasis']


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
        self._constant = constant

    @property
    def nullspace(self):
        """The PETSc NullSpace object for this :class:`.VectorSpaceBasis`"""
        return self._nullspace

    def orthogonalize(self, b):
        """Orthogonalize ``b`` with respect to this :class:`.VectorSpaceBasis`.

        :arg b: a :class:`.Function`

        .. note::

            Modifies ``b`` in place."""
        for v in self._vecs:
            dot = b.dat.inner(v.dat)
            b.dat -= dot * v.dat
        if self._constant:
            s = -b.dat.sum() / b.function_space().dof_count
            b.dat += s

    def is_orthonormal(self):
        """Is this vector space basis orthonormal?"""
        for i, iv in enumerate(self._vecs):
            for j, jv in enumerate(self._vecs):
                dij = 1 if i == j else 0
                # scaled by size of function space
                if abs(iv.dat.inner(jv.dat) - dij) / prod(iv.function_space().dof_count) > 1e-10:
                    return False
        return True

    def is_orthogonal(self):
        """Is this vector space basis orthogonal?"""
        for i, iv in enumerate(self._vecs):
            for j, jv in enumerate(self._vecs):
                if i == j:
                    continue
                # scaled by size of function space
                if abs(iv.dat.inner(jv.dat)) / prod(iv.function_space().dof_count) > 1e-10:
                    return False
        return True

    def _apply(self, matrix, transpose=False):
        """Set this VectorSpaceBasis as a nullspace for a matrix

        :arg matrix: a :class:`pyop2.op2.Mat` whose nullspace should
             be set.
        :kwarg transpose: Should this be set as the transpose
             nullspace instead?  Used to orthogonalize the right hand
             side wrt the provided nullspace.
        """
        if not isinstance(matrix, op2.Mat):
            return
        if transpose:
            matrix.handle.setTransposeNullSpace(self.nullspace)
        else:
            matrix.handle.setNullSpace(self.nullspace)

    def __iter__(self):
        """Yield self when iterated over"""
        yield self


class MixedVectorSpaceBasis(object):
    """A basis for a mixed vector space

    :arg function_space: the :class:`~.MixedFunctionSpace` this vector
         space is a basis for.
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

        nullspace = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
        solve(a == ..., nullspace=nullspace)

    """
    def __init__(self, function_space, bases):
        self._function_space = function_space
        for basis in bases:
            if isinstance(basis, VectorSpaceBasis):
                continue
            if basis.index is not None:
                continue
            raise RuntimeError("MixedVectorSpaceBasis can only contain vector space bases or indexed function spaces")
        for i, basis in enumerate(bases):
            if isinstance(basis, VectorSpaceBasis):
                continue
            # Must be indexed function space
            if i != basis.index:
                raise RuntimeError("FunctionSpace with index %d cannot appear at position %d" % (basis.index, i))
            if basis.parent != function_space:
                raise RuntimeError("FunctionSpace with index %d does not have %s as a parent" % (basis.index, function_space))
        self._bases = bases
        self._nullspace = None

    def _build_monolithic_basis(self):
        """Build a basis for the complete mixed space.

        The monolithic basis is formed by the cartesian product of the
        bases forming each sub part.
        """
        from itertools import product
        bvecs = [[None] for _ in self]
        # Get the complete list of basis vectors for each component in
        # the mixed basis.
        for idx, basis in enumerate(self):
            if isinstance(basis, VectorSpaceBasis):
                v = []
                if basis._constant:
                    v = [function.Function(self._function_space[idx]).assign(1)]
                bvecs[idx] = basis._vecs + v

        # Basis for mixed space is cartesian product of all the basis
        # vectors we just made.
        allbvecs = [x for x in product(*bvecs)]

        vecs = [function.Function(self._function_space) for _ in allbvecs]

        # Build the functions representing the monolithic basis.
        for vidx, bvec in enumerate(allbvecs):
            for idx, b in enumerate(bvec):
                if b:
                    vecs[vidx].sub(idx).assign(b)
        for v in vecs:
            v /= v.dat.norm

        self._vecs = vecs
        self._petsc_vecs = []
        for v in self._vecs:
            with v.dat.vec_ro as v_:
                self._petsc_vecs.append(v_)
        self._nullspace = PETSc.NullSpace().create(constant=False,
                                                   vectors=self._petsc_vecs)

    def _apply_monolithic(self, matrix, transpose=False):
        """Set this class:`MixedVectorSpaceBasis` as a nullspace for a
        matrix.

        :arg matrix: a :class:`pyop2.op2.Mat` whose nullspace should
             be set.

        :kwarg transpose: Should this be set as the transpose
             nullspace instead?  Used to orthogonalize the right hand
             side wrt the provided nullspace.

        Note, this only hangs the nullspace on the Mat, you should
        normally be using :meth:`_apply` which also hangs the
        nullspace on the appropriate fieldsplit ISes for Schur
        complements."""
        if self._nullspace is None:
            self._build_monolithic_basis()
        if transpose:
            matrix.handle.setTransposeNullSpace(self._nullspace)
        else:
            matrix.handle.setNullSpace(self._nullspace)

    def _apply(self, matrix_or_ises, transpose=False):
        """Set this :class:`MixedVectorSpaceBasis` as a nullspace for a matrix

        :arg matrix_or_ises: either a :class:`pyop2.op2.Mat` to set a
             nullspace on, or else a list of PETSc ISes to compose a
             nullspace with.
        :kwarg transpose: Should this be set as the transpose
             nullspace instead?  Used to orthogonalize the right hand
             side wrt the provided nullspace.

        .. note::

           If you're using a Schur complement preconditioner you
           should both call :meth:`_apply` on the matrix, and the ises
           defining the splits.

           If transpose is ``True``, nothing happens in the IS case,
           since PETSc does not provide the ability to set anything.
        """
        if isinstance(matrix_or_ises, op2.Mat):
            matrix = matrix_or_ises
            rows, cols = matrix.sparsity.shape
            if rows != cols:
                raise RuntimeError("Can only apply nullspace to square operator")
            if rows != len(self):
                raise RuntimeError("Shape of matrix (%d, %d) does not match size of nullspace %d" %
                                   (rows, cols, len(self)))
            # Hang the expanded nullspace on the big matrix
            self._apply_monolithic(matrix, transpose=transpose)
            return
        ises = matrix_or_ises
        if transpose:
            # PETSc doesn't give us anything here
            return
        for i, basis in enumerate(self):
            if not isinstance(basis, VectorSpaceBasis):
                continue
            # Compose appropriate nullspace with IS for schur complement
            if ises is not None:
                is_ = ises[i]
                is_.compose("nullspace", basis.nullspace)

    def __iter__(self):
        """Yield the individual bases making up this MixedVectorSpaceBasis"""
        for basis in self._bases:
            yield basis

    def __len__(self):
        """The number of bases in this MixedVectorSpaceBasis"""
        return len(self._bases)

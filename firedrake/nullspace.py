import numpy

from pyop2.mpi import COMM_WORLD, internal_comm

from firedrake import function
from firedrake.logging import warning
from firedrake.matrix import MatrixBase
from firedrake.petsc import PETSc


__all__ = ['VectorSpaceBasis', 'MixedVectorSpaceBasis']


class VectorSpaceBasis(object):
    r"""Build a basis for a vector space.

    You can use this basis to express the null space of a singular operator.

    :arg vecs: a list of :class:`.Vector`\s or :class:`.Function`\s
         spanning the space.
    :arg constant: does the null space include the constant vector?
         If you pass ``constant=True`` you should not also include the
         constant vector in the list of ``vecs`` you supply.
    :arg comm: Communicator to create the nullspace on.

    .. note::

       Before using this object in a solver, you must ensure that the
       basis is orthonormal.  You can do this by calling
       :meth:`orthonormalize`, this modifies the provided vectors *in
       place*.

    .. warning::

       The vectors you pass in to this object are *not* copied.  You
       should therefore not modify them after instantiation since the
       basis will then be incorrect.
    """
    def __init__(self, vecs=None, constant=False, comm=None):
        if vecs is None and not constant:
            raise RuntimeError("Must either provide a list of null space vectors, or constant keyword (or both)")

        if vecs is None:
            self._vecs = ()
        else:
            self._vecs = tuple(vecs)
        petsc_vecs = []
        for v in self._vecs:
            with v.vec_ro as v_:
                petsc_vecs.append(v_)
        self._petsc_vecs = tuple(petsc_vecs)
        self._constant = constant
        self._ad_orthogonalized = False
        if comm:
            self.comm = comm
        elif self._vecs:
            self.comm = self._vecs[0].comm
        else:
            warning("No comm specified for VectorSpaceBasis, COMM_WORLD assumed")
            self.comm = COMM_WORLD
        self._comm = internal_comm(self.comm, self)

    @PETSc.Log.EventDecorator()
    def nullspace(self, comm=None):
        r"""The PETSc NullSpace object for this :class:`.VectorSpaceBasis`.

        :kwarg comm: DEPRECATED pass to VectorSpaceBasis.__init__()."""
        if hasattr(self, "_nullspace"):
            return self._nullspace
        if comm:
            warning("Specify comm when initialising VectorSpaceBasis, ignoring comm argument")
        self._nullspace = PETSc.NullSpace().create(constant=self._constant,
                                                   vectors=self._petsc_vecs,
                                                   comm=self._comm)
        return self._nullspace

    @PETSc.Log.EventDecorator()
    def orthonormalize(self):
        r"""Orthonormalize the basis.

        .. warning::

           This modifies the basis *in place*.

        """
        constant = self._constant
        basis = self._petsc_vecs
        for i, vec in enumerate(basis):
            alphas = []
            for vec_ in basis[:i]:
                alphas.append(vec.dot(vec_))
            for alpha, vec_ in zip(alphas, basis[:i]):
                vec.axpy(-alpha, vec_)
            if constant:
                # Subtract constant mode
                alpha = vec.sum()
                vec.array -= alpha
            vec.normalize()
        self.check_orthogonality()
        self._ad_orthogonalized = True

    @PETSc.Log.EventDecorator()
    def orthogonalize(self, b):
        r"""Orthogonalize ``b`` with respect to this :class:`.VectorSpaceBasis`.

        :arg b: a :class:`.Function`

        .. note::

            Modifies ``b`` in place."""
        nullsp = self.nullspace()
        with b.vec as v:
            nullsp.remove(v)
        self._ad_orthogonalized = True

    @PETSc.Log.EventDecorator()
    def check_orthogonality(self, orthonormal=True):
        r"""Check if the basis is orthogonal.

        :arg orthonormal: If True check that the basis is also orthonormal.

        :raises ValueError: If the basis is not orthogonal/orthonormal.
        """
        eps = numpy.sqrt(numpy.finfo(PETSc.ScalarType).eps)
        basis = self._petsc_vecs
        if orthonormal:
            for i, v in enumerate(basis):
                norm = v.norm()
                if abs(norm - 1.0) > eps:
                    raise ValueError("Basis vector %d has norm %g", i, norm)
        if self._constant:
            for i, v in enumerate(basis):
                dot = v.sum()
                if abs(dot) > eps:
                    raise ValueError("Basis vector %d is not orthogonal to constant"
                                     " inner product is %g", i, abs(dot))
        for i, v in enumerate(basis[:-1]):
            for j, v_ in enumerate(basis[i+1:]):
                dot = v.dot(v_)
                if abs(dot) > eps:
                    raise ValueError("Basis vector %d not orthogonal to %d"
                                     " inner product is %g", i, i+j+1, abs(dot))

    def is_orthonormal(self):
        r"""Is this vector space basis orthonormal?"""
        try:
            self.check_orthogonality(orthonormal=True)
            return True
        except ValueError:
            return False

    def is_orthogonal(self):
        r"""Is this vector space basis orthogonal?"""
        try:
            self.check_orthogonality(orthonormal=False)
            return True
        except ValueError:
            return False

    def _apply(self, matrix, transpose=False, near=False):
        r"""Set this VectorSpaceBasis as a nullspace for a matrix

        :arg matrix: a :class:`~.MatrixBase` whose nullspace should
             be set.
        :kwarg transpose: Should this be set as the transpose
             nullspace instead?  Used to orthogonalize the right hand
             side wrt the provided nullspace.
        """
        if not isinstance(matrix, MatrixBase):
            return
        if near:
            if transpose:
                raise RuntimeError("No MatSetTransposeNearNullSpace operation in PETSc.")
            else:
                matrix.petscmat.setNearNullSpace(self.nullspace())
        else:
            if transpose:
                matrix.petscmat.setTransposeNullSpace(self.nullspace())
            else:
                matrix.petscmat.setNullSpace(self.nullspace())

    def __iter__(self):
        r"""Yield self when iterated over"""
        yield self


class MixedVectorSpaceBasis(object):
    r"""A basis for a mixed vector space

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
        self.comm = function_space.comm
        self._comm = internal_comm(self.comm, self)
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
        r"""Build a basis for the complete mixed space.

        The monolithic basis is formed by the cartesian product of the
        bases forming each sub part.
        """
        self._vecs = []
        for idx, basis in enumerate(self):
            if isinstance(basis, VectorSpaceBasis):
                vecs = basis._vecs
                if basis._constant:
                    vecs = vecs + (function.Function(self._function_space[idx]).assign(1), )
                for vec in vecs:
                    mvec = function.Function(self._function_space)
                    mvec.sub(idx).assign(vec)
                    self._vecs.append(mvec)

        self._petsc_vecs = []
        for v in self._vecs:
            with v.vec_ro as v_:
                self._petsc_vecs.append(v_)

        # orthonormalize:
        basis = self._petsc_vecs
        for i, vec in enumerate(basis):
            alphas = []
            for vec_ in basis[:i]:
                alphas.append(vec.dot(vec_))
            for alpha, vec_ in zip(alphas, basis[:i]):
                vec.axpy(-alpha, vec_)
            vec.normalize()

        self._nullspace = PETSc.NullSpace().create(constant=False,
                                                   vectors=self._petsc_vecs,
                                                   comm=self._comm)

    def _apply_monolithic(self, matrix, transpose=False, near=False):
        r"""Set this class:`MixedVectorSpaceBasis` as a nullspace for a
        matrix.

        :arg matrix: a :class:`~.MatrixBase` whose nullspace should
             be set.

        :kwarg transpose: Should this be set as the transpose
             nullspace instead?  Used to orthogonalize the right hand
             side wrt the provided nullspace.
        :kwarg near: Should this be set as the near nullspace instead?
             Incompatible with transpose=True.

        Note, this only hangs the nullspace on the Mat, you should
        normally be using :meth:`_apply` which also hangs the
        nullspace on the appropriate fieldsplit ISes for Schur
        complements."""
        if self._nullspace is None:
            self._build_monolithic_basis()
        if near:
            if transpose:
                raise RuntimeError("No MatSetTransposeNearNullSpace operation in PETSc.")
            else:
                matrix.petscmat.setNearNullSpace(self._nullspace)
        else:
            if transpose:
                matrix.petscmat.setTransposeNullSpace(self._nullspace)
            else:
                matrix.petscmat.setNullSpace(self._nullspace)

    def _apply(self, matrix_or_ises, transpose=False, near=False):
        r"""Set this :class:`MixedVectorSpaceBasis` as a nullspace for a matrix

        :arg matrix_or_ises: either a :class:`~.MatrixBase` to set a
             nullspace on, or else a list of PETSc ISes to compose a
             nullspace with.
        :kwarg transpose: Should this be set as the transpose
             nullspace instead?  Used to orthogonalize the right hand
             side wrt the provided nullspace.
        :kwarg near: Should this be set as the near nullspace instead?
             Incompatible with transpose=True.

        .. note::

           If you're using a Schur complement preconditioner you
           should both call :meth:`_apply` on the matrix, and the ises
           defining the splits.

           If transpose is ``True``, nothing happens in the IS case,
           since PETSc does not provide the ability to set anything.
        """
        if isinstance(matrix_or_ises, MatrixBase):
            matrix = matrix_or_ises
            rows, cols = matrix.block_shape
            if rows != cols:
                raise RuntimeError("Can only apply nullspace to square operator")
            if rows != len(self):
                raise RuntimeError("Shape of matrix (%d, %d) does not match size of nullspace %d" %
                                   (rows, cols, len(self)))
            # Hang the expanded nullspace on the big matrix
            self._apply_monolithic(matrix, transpose=transpose, near=near)
            return
        ises = matrix_or_ises
        if transpose:
            # PETSc doesn't give us anything here
            return

        key = "nearnullspace" if near else "nullspace"
        for i, basis in enumerate(self):
            if not isinstance(basis, VectorSpaceBasis):
                continue
            # Compose appropriate nullspace with IS for schur complement
            if ises is not None:
                is_ = ises[i]
                is_.compose(key, basis.nullspace())

    def __iter__(self):
        r"""Yield the individual bases making up this MixedVectorSpaceBasis"""
        for basis in self._bases:
            yield basis

    def __len__(self):
        r"""The number of bases in this MixedVectorSpaceBasis"""
        return len(self._bases)

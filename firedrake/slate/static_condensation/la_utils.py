from collections import namedtuple
from pyop3.pyop2_utils import as_tuple

from firedrake.parameters import parameters
from firedrake.slate.slate import Block, DiagonalTensor, Tensor, AssembledVector
from firedrake.petsc import PETSc


LAContext = namedtuple("LAContext",
                       ["lhs", "rhs", "field_idx"])
LAContext.__doc__ = """\
Context information for systems of equations after
applying algebraic transformation via Slate-supported
operations. This object provides the symbolic expressions
for the transformed linear system of equations.

:param lhs: The resulting expression for the transformed
            left-hand side matrix.
:param rhs: The resulting expression for the transformed
            right-hand side vector.
:param field_idx: An integer or iterable of integers
                  (if the system is mixed) denoting
                  which field(s) the resulting solution
                  is defined on.
"""


def condense_and_forward_eliminate(A, b, elim_fields, prefix, pc):
    """Return the Slate expressions of the statically condensed system.

    Parameters
    ----------
    A : firedrake.slate.slate.Tensor
        The operator on the mixed space.
    b : firedrake.cofunction.Cofunction
        The right-hand side.
    elim_fields : collections.abc.Sequence of int
        The indices of the subspaces to eliminate.
    prefix : str
        The options prefix of the condensed system.
    pc : petsc4py.PETSc.PC
        The preconditioner of the mixed system.

    Returns
    -------
    LAContext
        The Schur complement on the retained subspaces and the contribution
        of the eliminated subspaces to its right-hand side.
    SchurComplementBuilder
        The builder of the local inverses.

    Notes
    -----
    By default, a nested local Schur complement puts the first subspace of
    ``elim_fields`` in ``fieldsplit_0`` and the other subspaces in
    ``fieldsplit_1``.
    """
    if not isinstance(A, Tensor):
        raise ValueError("Left-hand operator must be a Slate Tensor")

    e = tuple(elim_fields)
    c = tuple(i for i in range(len(A.arg_function_spaces[0])) if i not in e)

    A_ee = Block(A, (e, e))
    A_ec = Block(A, (e, c))
    A_ce = Block(A, (c, e))
    A_cc = Block(A, (c, c))
    b_e = AssembledVector(b.sub(e))

    schur_builder = SchurComplementBuilder(prefix, A_ee, A_ce, A_ec, pc, 0, tuple(range(1, len(e))),
                                           non_zero_saddle_mat=A_cc)
    r, S = schur_builder.build_schur(b_e)
    return LAContext(lhs=S, rhs=r, field_idx=c), schur_builder


def backward_solve(b, x, schur_builder, reconstruct_fields):
    """Return the Slate expressions that recover the eliminated subspaces.

    Parameters
    ----------
    b : firedrake.cofunction.Cofunction
        The right-hand side.
    x : firedrake.function.Function
        The solution, whose retained subspaces are known.
    schur_builder : SchurComplementBuilder
        The builder of the local inverses.
    reconstruct_fields : collections.abc.Sequence of int
        The indices of the eliminated subspaces in the space of ``x``.

    Returns
    -------
    list of LAContext
        The local systems, in the order in which they are solved.

    Notes
    -----
    The eliminated subspaces solve :math:`A_{ee} x_e = b_e - A_{ec} x_c`,
    where :math:`A_{ee}` is ``schur_builder.Atilde`` and :math:`A_{ec}` is
    ``schur_builder.KT``. A nested local Schur complement solves for
    ``fieldsplit_1`` first, and then for ``fieldsplit_0``.
    """
    e = tuple(reconstruct_fields)
    c = tuple(i for i in range(len(x.function_space())) if i not in e)

    r = AssembledVector(b.sub(e)) - schur_builder.KT * AssembledVector(x.sub(c))
    if not schur_builder.nested:
        return [LAContext(lhs=schur_builder.Atilde_inv_hat, rhs=r, field_idx=e)]

    # Solve S x_1 = r_1 - A_10 A_00^{-1} r_0, and then A_00 x_0 = r_0 - A_01 x_1.
    id0, id1 = map(as_tuple, (schur_builder.vidx, schur_builder.pidx))
    e0 = tuple(e[i] for i in id0)
    e1 = tuple(e[i] for i in id1)
    _, A01, A10, _ = schur_builder.list_split_mixed_ops
    r0 = Block(r, (id0,))
    r1 = Block(r, (id1,))
    x1 = AssembledVector(x.sub(e1))
    return [LAContext(lhs=schur_builder.inner_S_inv_hat, rhs=r1 - A10 * schur_builder.A00_inv_hat * r0, field_idx=e1),
            LAContext(lhs=schur_builder.A00_inv_hat, rhs=r0 - A01 * x1, field_idx=e0)]


class SchurComplementBuilder(object):

    """A Slate-based Schur complement expression builder. The expression is
    used in the trace system solve and parts of it in the reconstruction
    calls of the other two variables of the hybridised system.
    How the Schur complement if constructed, and in particular how the local inverse of the
    mixed matrix is built, is controlled with PETSc options. All corresponding PETSc options
    start with ``hybridization_localsolve``.
    The following option sets are valid together with the usual set of hybridisation options:

    .. code-block:: text

        {'localsolve': {'ksp_type': 'preonly',
                        'pc_type': 'fieldsplit',
                        'pc_fieldsplit_type': 'schur'}}

    A Schur complement is requested for the mixed matrix inverse which appears inside the
    Schur complement of the trace system solve. The Schur complements are then nested.
    For details see defition of :meth:`build_schur`. No fieldsplit options are set so all
    local inverses are calculated explicitly.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                       'pc_type': 'fieldsplit',
                       'pc_fieldsplit_type': 'schur',
                       'fieldsplit_1': {'ksp_type': 'default',
                                        'pc_type': 'python',
                                        'pc_python_type': __name__ + '.DGLaplacian'}}

    The inverse of the Schur complement inside the Schur decomposition of the mixed matrix inverse
    is approximated by a default solver (LU in the matrix-explicit case) which is preconditioned
    by a user-defined operator, e.g. a DG Laplacian, see :meth:`build_inner_S_inv`.
    So :math:`P_S * S * x = P_S * b`.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                        'pc_type': 'fieldsplit',
                        'pc_fieldsplit_type': 'schur',
                        'fieldsplit_1': {'ksp_type': 'default',
                                        'pc_type': 'python',
                                        'pc_python_type': __name__ + '.DGLaplacian',
                                        'aux_ksp_type': 'preonly'}
                                        'aux_pc_type': 'jacobi'}}}}

    The inverse of the Schur complement inside the Schur decomposition of the mixed matrix inverse
    is approximated by a default solver (LU in the matrix-explicit case) which is preconditioned
    by a user-defined operator, e.g. a DG Laplacian. The inverse of the preconditioning matrix is
    approximated through the inverse of only the diagonal of the provided operator, see
    :meth:`build_Sapprox_inv`. So :math:`diag(P_S).inv * S * x = diag(P_S).inv * b`.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                       'pc_type': 'fieldsplit',
                       'pc_fieldsplit_type': 'schur',
                       'fieldsplit_0': {'ksp_type': 'default',
                                        'pc_type': 'jacobi'}

    The inverse of the :math:`A_{00}` block of the mixed matrix is approximated by a default solver
    (LU in the matrix-explicit case) which is preconditioned by the diagonal matrix of :math:`A_{00},
    see :meth:`build_A00_inv`. So :math:`diag(A_{00}).inv * A_{00} * x = diag(A_{00}).inv * b`.

    .. code-block:: text

        'localsolve': {'ksp_type': 'preonly',
                       'pc_type': 'fieldsplit',
                       'pc_fieldsplit_type': 'None',
                       'fieldsplit_0':  ...
                       'fieldsplit_1':  ...

    All the options for ``fieldsplit_`` are still valid if ``'pc_fieldsplit_type': 'None'.`` In this case
    the mixed matrix inverse which appears inside the Schur complement of the trace system solve
    is calculated explicitly, but the local inverses of :math:`A_{00}` and the Schur complement
    in the reconstructions calls are still treated according to the options in ``fieldsplit_``.

    """

    def __init__(self, prefix, Atilde, K, KT, pc, vidx, pidx, non_zero_saddle_mat=None):
        # set options, operators and order of sub-operators
        self.Atilde = Atilde
        self.K = K
        self.KT = KT

        # prefixes
        self.prefix = prefix + "localsolve_"

        # Get the subspaces of Atilde in each split of the nested Schur complement.
        opts = PETSc.Options(self.prefix)
        self.vidx = tuple(map(int, opts.getIntArray("pc_fieldsplit_0_fields", as_tuple(vidx))))
        self.pidx = tuple(map(int, opts.getIntArray("pc_fieldsplit_1_fields", as_tuple(pidx))))
        self._retrieve_options(pc)

        self.non_zero_saddle_mat = non_zero_saddle_mat

        # Check if Atilde is mixed
        all_fields = list(range(len(Atilde.arg_function_spaces[0])))
        self.nfields = len(all_fields)
        self._split_mixed_operator()

        # build all inverses
        P = DiagonalTensor(self.Atilde).inv
        self.Atilde_inv_hat = self.inv(self.Atilde, P, self.jacobi_A00, self.preonly_A00)
        self.A00_inv_hat = self.build_A00_inv()
        if self.nfields > 1:
            self.schur_approx = self.retrieve_user_S_approx(pc, self.schur_approx) if self.schur_approx else None
            self.inner_S = self.build_inner_S()
            self.inner_S_approx_inv_hat = self.build_Sapprox_inv()
            self.inner_S_inv_hat = self.build_inner_S_inv()

    def _split_mixed_operator(self):
        if self.nfields > 1:
            id0, id1 = map(as_tuple, (self.vidx, self.pidx))
            A00 = Block(self.Atilde, (id0, id0))
        else:
            A00 = self.Atilde
        self.list_split_mixed_ops = [A00, None, None, None]

        if self.nfields > 1:
            A01 = Block(self.Atilde, (id0, id1))
            A10 = Block(self.Atilde, (id1, id0))
            A11 = Block(self.Atilde, (id1, id1))
            self.list_split_mixed_ops = [A00, A01, A10, A11]

            trace_fields = tuple(range(len(self.K.arg_function_spaces[0])))
            K0 = Block(self.K, (trace_fields, id0))
            K1 = Block(self.K, (trace_fields, id1))
            self.list_split_trace_ops = [K0, K1]

            K0 = Block(self.KT, (id0, trace_fields))
            K1 = Block(self.KT, (id1, trace_fields))
            self.list_split_trace_ops_transpose = [K0, K1]

    def _check_options(self, valid_options):
        opts = PETSc.Options(self.prefix)
        for key, supported in valid_options:
            try:
                value = opts.getString(key)
            except KeyError:
                continue

            if value not in supported:
                raise ValueError(f"Unsupported value ({value}) for '{self.prefix + key}'. "
                                 f"Should be one of {supported}")

    def _retrieve_options(self, pc):
        get_option = lambda key: PETSc.Options(self.prefix).getString(key, default="")

        # Get options for Schur complement decomposition
        self._check_options([("ksp_type", {"preonly"}), ("pc_type", {"fieldsplit"}), ("pc_fieldsplit_type", {"schur"})])
        self.nested = (get_option("ksp_type") == "preonly"
                       and get_option("pc_type") == "fieldsplit"
                       and get_option("pc_fieldsplit_type") == "schur")

        # Get preconditioning options for A00
        fs0, fs1 = ("fieldsplit_0", "fieldsplit_1")
        self._check_options([(fs0+"ksp_type", {"preonly", "default"}), (fs0+"pc_type", {"jacobi"})])
        self.preonly_A00 = get_option(fs0+"_ksp_type") == "preonly"
        self.jacobi_A00 = get_option(fs0+"_pc_type") == "jacobi"

        # Get preconditioning options for the Schur complement
        self._check_options([(fs1+"ksp_type", {"preonly", "default"}), (fs1+"pc_type", {"jacobi", "python"})])
        self.preonly_S = get_option(fs1+"_ksp_type") == "preonly"
        self.jacobi_S = get_option(fs1+"_pc_type") == "jacobi"

        # Get user supplied operator and its options
        self.schur_approx = (get_option(fs1+"_pc_python_type") if get_option(fs1+"_pc_type") == "python" else False)
        self._check_options([(fs1+"aux_ksp_type", {"preonly", "default"}), (fs1+"aux_pc_type", {"jacobi"})])
        self.preonly_Shat = get_option(fs1+"_aux_ksp_type") == "preonly"
        self.jacobi_Shat = get_option(fs1+"_aux_pc_type") == "jacobi"

        if self.jacobi_Shat or self.jacobi_A00:
            assert parameters["slate_compiler"]["optimise"], ("Local systems should only get preconditioned with "
                                                              "a preconditioning matrix if the Slate optimiser replaces "
                                                              "inverses by solves.")

    def build_inner_S(self):
        """Build the inner Schur complement."""
        _, A01, A10, A11 = self.list_split_mixed_ops
        return A11 - A10 * self.A00_inv_hat * A01

    def inv(self, A, P, prec, preonly=False):
        """ Calculates the inverse of an operator A.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with the preconditioner P
            if prec is True.
            The inverse of A may be just approximated with the inverse of P
            if prec and replace.
        """
        return (P if prec and preonly else
                (P*A).inv * P if prec else
                A.inv)

    def build_inner_S_inv(self):
        """ Calculates the inverse of the schur complement.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with the preconditioner P.
        """
        A = self.inner_S
        P = self.inner_S_approx_inv_hat
        prec = bool(self.schur_approx) or self.jacobi_S
        return self.inv(A, P, prec, self.preonly_S)

    def build_Sapprox_inv(self):
        """ Calculates the inverse of preconditioner to the Schur complement,
            which can be either the schur complement approximation provided by the user
            or jacobi.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with jacobi.
        """
        prec = (bool(self.schur_approx) and self.jacobi_Shat) or self.jacobi_S
        A = self.schur_approx if self.schur_approx else self.inner_S
        P = DiagonalTensor(A).inv
        preonly = self.preonly_Shat if self.schur_approx else True
        return self.inv(A, P, prec, preonly)

    def build_A00_inv(self):
        """ Calculates the inverse of :math:`A_{00}`, the (0,0)-block of the mixed matrix Atilde.
            The inverse is potentially approximated through a solve
            which is potentially preconditioned with jacobi.
        """
        A, _, _, _ = self.list_split_mixed_ops
        P = DiagonalTensor(A).inv
        return self.inv(A, P, self.jacobi_A00, self.preonly_A00)

    def retrieve_user_S_approx(self, pc, usercode):
        """Retrieve a user-defined :class:firedrake.preconditioners.AuxiliaryOperator from the PETSc Options,
        which is an approximation to the Schur complement and its inverse is used
        to precondition the local solve in the reconstruction calls (e.g.).
        """
        _, _, _, A11 = self.list_split_mixed_ops
        test, trial = A11.arguments()
        if usercode != "":
            (modname, funname) = usercode.rsplit('.', 1)
            mod = __import__(modname)
            fun = getattr(mod, funname)
            if isinstance(fun, type):
                fun = fun()
            return Tensor(fun.form(pc, test, trial)[0])
        else:
            return None

    def build_schur(self, rhs):
        """Construct the Schur complement and the contribution of the eliminated fields to its right-hand side.

        Parameters
        ----------
        rhs : AssembledVector or Block
            The residual on the fields that will be eliminated.

        Returns
        -------
        schur_rhs : TensorBase
            The contribution of the eliminated fields to the right-hand side
            of the reduced system. For a system with a nonzero diagonal block,
            the right-hand side of the reduced system is the residual on the
            retained fields minus this contribution.
        schur_comp : TensorBase
            The operator of the reduced system.

        Notes
        -----
        The Schur complement in the operators of the trace solve contains
        the inverse on a mixed system.  Users may want this inverse to be treated
        with another Schur complement.

        Let the mixed matrix Atilde be called A here.
        Then, if a nested schur complement is requested, the inverse of Atilde
        is rewritten with help of a a Schur decomposition as follows.

        .. code-block:: text

                A.inv=[[I, -A00.inv * A01] * [[A00.inv, 0    ] * [[I,            0]
                      [0,  I             ]]  [0,        S.inv]]  [-A10* A00.inv, I]]
                      --------------------   -----------------   ------------------
                             block1                block2              block3
                with the (inner) schur complement S = A11 - A10 * A00.inv * A01
        """

        if self.nested:
            _, A01, A10, _ = self.list_split_mixed_ops
            K0, K1 = self.list_split_trace_ops
            KT0, KT1 = self.list_split_trace_ops_transpose
            R = [Block(rhs, (as_tuple(self.vidx),)),
                 Block(rhs, (as_tuple(self.pidx),))]
            # K * block1
            K_Ainv_block1 = [K0, -K0 * self.A00_inv_hat * A01 + K1]
            # K * block1 * block2
            K_Ainv_block2 = [K_Ainv_block1[0] * self.A00_inv_hat,
                             K_Ainv_block1[1] * self.inner_S_inv_hat]
            # K * block1 * block2 * block3
            K_Ainv_block3 = [K_Ainv_block2[0] - K_Ainv_block2[1] * A10 * self.A00_inv_hat,
                             K_Ainv_block2[1]]
            # K * block1 * block2 * block3 * broken residual
            schur_rhs = (K_Ainv_block3[0] * R[0] + K_Ainv_block3[1] * R[1])
            # K * block1 * block2 * block3 * K.T
            schur_comp = K_Ainv_block3[0] * KT0 + K_Ainv_block3[1] * KT1
        else:
            schur_rhs = self.K * self.Atilde_inv_hat * rhs
            schur_comp = self.K * self.Atilde_inv_hat * self.KT
        if self.non_zero_saddle_mat:
            # problem is not a saddle point problem
            schur_comp = self.non_zero_saddle_mat - schur_comp
        return schur_rhs, schur_comp
